/* Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * exec_hag.h -- C ABI over HAG's experimental bounded compute queue.
 *
 * This header lets non-Rust applications use HAG through an explicit C ABI.
 * It adds nothing to the queue: every bound, refusal and poisoning rule of
 * the Rust interface applies unchanged.
 *
 * Link against libexec_hag.so or libexec_hag.a, built with:
 *     cargo build --release --features capi
 *
 * OWNERSHIP
 *   The caller provides every buffer and frees nothing HAG allocated. The one
 *   opaque handle is released by hag_queue_close and by nothing else; it must
 *   not be closed twice or used afterwards.
 *
 * POINTERS AND THREADS
 *   Serialize every call on the same handle, including queries and close:
 *   calls must never overlap. Distinct handles may be used independently.
 *   Every pointer must be aligned for its declared type. Every array or byte
 *   region must lie within one live allocation and remain valid for the whole
 *   call. Readable regions must not be modified during the call. Writable
 *   regions require exclusive access and must not overlap other arguments or
 *   the opaque handle. Except where expressly documented, null pointers are
 *   invalid. These are caller obligations; HAG cannot validate raw pointers.
 *
 * PANICS
 *   No Rust unwind crosses this boundary. A caught unwinding panic returns
 *   HAG_PANIC and poisons a surviving handle. Close consumes the handle even
 *   on failure. Abort-on-panic builds and process aborts cannot be recovered.
 *
 * TRUST
 *   Programs are trusted, audited GFX1151 wave64 ISA. Reaching HAG from
 *   another language does not relax that contract. Never pass untrusted model
 *   or user bytes as shader code; a faulting program can affect other GPU
 *   clients on the system.
 */
#ifndef EXEC_HAG_H
#define EXEC_HAG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. Negative values are failures. */
#define HAG_OK                 0
#define HAG_INVALID_ARGUMENT  (-1)  /* null pointer, zero where disallowed,
                                     * or a path that is not a render node   */
#define HAG_REFUSED           (-2)  /* out of bounds, bad geometry, bad ABI   */
#define HAG_DEVICE            (-3)  /* driver or hardware rejected it         */
#define HAG_POISONED          (-4)  /* queue will refuse every further call   */
#define HAG_PANIC             (-5)  /* panic caught at the boundary           */
#define HAG_BUDGET            (-6)  /* allocation exceeded the budget         */
#define HAG_GUARD_CORRUPTION  (-7)  /* something wrote outside its region     */
#define HAG_INCOMPLETE        (-8)  /* no completion by the fence deadline    */

/* Opaque queue handle. */
typedef struct hag_queue hag_queue;

/* A borrowed, static description of a status code. Never null, never freed. */
const char *hag_error_text(int32_t code);

/* The largest workgroup count hag_queue_open accepts, and the count a queue
 * uses when max_groups is zero. The default of 64 suits a kernel covering at
 * most 4096 elements; it is far too low for an image, because a dispatch
 * costs about the same whether it covers 64 elements or 4000 -- the cost is
 * the submit and the fence, not the shading -- so tiling an image multiplies
 * the fixed cost by the number of tiles. */
uint32_t hag_max_dispatch_groups(void);
uint32_t hag_default_dispatch_groups(void);

/* The per-wave VGPR allocation range hag_queue_open admits. The qualified ABI
 * uses the minimum; a library may ask for more, which leaves every other
 * library's resource descriptor untouched. Must be a multiple of four. */
uint32_t hag_min_kernel_vgprs(void);
uint32_t hag_max_kernel_vgprs(void);

/* Open a queue over an immutable program library.
 *
 *   device_path    NUL-terminated DRM render node path
 *   programs       program_count pointers to ISA payloads
 *   program_sizes  matching byte lengths
 *   program_count  1 through 32; each payload is a nonzero multiple of four
 *                  bytes, at most 4096 bytes
 *   slot_sizes     three data-slot logical capacities, in bytes
 *   vgprs          per-wave register allocation; see hag_min_kernel_vgprs
 *   max_groups     largest workgroup count a dispatch may launch, or 0 for
 *                  the qualified default
 *   budget_bytes   allocation ceiling, or 0 for unlimited
 *   out            receives an owned handle on success
 *
 * Input storage may be released after this function returns. Oversized
 * program_count is refused before accessing the pointer arrays; invalid
 * program lengths are refused before accessing those payloads.
 * Returns HAG_OK, or a negative status. On failure *out is untouched; callers
 * may initialize it to NULL before calling. */
int32_t hag_queue_open(const char *device_path,
                       const uint8_t *const *programs,
                       const uint32_t *program_sizes,
                       uint32_t program_count,
                       const uint32_t *slot_sizes,
                       uint32_t vgprs,
                       uint32_t max_groups,
                       uint64_t budget_bytes,
                       hag_queue **out);

/* Copy bytes into, or out of, a data slot at a byte offset. Zero length is
 * invalid. Lengths above 4 GiB minus 4 KiB, or the host's addressable Rust slice
 * limit (SIZE_MAX and INTPTR_MAX), are refused before accessing bytes or queue.
 * For admitted lengths, bytes must cover length readable/writable bytes. */
int32_t hag_queue_write(hag_queue *queue, uint32_t slot, uint64_t offset,
                        const uint8_t *bytes, uint64_t length);
int32_t hag_queue_read(hag_queue *queue, uint32_t slot, uint64_t offset,
                       uint8_t *bytes, uint64_t length);

/* Submit one program and wait for completion.
 *
 *   program   index into the library given at open
 *   offsets   three per-slot byte offsets
 *   lengths   three per-slot byte lengths the program may address
 *   scalars   two scalar arguments
 *   groups    workgroup count
 *
 * The selected trusted program must terminate and obey its descriptor bounds,
 * alignment and indexing requirements for every argument it is given. */
int32_t hag_queue_dispatch(hag_queue *queue, uint32_t program,
                           const uint64_t *offsets, const uint32_t *lengths,
                           const uint32_t *scalars, uint32_t groups);

/* Total page-rounded allocation charge, including program, IB and guards. */
uint64_t hag_queue_allocation_bytes(const hag_queue *queue);
/* Remaining budget in bytes; 0 for a null handle. An unlimited budget reports
 * UINT64_MAX minus its live allocation charge. */
uint64_t hag_queue_budget_remaining(const hag_queue *queue);
/* Nonzero when the queue is poisoned. A null handle reports poisoned. */
int32_t hag_queue_is_poisoned(const hag_queue *queue);

/* Close and release the handle. Returns HAG_OK on a clean shutdown; a
 * poisoned queue is still released and reports its status. The handle is
 * consumed even on failure, including HAG_PANIC, and must never be reused. */
int32_t hag_queue_close(hag_queue *queue);

#ifdef __cplusplus
}
#endif

#endif /* EXEC_HAG_H */
