# Native GFX1151 packed matvec and elementwise proof kernels

These are original, Apache-2.0-licensed, bounded serial correctness kernels, not
production throughput kernels. They retain packed BF16 or GGML Q8_0 weights and
widen only the current element inside the shader. No expanded checkpoint, Vulkan,
HIP, ROCm runtime, or runtime shader compiler is involved.

## Shared ABI

Most dispatches use `ceil(rows / 64)` or `ceil(elements / 64)` workgroups, each with
64 threads in wave64 mode. Attention softmax uses one group and striped output
positions, as specified below. `s_version 0x2006` matches GFX11 wave64; the host must
select wave64 as well. Each lane handles global index `group_x * 64 + local_x`.
Matvec admission requires `1 <= rows <= 4096`, `1 <= cols <= 4096`, exact buffer
lengths, and complete 32-element blocks for Q8_0. The shader refuses invalid
geometry and group IDs above 63 before accessing memory. The final group's
inactive lanes are masked by the global row/element count before any load/store.
`dispatch_groups()` refuses zero and out-of-range counts; an empty semantic
operation must remain an adapter no-op rather than dispatching a zero-sized grid.

The immutable program library has eleven entries: BF16 (adapter ID 0), Q8_0 (1),
SwiGLU (2), scaled-add (3), unweighted RMSNorm (4), pointwise multiply (5),
RoPE (6), attention scores (7), attention softmax (8), attention values (9),
and Q8_0 row decode (10).
Each includes its own 256-byte instruction-fetch padding. Their payload lengths
are 484, 556, 568, 412, 540, 396, 548, 528, 836, 528 and 456 bytes; packing at
256-byte-aligned offsets 0, 512, 1280, 2048, 2560, 3328, 3840, 4608, 5376,
6400 and 7168 ends at byte 7624. The page-rounded shader allocation remains
8192 bytes.
`program_bytes()` defines the common program-ID order
used for both queue creation and allocation cost. Adding RoPE leaves the prior
six payloads and entry offsets byte-identical; appending attention also leaves
RoPE unchanged. The historical row-decoder addition preserved all ten preceding
payloads and offsets. The C512 attention extension changed programs 7–9 and grew
softmax's output loop, moving the value/row entry offsets by 256 bytes while
preserving the other eight payloads. Its selected-model live qualification is
recorded in the [C512 result](../../../../../docs/hag-editor-context.md#c512-native-qualification-and-serving-cutover).
The additive C2048 extension changes only the admission guards in programs 7–9;
their arithmetic, byte lengths, entry offsets and the other eight payloads remain
unchanged. The separate C2048 attention boundary fixtures passed their supervised
GPU gate; see the [finite qualification and scope](../../../../../docs/hag-editor-context.md#qualified-c2048-attention-boundaries).
Full-model C2048 remains unqualified. Dated evidence below keeps its original
frozen shader identities and limits; raising the backend ceiling does not extend
those results or change the C512 serving default.
HAG's `shader_library_bytes()` admits
1..16 programs, each at most 4096 bytes, with total page-rounded storage at most
16 KiB; it reports the actual charge without reserving memory or validating ISA.
The RMSNorm qualification below used the six-program, 4096-byte shader allocation.
Each dated result retains its frozen payloads and allocation charge. The isolated
RoPE gate executed across the first page boundary in the expanded allocation.
Attention then passed its isolated fixtures and the real layer-0 attention block;
see the scoped qualification below.

The row decoder and selected real-checkpoint vocabulary component passed their
separate supervised gates recorded below. The prior RoPE/attention results retain
their frozen ten-program scope. The row-decoder milestone's offline run passed 143
backend tests, two embedding-diagnostic tests, and the explicit LLVM assembly,
extraction and disassembly comparison for all eleven embedded payloads. These
checks require neither a model nor GPU execution; live evidence remains scoped
to the recorded fixtures.

### Packed matvec bindings

| Registers | Meaning |
| --- | --- |
| `s[0:3]` | Packed weights, raw byte-addressed SRD |
| `s[4:7]` | F32 input vector, raw byte-addressed SRD |
| `s[8:11]` | F32 output vector, raw byte-addressed SRD |
| `s12`, `s13` | Columns, rows |
| `s14` | System-supplied workgroup X ID, 0..63 |
| `s15..s19` | Shader scratch; no host initialization |
| `v0` | Packed local work-item ID on entry; then global row index |
| `v1..v6` | Shader scratch |

### Q8_0 row-decode bindings

| Registers | Meaning |
| --- | --- |
| `s[0:3]` | Selected packed row, exactly `(columns / 32) * 34` bytes |
| `s[4:7]` | Private retained four-byte dummy; never read by the shader |
| `s[8:11]` | Write-only F32 output, exactly `columns * 4` bytes |
| `s12`, `s13` | Columns, constant 1 |
| `s14`, `s15` | Workgroup X ID, divisibility scratch |
| `v0`, `v1` | Global column index, output byte offset |
| `v2`, `v3` | Packed block base, quant byte offset before adding 2 |
| `v4`, `v5` | F32 scale, signed quant then F32 product |

The owned `Command::EmbeddingRow` selects a checked row of a prepared Q8_0 tile.
The adapter validates its owner, format, row index and exact output width before
recording, then binds only that row's resident packed extent and retains a private
four-byte dummy. No token ID or table stride is interpreted by the shader.
Columns must be a multiple of 32 in 32..4096, `s13` must equal one, and the grid
must contain `ceil(columns / 64)` workgroups, at most 64. The shader rejects
invalid geometry or group IDs above 63 and masks inactive columns before any
memory access. All three
physical descriptor ranges must be disjoint. Packed row starts and F16 scales
are two-byte aligned; a row of 96 columns has a 102-byte stride, so row starts
must not be rounded to four-byte boundaries. Dummy/output addresses are
four-byte aligned. Packed row descriptors reach at most 4,352 bytes and output
descriptors at most 16,384 bytes. Recorded batches and submissions retain the
selected prepared weight, output and dummy through completion; replay performs
one physical dispatch without uploading or decoding the row on the host.

Column `i` loads its F16 scale at `(i >> 5) * 34` and signed byte at that address
plus `2 + (i & 31)`. Both loads complete before exact widening to F32 and one
separate `scale * quant` multiplication. The output is stored at `4 * i` and
stores drain before termination. This uses six VGPRs within the fixed eight-VGPR
ABI, with no reduction, FMA, host dequantization or expanded resident checkpoint.
Hardware-free tests trace the encoded offsets and masked tails, compare CPU
decoding at 32/96/4096 columns with changed packed bytes, and cover every signed
byte with signed-zero and F16 subnormal scales. Those software checks do not
establish hardware behavior for arbitrary values.

### Elementwise bindings

| Registers | SwiGLU | Scaled-add |
| --- | --- | --- |
| `s[0:3]` | One F32 finite clamp limit | One F32 finite scale |
| `s[4:7]` | F32 up vector | F32 input/scratch vector |
| `s[8:11]` | Read-write F32 gate vector | Read-write F32 residual vector |
| `s12`, `s13` | Element count, constant 1 | Element count, constant 1 |
| `s14` | System workgroup X ID, 0..63 | System workgroup X ID, 0..63 |
| `v0..v7` | Lane, byte offset, scratch | Lane, byte offset, scratch |

Elementwise host admission requires `1 <= count <= 4096`, scalar 1 in `s13`,
`ceil(count / 64)` groups, four-byte-aligned addresses, an exact four-byte parameter descriptor and
exact `count * 4` byte input/output descriptors. The three descriptor extents
must be disjoint. Reading and writing the destination through SRD2 is intentional;
aliasing SRD1 with SRD2 is not permitted. All parameter and vector loads complete
before arithmetic, and stores complete before termination. No scratch/LDS, atomics,
barriers or data-dependent memory addressing are used.

### RMSNorm and pointwise-multiply bindings

| Registers | Unweighted RMSNorm | Pointwise multiply |
| --- | --- | --- |
| `s[0:3]` | One F32 finite nonnegative epsilon | Resident F32 weight vector |
| `s[4:7]` | Resident F32 input vector | Resident F32 normalized vector |
| `s[8:11]` | Write-only F32 normalized vector | Write-only F32 weighted vector |
| `s12`, `s13` | Element count, constant 1 | Element count, constant 1 |
| `s14` | System workgroup X ID, 0..63 | System workgroup X ID, 0..63 |
| `s15`, `s16` | Reduction element index, byte offset | Unused |
| `v0`, `v1` | Global element index, output byte offset | Global index, byte offset |
| `v2:v3` | F64 sum, then inverse; finally F32 inverse | Loaded weight, input/result |
| `v4:v5`, `v6` | F64 temporary, F32 load | Unused |

Both programs require `1 <= count <= 4096`, scalars `[count, 1]`, and
`ceil(count / 64)` groups. All addresses are four-byte aligned and all three
descriptor extents must be disjoint. RMSNorm's parameter descriptor is exactly
four bytes; every vector descriptor is exactly `count * 4` bytes. Multiply has
three vector descriptors. RMSNorm independently reduces the full input in each
active output lane, so it needs no cross-wave synchronization or additional
register allocation. This intentionally repeats work for a bounded correctness
proof. Input, normalized scratch and weight values are consumed at execution;
they are not host-precomputed normalized values.

### RoPE bindings

| Registers | Meaning |
| --- | --- |
| `s[0:3]` | Resident F32 cosine vector, exactly `pairs * 4` bytes |
| `s[4:7]` | Resident F32 sine vector, exactly `pairs * 4` bytes |
| `s[8:11]` | Read-write F32 x vector, exactly `pairs * 8` bytes |
| `s12`, `s13` | Pair count, encoded flags `1 + split_half + 2 * inverse` |
| `s14` | System workgroup X ID, 0..31 |
| `s15`, `s16` | Decoded flags, sign/offset scratch |
| `v0`, `v1` | First and second coordinate byte offsets |
| `v2..v5` | Original a/b, cosine and signed sine |
| `v6`, `v7` | Arithmetic scratch |

RoPE requires an even x length of 2..4096, hence 1..2048 pairs, flags 1..4, and
`ceil(pairs / 64)` groups. Empty operations are retained host no-ops. All three
four-byte-aligned physical descriptor ranges must be disjoint, including cosine
versus sine. Each lane owns one complete pair: interleaved offsets are `8*i` and
`8*i+4`; split-half offsets are `4*i` and `4*(i+pairs)`. Both coordinates and
coefficients load before either output store. Masked lanes and excess groups do
not access memory. The kernel reads current resident coefficient values on each
execution; it does not generate or cache trigonometric tables.

### Attention bindings

Attention uses three separate resident passes, each retaining the three-SRD,
two-scalar ABI. Every pass requires pairwise disjoint physical descriptor ranges.
The host supplies exact descriptor extents, eight-byte-aligned F64 score scratch,
and four-byte-aligned F32 vectors and scale. Head dimension is 1..256 and attended
prefix length is 1..2048; KV stride is at least head dimension and at most 4096
F32 elements. Scores use at most 32 wave64 workgroups; softmax uses exactly
one; values use at most four.

| Program | SRD0 | SRD1 | SRD2 | Scalars |
| --- | --- | --- | --- | --- |
| Scores (7) | Q head, `head_dim * 4` bytes | Strided K prefix | Unscaled F64 scores, `positions * 8` bytes | `[head_dim, kv_stride]` |
| Softmax (8) | Positive finite F32 scale, 4 bytes | F64 scores, `positions * 8` bytes | F32 probabilities, `positions * 4` bytes | `[positions, 1]` |
| Values (9) | F32 probabilities, `positions * 4` bytes | Strided V prefix | F32 output head, `head_dim * 4` bytes | `[positions, kv_stride]` |

The exact K/V prefix extent is `((positions - 1) * kv_stride + head_dim) * 4`
bytes, at most 33,539,072. Score descriptors reach 16,384 bytes and probability
descriptors reach 8,192 bytes. Prefix descriptors begin at the selected KV head, so
other heads' elements are skipped by stride. The score shader derives positions
from SRD2's byte count (`s10 / 8`); the value shader derives head dimension from
`s10 / 4`. Both explicitly reject zero, misaligned, or oversized output extents
before memory access. All three shaders reject excess groups and mask final
lanes before loads. Exact lengths for every other binding remain host obligations.

Each score lane computes one key-position dot product. Each softmax lane scans
all scores in order for the same maximum and sum, then writes positions
`lane + 64*k`. The output loop shrinks the active-lane mask before each subsequent
score load/store. Its separate nonfinite-maximum path writes positive zero to
every output without further score loads. No cross-lane reduction or changed
F32 summation order is introduced.
Each value lane serially reduces one output coordinate. `v2:v3` holds the F64
sum in score/value; `v4:v5` and `v6:v7` hold widened operands. Softmax uses `v2`
for the maximum, `v3` for its F32 sum, `v4:v5` for score/scratch and `v6:v7` for
F64 scale; `s17` selects the zero path and `s18` bounds output-byte offsets.
All programs remain within eight VGPRs and scratch SGPRs `s15..s18`.

These kernels consume a host-validated causal prefix. They do not implement
absolute query-position validation, GQA head selection, KV append/copy, or cache
ownership; those are responsibilities of recorded lowering and its admission.

At the maximum 4096-by-4096 geometry, BF16 weights occupy 33,554,432 bytes and
Q8_0 weights occupy 17,825,792 bytes. Vector payloads are at most 16,384 bytes.
Row and column offsets therefore remain within the 32-bit byte offsets used by
the instructions. Exact descriptor extents, arena admission, page-rounded memory
budgets and disjoint bindings remain separate host obligations; larger shape
bounds do not allocate storage or make an undersized arena sufficient.

The SRDs use base address low/high, stride zero, exact byte length in word 2, and
word 3 `0x30016fac`: XYZW selection, GFX11 format 32_FLOAT (22), raw bounds mode
(3), buffer type zero, no swizzle or resource-level bit. Unformatted load/store
instructions address bytes and do not perform format conversion. See Mesa's
[raw descriptor builder](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/common/ac_descriptors.c)
and [GFX11 resource fields](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/registers/gfx11-rsrc.json).

Required compute register values:

- `COMPUTE_PGM_RSRC1 = 0x408f0001`: eight VGPRs, reserved SGPR-count bits zero,
  nearest-even rounding, preserve FP32 and FP16/FP64 denormals, IEEE mode, no DX10 NaN
  clamp, CU mode, ordered memory.
- `COMPUTE_PGM_RSRC2 = 0x9c`: 14 user SGPRs and system workgroup-X ID; no scratch,
  traps, LDS, extra work-item dimensions, or exceptions.
- `COMPUTE_PGM_RSRC3 = 0`: no shared VGPRs or startup prefetch requirement.
- Thread dimensions `[64, 1, 1]`, group dimensions `[1..64, 1, 1]`, dispatch initiator
  `1` (compute enabled, wave64).

These fields follow the [LLVM AMDGPU register documentation](https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-kernel-descriptor).
There are explicit VMEM load/store completion waits and conservative eight-cycle
NOP spacing between dependent vector operations. These stalls are intentional in
the first proof and must not be optimized away without numerical validation.

## Numerical contracts

### Attention

Score lanes widen Q/K elements before multiplying and accumulate separately in
F64, in increasing dimension order. Unscaled scores remain F64 in scratch.
Softmax multiplies each score by the F64-converted F32 scale, then narrows to F32
before the maximum or exponential calculation, matching the CPU attention stages.
An ordered maximum ignores NaNs. A nonfinite maximum produces all-zero
probabilities; otherwise nonfinite scores contribute exactly zero. Each finite
score uses `exp2((score - max) * f32(log2(e)))`, followed by an increasing-position
F32 sum and approximate F32 reciprocal multiplication. Value lanes widen F32
probabilities and values, multiply and accumulate separately in F64 in increasing
position order, then convert only the final sum to F32. No FMA is used.

The exponential/reciprocal path requires tolerance qualification, not bitwise CPU
parity. Hardware-free tests stress uniform -1/0/+1-ULP and alternating ±1-ULP
exponential errors, and -1/0/+1-ULP reciprocal errors. The finite grid covers
1, 2, 31, 32, 63, 64, 65, 127, 128, 255, 256, 257, 341, 342, 511, 512,
513, 1023, 1024, 1025, 2047 and 2048 positions; flat, varied,
dominant-score and near-equal inputs; and scales 0.0625, F32(0.088388346), 0.5,
1 and 2. The declared per-probability bound is
`abs_error <= 2e-6 + 2e-5 * abs(CPU_reference)`, with a probability-sum bound of
2e-5 around one. Separate models exercise nonfinite masking, preservation of
F64 scores until after scaling, and small weighted terms lost by F32 reduction.
These finite software experiments do not establish a global numerical bound or
hardware behavior. Embedded-word address tests separately trace finite/zero output
loops, exact coverage, and pre-access masking at those boundaries. The 341/342
pair also straddles the `12*positions + 4` workspace's 4096-byte threshold used
by a composed layer with a 1024-element weighted norm.
The separate live diagnostic and real-layer results below
qualify only their stated fixtures; throughput and arbitrary inputs remain open.

### RoPE

For each pair `(a, b)` and resident coefficients `(c, sin)`, the kernel computes
`s = sign * sin`, `first = a*c - b*s`, then `second = a*s + b*c`, with sign -1
for inverse rotation and +1 otherwise. Each multiplication, subtraction and
addition is a separate F32 operation, matching the CPU vector routines' order.
No FMA or transcendental approximation is involved. Software tests compare both
layouts and directions, changed coefficients, signed zeros and subnormal inputs
with the CPU routines, and distinguish rounded products from fused arithmetic.
These model/encoding checks exceed the live fixture scope. The isolated live gate
passed 1,562 bitwise-exact values and 26 guards across 13 dispatches, covering both
layouts/directions and a 65-pair tail. Arbitrary inputs and all admitted sizes remain
unqualified; see [the exact result](../../../../../docs/hag-backend.md#native-rope-attention-and-resident-attention-block--2026-10-03).

### RMSNorm and weight multiplication

The unweighted kernel widens each F32 input to F64, squares it in F64, and adds
to a positive-zero F64 accumulator in increasing element order, matching the
reduction structure of `beholdd_kernel::vecops::rmsnorm`. It then computes
`sum * rcp_f64(count) + f64(epsilon)` and applies `rsq_f64`. Only the resulting
inverse is converted to F32, followed by the F32 `input * inverse`. The optional
second program loads current resident weights and computes `normalized * weight`
as another F32 multiplication. There is no FMA, F32 reduction or reassociation
of the two final multiplications.

LLVM documents the [GFX11 F64 instructions](https://releases.llvm.org/19.1.0/docs/AMDGPU/AMDGPUAsmGFX11.html).
F64 reciprocal and reciprocal-square-root instructions are approximations, not
CPU division/square-root equivalents. The hardware-free model stresses each
stage at -2^29, zero and +2^29 binary64 ULP, using the precision envelope described
for these instructions in the [AMD RDNA2 ISA](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content).
This envelope is an arithmetic stress model; it is not a measured GFX1151 bound
or a claim of bitwise CPU inverse parity. Finite grids cover lengths 1, 2, 31,
32, 63, 64, 65, 127, 128, 1024, 3072 and 4096 with epsilon 0, 1e-6, 1e-5 and 0.5,
with and without weights. Their explicit acceptance bound is
`abs_error <= 2e-6 + 2e-6 * abs(CPU_reference)`.

Separate model regressions preserve F64 square range for normal F32 inputs near
1e20 and 1e-30, retain small squared terms that sequential F32 accumulation loses,
check changed inputs/weights, and check signed zero. Epsilon zero is valid; an
all-zero vector then models `0 * infinity` and produces NaNs, as does the CPU
reference. These exceptional cases and arbitrary input bit patterns remain
outside the finite live qualification recorded below.

### Packed matvec

Both kernels accumulate from positive zero in increasing-column order, with
separate F32 multiply and add instructions (no FMA or reassociation). BF16 widening
is the exact bit shift by 16. Each Q8_0 block is 34 bytes: a little-endian F16 scale
followed by 32 signed I8 values. Q8_0 first computes `f32(scale) * f32(quant)`, then
multiplies by the input and adds to the accumulator. Scale reads at offsets
`row_stride + block*34` remain two-byte aligned even at non-four-byte boundaries.

The matching Beholdd CPU references are `bf16::dot_bytes_scalar` and
`kquant::q8_0::dot`. Compare against those separately from an accelerated CPU
backend, which may use FMA and a different reduction order. NaN payload identity
is not a portable cross-device promise. The surrounding proof must distinguish
its finite test coverage from full model/inference qualification.

### Scaled-add

Each lane computes `product = scale * input`, then `output = old_output + product`
with separate round-to-nearest F32 instructions. There is no FMA or reassociation.
The old output is loaded before the store. The host requires a finite scale;
this does not qualify all possible input/output bit patterns or NaN payloads.

### SwiGLU

For a positive finite limit, gate is clamped only above and up is clamped to
`[-limit, limit]`. Ordered comparisons/selects deliberately implement Rust's
distinct semantics: a NaN gate becomes the positive limit (`f32::min`), while a
NaN up is retained (`f32::clamp`). The shader disables clamping for limits <= 0;
the public execution contract rejects negative or nonfinite limits before recording.

The activation uses `exp2((-gate) * f32(log2(e)))` and reciprocal multiplication
to approximate `(gate / (1 + exp(-gate))) * up`. This is not bitwise-equivalent
to the CPU library exponential/division. AMD documents one-ULP exp2/reciprocal
instructions with denormal flushing in the [RDNA3 ISA, sections 16.8/16.12](https://docs.amd.com/api/khub/documents/UkT_UPQL21KfKAMUBFnZTw/content).
For denominators above `2^96`, the shader evaluates
`(gate * rcp(denominator * 2^-32)) * 2^-32` before multiplying up. This avoids
premature reciprocal flushing near gate = -88, where a large up could magnify
the otherwise-lost result. No FMA is used.

Hardware-free staged tests cover a moderate finite grid with `|gate| <= 20`,
`|up| <= 20`, and limits 0, 0.5, 2, 10 and 20 using the explicit acceptance bound
`abs_error <= 2e-5 + 2e-5 * abs(f64_reference)`. They perturb modeled normal SFU
outputs by -1/0/+1 representable values. Separate regressions exercise gate near
-88 with up = 1e38, adjacent F32 values around exp overflow at -88.72284, clamp
asymmetry, zero signs and NaN/infinity classification. These are arithmetic-model
checks, not GPU emulation or proof of a global error bound. Physical SFU behavior
at overflow/underflow boundaries, exceptional values and arbitrary inputs remains
unqualified; no full-model or lossless inference claim follows from these tests.

## Offline verification

The eleven arrays in `mod.rs` are compiled from the adjacent `.s` files with LLVM 21
for `gfx1151`, explicitly enabling wave64. Each includes 64 `s_code_end` padding
words, occupies less than 4096 bytes, and contains no relocations. The complete
library occupies two pages as described above.
The assembler is a developer verification tool, not a Cargo build dependency.

From the Beholdd repository:

```sh
BEHOLDD_LLVM_BIN=/usr/lib/llvm-21/bin cargo test --locked -p beholdd-backend --features hag-native \
  hag::kernels::tests::llvm_sources_match_embedded_words -- --ignored --exact
```

This test assembles, extracts the `.text` section, compares every byte with the
embedded words and decodes the resulting object. It never opens a GPU or loads a
model. Ordinary tests check bounds, register fields and fixed instruction shape.
A deliberately narrow decoder exercises the embedded admission/indexing prologues:
boundary grids cover each valid row/element exactly once, tail lanes and excess
groups reach no memory instructions, and invalid dimensions/block counts refuse.
This is not an arithmetic or hardware emulator. The arithmetic bodies retain
the prior serial order; additional workgroups parallelize independent output rows.

Queues must upload shader bytes once and keep them immutable, retain all BOs and
VAs until confirmed completion, use native cache invalidation before consuming
host-rewritten inputs/IBs, and reject output on timeout or reset. A source/encoding
match does not prove live hardware stability or model correctness.

The supervised 2026-09-26 finite fixture suite passed 16 BF16/Q8_0 cases across
32 reused-queue dispatches, with 604 bitwise-checked output rows, unchanged
inputs/guards and complete cleanup. This does not qualify arbitrary numerical
inputs, larger shapes, throughput or model execution. See the authoritative
[HAG validation record](../../../../../vendor/exec-hag/docs/native-compute.md).
The prior one-group elementwise payloads also passed an isolated SwiGLU and
bounded resident FFN proof on 2026-10-03. Those records apply to the archived
payloads and tested small fixtures.

The extended Q8_0/SwiGLU payloads subsequently passed the following bounded gates
on 2026-10-03, with frozen binary SHA-256
`b4fb5ee0472998fc4ddfe354cc72290b94d04caec384da2f8eac9198bba1f5bf`:

- One 65-by-32 Q8_0 operation used two workgroups, matched all 65 output rows
  bitwise and preserved both adjacent output canaries, including the masked tail.
- One real-checkpoint layer-0 FFN used Q8_0 gate/up matrices of 3072-by-1024,
  SwiGLU over 3072 elements, and a Q8_0 down matrix of 1024-by-3072. Three changed
  deterministic inputs reused one recorded batch: 12 dispatches and 3,072 final
  outputs passed the declared CPU-reference bounds. Maximum absolute error was
  `1.1920928955078125e-7`; this is finite-fixture tolerance qualification.

The model-shaped gate retained 10,027,008 packed bytes inside a 10,125,312-byte
HAG charge. Native preparation performed three source reads, versus nine for the
CPU oracle; submit/wait performed no adapter tensor transfers. Explicit final
verification readback was 12,288 bytes and the final HAG budget was zero. The
qualification cgroup peaked at 67,297,280 bytes, maximum sampled Tctl was 44.625°C,
and no OOM, swap or matching driver fault was recorded. See the
[Beholdd qualification record](../../../../../docs/hag-backend.md#model-shaped-q8_0swiglu-qualification--2026-10-03)
for numerical bounds, supervision and the retained evidence archive.

### RMSNorm and residual FFN block qualification — 2026-10-03

Frozen binary SHA-256
`7e191d5d149a4f8f51075c619c4f846581a5fe0f0da2166a78020ced465f227f`
passed the isolated RMSNorm diagnostic and the subsequent checkpoint block gate
in one root crash-capture session. The previous four shader payloads remained
byte-identical when RMSNorm and multiply were added.

- The diagnostic first executed unweighted RMSNorm with one element, input 2 and
  epsilon zero, producing exactly 1. It then reused weighted RMSNorm over 65
  elements across three changed input/weight rounds, using two workgroups and
  epsilon 1e-5. All 196 checked values matched the CPU reference exactly, all
  eight adjacent guard checks passed, and seven physical dispatches completed
  across four submissions. The preregistered bound was
  `abs_error <= 2e-6 + 2e-6 * abs(CPU_reference)` with finite/sign/zero checks.
- The real Qwen3 checkpoint layer-0 block used weighted RMSNorm over 1024
  elements with checkpoint epsilon F32(1e-6), Q8_0 gate/up matrices of
  3072-by-1024, SwiGLU over 3072 elements, Q8_0 down projection of
  1024-by-3072, and scaled-add residual update over 1024 elements. Three changed
  inputs reused one five-command batch: 21 physical dispatches and 3,072 final
  outputs passed finite/sign/zero checks and the preregistered bound
  `abs_error <= 3e-4 + 3e-5 * abs(CPU_reference)`. Maximum absolute error was
  `2.384185791015625e-7`.

The block retained 10,027,008 packed matrix bytes and used a 10,129,408-byte HAG
charge, including the resident normalization weights and scratch. Preparation
performed four native source reads versus twelve for the CPU oracle; submit/wait
performed no adapter tensor transfers. Explicit verification readback was 12,288
bytes, reuse did not grow resident resources, and final HAG budget usage was zero.
The qualification cgroup peaked at 66,781,184 bytes; maximum sampled Tctl was
54.75°C. Both gates recorded zero swap and OOM events, no matching kernel driver
faults, and healthy crash capture. The diagnostic used a 45,056-byte HAG charge,
peaked at 21,487,616 cgroup bytes and sampled at most 42.75°C.

Results, logs and thermal samples are retained in the
`hag-rmsnorm-20261003` qualification evidence. See the
[Beholdd qualification record](../../../../../docs/hag-backend.md)
for supervision and the retained archive. These results qualify the stated
finite RMSNorm/multiply fixtures and the 1024-element normalized residual block,
including its scaled-add payload. They do not establish bitwise CPU inverse
parity for arbitrary inputs.

The extended BF16 payload and the full admitted 4096-by-4096/4096-element domain
remain outside live qualification. The records cover their stated shapes,
payloads and inputs; they do not establish arbitrary numerical coverage, other
layers, full-model serving or throughput.

### RoPE, causal attention and resident attention block — 2026-10-03

The ten-program library passed all three supervised gates at 20:19–20:20 UTC.
The prior six payloads and offsets were unchanged. The diagnostic binary SHA-256
was `fb35bbd7fab3c433839e21e195cc2e3d35f40527f8f18d29c2f3e6aaee99b329`;
the real-layer binary was
`4028dd08a32a75d39b8c14ba4e2185b2ab4c9b5ec95432799f4d72604d831eeb`.

- RoPE passed an analytic pair followed by three changed-input/coefficient rounds
  of all four layout/direction combinations over 65 pairs. All 1,562 values and
  26 guards matched bitwise across 13 dispatches, also exercising the first
  shader-page crossing. Maximum absolute error was zero.
- Attention passed an analytic one-position case, then three changed rounds of
  three-query prefill at head width 4 and two cached queries starting at position
  63 with head width 128. The latter fixtures used four query/two KV heads with
  NaN-filled future K/V rows. Across seven submissions and 183 dispatches, all
  3,218 values and 14 guards passed; maximum absolute error was
  `1.1920928955078125e-7` under the preregistered
  `2e-5 + 2e-5 * abs(reference)` bound.
- The Qwen3-0.6B-Q8_0 layer-0 attention/residual block passed two requests of three
  serial positions each, with changed synthetic boundary hidden values and real
  weights. It exercised width-1024 input norm, 16 query/8 KV heads of width 128,
  weighted Q/K head normalization, absolute-position split-half RoPE, causal GQA
  and output/residual projection. Capacity eight was allocated; only positions
  0..2 ran per request. Reset reported zero visible positions. Each step had 54
  semantic commands and 127 dispatches; 762 total dispatches checked 6,144 final
  hidden values. Maximum absolute error was `2.384185791015625e-7` under the
  unchanged `3e-4 + 3e-5 * abs(reference)` bound, including finite/sign/zero checks.

The real block retained 6,684,672 packed matrix bytes, 111,616 public tensor bytes
and 165,168 live auxiliary bytes, with an actual **7,098,368-byte HAG charge**.
Seven native preparation reads totaled 6,689,792 bytes; the independent immediate
one-worker CPU oracle made 42 reads before native access. Replays did not reread
or reupload weights, submit/wait transferred zero tensor bytes, and only final
hidden was read back (24,576 bytes). KV and intermediates remained resident;
cleanup released the complete charge. No intermediate numerical parity is claimed.

Both isolated gates charged 307,200 HAG bytes and released them fully. Peak
process/cgroup bytes were 21,790,720 for RoPE, 22,134,784 for attention and
50,274,304 for the block; maximum sampled Tctl was 65.375°C, 63.625°C and 55.75°C.
All runs verified 1-GiB memory, zero-swap, 64-task and 100%-CPU limits and no new
privileges, with the unchanged 90°C cutoff. Memory/OOM events, swap and matching
GPU faults were zero. Fresh root capture stopped cleanly with no dumps; the
complete 213-line kernel window contained no matching GPU/OOM/fault messages.
An earlier hot-session hold remains historical and did not relax any bound.

Reports, source/checkpoint identities and exact scope are in the
[Beholdd qualification record](../../../../../docs/hag-backend.md#native-rope-attention-and-resident-attention-block--2026-10-03)
and `outputs/hag-rope-attention-qualified-20261003.tar.gz` in the development chat.
These finite fixtures do not qualify arbitrary sequences, every admitted geometry,
other layers, embedding/logit execution, whole-model budgeting or parity, serving
integration or throughput. At that milestone, native serving remained CPU-only.

### Q8_0 row decode and tied vocabulary component — 2026-10-03

The eleven-program library passed isolated row decoding at 21:02:34–21:02:37 UTC
and the selected Qwen3-0.6B Q8_0 vocabulary component at 21:02:46–21:02:50 UTC.
All ten earlier payloads and offsets remained byte-identical. Frozen binary
SHA-256 identities were
`81a65a75df034931068cb62f0bea4186636919ec9b3f54b5be2a532b1894bdf2`
for `hag_embedding_probe` and
`3877e0e016374f1e6bb3623df8b369eaffe2bb006838ed0f7d862a0988461c75`
for `hag_vocabulary_probe`. The 124-file source manifest has SHA-256
`f66a1eb1a1295be62f1f59759b9dfd7f63a26ecf1f062282964ebc75902409b2`;
checkpoint SHA-256 is
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.

- The diagnostic selected rows from 32-, 96- and 4096-column fixtures and replayed
  each twice. Sixteen dispatches checked 17,152 values bitwise and 32 guards,
  covering masked tails, two-byte-aligned row starts, signed zeros and F16
  subnormal scales. HAG charged 90,112 bytes and returned to zero.
- The vocabulary gate retained one shared 165,306,368-byte packed table as
  37 tiles of 4096 rows and a final 384-row tile, all 1024 columns wide. Six
  lookups at token IDs 0, 4095, 4096, 65536, 151551 and 151935 checked 6,144
  embedding values bitwise. Three changed synthetic hidden vectors exercised
  weighted final RMSNorm and all 151,936 logits, checking 455,808 head values.
  Total execution was 126 dispatches, nine submissions and 18 output guards.

The preregistered head policy required finite values within
`3e-4 + 3e-5 * abs(CPU_reference)` and exact CPU greedy selection; sign and zero
mismatches were reported separately. Observed maximum absolute error was zero,
both mismatch counts were zero, and all three greedy IDs matched (117270,
121057 and 253). This result is finite-fixture tolerance qualification, not a
general bitwise-logit guarantee.

The vocabulary HAG charge matched 165,974,016 bytes and returned to zero. Native
preparation made 39 source reads versus 123 for the independent CPU oracle, which
finished before native access. Tiles were shared by lookup and the head without
reread or reupload; largest packed staging was 4,456,448 bytes and retained CPU
references occupied 1,847,808 bytes. Submit/wait transferred zero tensor bytes.
Explicit output verification read 1,847,880 bytes; recording wrote 4,124 parameter
and scratch bytes, separate from boundary input writes.

Both gates verified 1-GiB process memory, zero swap, 64 tasks, 100% CPU and no new
privileges, with thermal and runtime supervision. No OOM, swap or matching driver
fault occurred. Root watcher PID 1229789 stopped cleanly at 21:03:07.816456 UTC
with zero dumps; the complete 469-line kernel window had no GPU/OOM/fault matches.
All 124 frozen sources, binaries and diagnostic dependencies matched at closeout.
The development chat retains results, identities and the final capture audit in
`work/hag-vocabulary-20261003/`.

These gates qualify the recorded row fixtures and selected embedding/final-head
component, including its shared resident table and full-vocabulary output. They
do not compose all model layers or establish arbitrary input/shape coverage,
whole-model logits/token parity, serving integration or throughput. At that
milestone, native serving remained CPU-only.
