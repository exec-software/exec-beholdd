# Delta diagnostic packet v2

This contract belongs only to an explicit `Qwen35Probe` diagnostic. It neither
admits a model nor changes serving or fallback. The shaders have passed LLVM
assembly/embedded-byte checks and host contract tests; GPU numerical qualification
remains separate. `tools/qwen35_delta/packet_contract.py` independently describes
the byte formulas below; its import performs no work.

## Geometry and scalars

Fixed convolution K=4. D is a power of two in 1..128; Hk is a power of two in
1..16; Hv is in 1..48, divisible by Hk. Define Q=Hk*D, V=Hv*D,
C=2*Q+V, R=V*D. All fields and derived arithmetic must be checked before GPU
allocation/recording, even though these caps make products fit u32.

* `s12` is ONE PLUS the packed value whose bits0..2 are log2(D), bits3..5
  log2(Hk), bits6..11 Hv-1, bit12 active bank b; bits13..31 zero.
  Subtract one before decoding. Destination bank is1-b.
* `s13` is exactly1. Both raw scalars remain in the existing transport range
  1..65536. Epsilon is an immutable F32 parameter trailer, not a raw scalar.
* `s14`: workgroup X, supplied by the queue. No raw addresses are user scalars.

All three SRDs refer to disjoint allocations (or provably disjoint extents) owned
by the same queue. Base/length are multiples of four. Lengths are exact, not arena
capacity. Shader prologues reject bad geometry, descriptor extents or groups;
host admission supplies the authoritative error before dispatch. Every shader
must preserve bounds even when invoked with malformed scalars.

## Exact byte layout

SRD0 (`s[0:3]`): immutable parameters, length **16*C + 8*Hv + 4*D + 4**.

| Field | Byte offset | F32 elements / order |
| --- | ---: | --- |
| conv | 0 | 4*C; [channel][oldest tap .. current tap] |
| A | 16*C | Hv; stored non-positive -exp(A_log), signed zero allowed |
| dt | 16*C + 4*Hv | Hv |
| norm | 16*C + 8*Hv | D; actual stored weight, no +1 transform |
| epsilon | 16*C + 8*Hv + 4*D | 1; finite positive F32, zero/negative zero refuse |

SRD1 (`s[4:7]`): projections, length **4*(C+V+2*Hv)**.

| Field | Byte offset | F32 elements / order |
| --- | ---: | --- |
| qkv | 0 | C; Q query, Q key, V value |
| z | 4*C | V; [value head][coordinate] |
| alpha | 4*(C+V) | Hv |
| beta input | 4*(C+V+Hv) | Hv |

SRD2 (`s[8:11]`): private frame, length **2*B+4*(C+2*Q+2*Hv+V+Hv)**,
where **B=4*(3*C+R+V)** is the bank stride. Both banks stay resident.

| Field | Byte offset | Elements / order |
| --- | ---: | --- |
| bank 0 | 0 | B bytes |
| bank 1 | B | B bytes |
| bank-relative conv history | 0 | 3*C F32; [channel][oldest,middle,newest], RAW projections |
| bank-relative recurrent | 12*C | R F32; [value head][value coordinate j][key coordinate i] |
| bank-relative output | 12*C + 4*R | V F32 |
| convolved QKV | 2*B | C F32, after SiLU |
| normalized Q | 2*B + 4*C | Q F32 |
| normalized K | 2*B + 4*(C+Q) | Q F32 |
| decay rho | 2*B + 4*(C+2*Q) | Hv F32 |
| beta gate | 2*B + 4*(C+2*Q+Hv) | Hv F32 |
| core readout | 2*B + 4*(C+2*Q+2*Hv) | V F32 |
| status | 2*B + 4*(C+2*Q+2*Hv+V) | Hv U32 (raw bits, not numeric F32) |

OpenMythos Hk=16,Hv=48,D=128: Q=2048,V=6144,C=10240,R=786432,
B=3293184. Parameter offsets: conv=0,A=163840,dt=164032,norm=164224,
epsilon=164736,length=164740. Projection offsets: qkv=0,z=40960,alpha=65536,beta=65728,
length=65920. Frame: bank1=3293184; bank-relative recurrent=122880,
output=3268608; convolved=6586368,Q=6627328,K=6635520,rho=6643712,
beta=6643904,core=6644096,status=6668672,length=6668864.
Three-descriptor payload total=6899524 bytes, excluding rounding/guards/library.

## Program responsibilities and publication

12, convolution: dispatch ceil(C/64) groups; each active lane c owns one channel.
Before any arithmetic refusal, lanes c<V initialize core[c] to canonical qNaN
0x7fc00000; lanes c<Hv initialize status[c] to PENDING 0xffffffff. These are
unconditional bounded writes before any lane masks itself. Compute conv from
active-bank raw history and current projection in oldest-to-newest F32 order,
then stable SiLU. Write all next-bank raw history and convolved QKV; invalid
input/intermediate writes canonical NaN to that convolved coordinate. Never mask
an invalid lane without writing its convolved invalid marker. Input/weight and
every intermediate finite check must not be accidentally suppressed by min/max.

13, normalization/gates: exactly two groups. Group0: lane<2*Hk owns one entire
Q or K head, scans D coordinates in order, then writes D normalized coordinates
out of place. Group1: lane<Hv owns rho/beta for that value head. This is a uniform
group branch, not a divergent role branch. Always write either all valid results
or canonical NaNs for the failed head. Any overflow of a F32 square or biased
alpha is failure even if a later reciprocal would hide it. Q/K use head order
already stored in GGUF, with no extra reordering.

14, recurrence: ceil(V/64) groups; lane n owns h=n/D, j=n%D and exactly D state
cells. Query/key head is h%Hk. Prediction uses decayed ACTIVE state. Per cell:
old*rho in F32, then that*k[i] in F32, accumulate in F64 increasing i, cast sum
to F32. delta=(v[j]-prediction)*beta in F32. Second pass recomputes old*rho,
updates inactive cell = decayed + k[i]*delta with separate F32 multiply/add,
then accumulates updated*q[i] F32 products in F64. Cast sum to F32, multiply by
F32 reciprocal(sqrt(F32(D))), write core[n]. Any invalid operand/intermediate
masks that lane, preserving preinitialized NaN core. Partial inactive state
is allowed on refusal and is never published. No cross-lane state writes.

15, final gate: one group; lane h<Hv owns one entire value head. Scan core[D],
accumulate F32 squares in F64; mean=F32(sum/F64(D)), denominator=F32(mean+eps),
scale=F32(1/F32(sqrt(denominator))). For each j in order, inactive output is
F32(F32(F32(core[j]*scale)*norm[j])*SiLU(z[j])). Check every core, square,
normalization intermediate, z and final product. Status[h]=SUCCESS(0) only after
all D outputs are valid; otherwise status[h]=NONFINITE(1), or a masked failed
lane preserves its preinitialized PENDING. Both cause refusal. No failure path
may write SUCCESS. No atomics/barriers/cross-workgroup reduction are needed.

Each dispatch is fenced before the next. The diagnostic step then explicitly
reads only the Hv status words (at most192 bytes) OUTSIDE `submit`. Pending,
unknown or nonzero statuses refuse. Count this control read in transfer receipts.
No activation/state readback or CPU arithmetic fallback happens during a step.
Only then swap active bank and increment the token counter, publishing new output.
Previous output/state remain untouched on arithmetic refusal. All next-bank state
and per-token scratch are fully overwritten by the next successful attempt.

The host exclusively owns sequence bank metadata; callers cannot select bank or
record partial stages. Mutable packet aliases must not escape. Recorded parity
batches retain every allocation and are reusable with changed input values.
Reset explicitly zeros BOTH bank histories/recurrent/output in bounded transfer
chunks, resets bank0/token0, and counts those host writes. It cannot unpoison a
device. Transport uncertainty permanently poisons/quarantines the queue using
existing native policy. Cancellation after model-layer commits requires a whole
model reset or abandonment, not token metadata rollback.

## Arithmetic and ISA budget

Stored values/products/additions are F32 unless explicitly marked F64. No FMA.
Norm and dot products round the product to F32 before widening for serial F64
addition. Normalization preserves the F32 mean, epsilon addition, sqrt and
reciprocal boundaries. The current wide RMS kernel differs and is not reusable.
Q/K L2 is (x*RMSNorm(eps/D))*F32(1/sqrt(F32(D))) with separate multiplies.
rho=exp(A*softplus(F32(alpha+dt))), beta=sigmoid(beta input), and both SiLU
operations are x*sigmoid(x). Stable sigmoid uses exp(-abs(x)); the exp result
must first round to F32, including subnormal values or zero, BEFORE multiplying
by x. For abs(x)>48 the shader computes exp2(-min(abs(x),128)*log2(e)+64), then
multiplies that result by2^-64 to recover and round F32 exp. It must not multiply
x before this descaling: x=-105 has literal-reference F32 exp zero, whereas
x=-90 retains its representable subnormal F32 exp. FLOAT_MODE0xf0 preserves
arithmetic denormals; native probes must verify the assumed behavior. Stable
softplus requires a log1p approximation with an accurate small-tail branch.
SFU results are tolerance-based. Epsilon remains any positive finite F32 at shape admission;
underflow/overflow/nonfinite normalization may still cause arithmetic refusal.
Denormal behavior is a qualification boundary, not an excuse for false success.

Current library ends at9756; next aligned start9984 leaves6400 bytes under the
old16384-byte cap. Root reports the32-program/32768-byte extension tested and
committed in canonical HAG cd23d42 and Beholdd vendor057a92d. Qualification of
these new programs is still outstanding.
Every new entry starts on256-byte alignment and includes256 fetch-padding bytes;
every entry must fit4096 bytes. Four new entries therefore use at most16384 bytes.
No dispatch or capability gets enabled until exact assembled layout passes the
actual queue bounds. Existing IDs0..11 remain stable.

Keep v0..v7 if feasible; s0..s14 are ABI inputs, s15..s31 are the scratch envelope
used by existing kernels. No SGPR allocation changes. Source authors must report
live-register maps and pressure before using more than8 VGPRs. A wider VGPR ABI
would be a reviewed Qwen35Probe-only library change and require repeating its
prior operator qualification. DenseQwen3 remains unchanged.

This four-slot diagnostic consumes the old program table. The proposed wider
table permits a separate full-attention sigmoid output gate; do not silently
give the Delta final kernel a second meaning to claim a complete model runtime.

After validating the s13 sentinel and the exact parameter extent, a shader may
reuse s13 for epsilon bits loaded uniformly from the immutable trailer using a
temporary VGPR plus v_readfirstlane. This is local register reuse; the dispatched
scalar remains1. v0 (initial lane ID) must survive uniform admission.
