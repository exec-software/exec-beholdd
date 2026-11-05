# Measuring packed matrix operations

A packed matrix operation includes weight decoding, scaling, accumulation,
addressing and sometimes staging. Profiling only the multiply-add instructions
can miss the dominant cost.

## Method

Measure component costs using equivalent threading, shapes and inputs. A
component apparently costing more than the complete operation should trigger a
measurement review before an optimization claim. Include cache and staging
effects in the experiment rather than assuming that lower arithmetic counts
guarantee lower latency.

Beholdd's `beholdd-kernel` implements packed formats, while backend benchmarks
exercise complete device operations. Keep checkpoint values packed where useful
and account for any expanded representation in the memory plan.

## Numerical requirements

Changing accumulation width, instruction order, fused operations or decoding
strategy can alter numerical results even when checkpoint bytes are unchanged.
Low input precision alone does not prove that a different reduction has no
effect on model outputs.

State the reference, tolerances and exceptional-value behavior before evaluating
a candidate. Bitwise parity, numerical parity and unchanged model quality are
separate claims. Use independent checks and preserve failed cases. The
[exact-inference contract](../vendor/exec-hag/docs/exact-inference.md) defines the
acceptance distinctions used by the experimental backend work.
