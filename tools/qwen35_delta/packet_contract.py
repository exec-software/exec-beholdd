"""Source-only v2 packet contract for the native DeltaNet diagnostic.

This module opens no devices/files and performs no work on import. All offsets
are bytes from one of three exact, disjoint raw SRDs. It is independent of the
production implementation. No admission/serving capability is implied.
"""

from dataclasses import dataclass
import math
import struct

F32 = struct.Struct("<f")
U32 = struct.Struct("<I")
QNAN_BITS = 0x7FC00000
PENDING = 0xFFFFFFFF
SUCCESS = 0
NONFINITE = 1


def f32(value):
    """Round once to finite IEEE binary32; reject overflow and nonfinite input."""
    try:
        result = F32.unpack(F32.pack(value))[0]
    except (OverflowError, struct.error, TypeError) as error:
        raise ValueError("value is not representable as finite f32") from error
    if not math.isfinite(result):
        raise ValueError("value is not finite f32")
    return result


@dataclass(frozen=True)
class Geometry:
    key_heads: int
    value_heads: int
    head_dim: int
    epsilon: float = 1e-6

    def __post_init__(self):
        h, v, d = self.key_heads, self.value_heads, self.head_dim
        if any(type(x) is not int for x in (h, v, d)):
            raise ValueError("geometry fields must be integers")
        if not (1 <= h <= 16 and h & (h - 1) == 0):
            raise ValueError("key heads must be a power of two in 1..16")
        if not (1 <= d <= 128 and d & (d - 1) == 0):
            raise ValueError("head dimension must be a power of two in 1..128")
        if not (1 <= v <= 48 and v % h == 0):
            raise ValueError("value heads must be in 1..48 and divisible by key heads")
        epsilon = f32(self.epsilon)
        if epsilon <= 0.0:
            raise ValueError("epsilon must be finite positive f32")
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def key_width(self):
        return self.key_heads * self.head_dim

    @property
    def value_width(self):
        return self.value_heads * self.head_dim

    @property
    def channels(self):
        return 2 * self.key_width + self.value_width

    @property
    def bank_bytes(self):
        return 4 * (3 * self.channels + self.value_width * self.head_dim + self.value_width)

    def parameters(self):
        c, v, d = self.channels, self.value_heads, self.head_dim
        return {"conv": 0, "a": 16*c, "dt": 16*c + 4*v,
                "norm": 16*c + 8*v, "epsilon": 16*c + 8*v + 4*d,
                "bytes": 16*c + 8*v + 4*d + 4}

    def projections(self):
        c, w, v = self.channels, self.value_width, self.value_heads
        return {"qkv": 0, "z": 4*c, "alpha": 4*(c+w),
                "beta": 4*(c+w+v), "bytes": 4*(c+w+2*v)}

    def frame(self):
        c, k, w, v, d = (self.channels, self.key_width, self.value_width,
                          self.value_heads, self.head_dim)
        base = 2 * self.bank_bytes
        conv = base
        normalized_q = conv + 4*c
        normalized_k = normalized_q + 4*k
        decay = normalized_k + 4*k
        beta = decay + 4*v
        core = beta + 4*v
        status = core + 4*w
        return {"bank0": 0, "bank1": self.bank_bytes,
                "bank_conv": 0, "bank_recurrent": 12*c,
                "bank_output": 12*c + 4*w*d,
                "conv_qkv": conv, "normalized_q": normalized_q,
                "normalized_k": normalized_k, "decay": decay,
                "beta": beta, "core": core, "status": status,
                "bytes": status + 4*v}

    def scalars(self, active_bank):
        if type(active_bank) is not int or active_bank not in (0, 1):
            raise ValueError("active bank must be 0 or 1")
        packed = ((self.head_dim.bit_length()-1)
                  | ((self.key_heads.bit_length()-1) << 3)
                  | ((self.value_heads-1) << 6) | (active_bank << 12))
        return packed + 1, 1

    def dispatch_groups(self):
        return ((self.channels+63)//64, 2, (self.value_width+63)//64, 1)


def decode_scalars(word0, word1, epsilon):
    if any(type(x) is not int or not 0 <= x <= 0xFFFFFFFF for x in (word0, word1)):
        raise ValueError("scalars must be u32")
    if word0 == 0 or word1 != 1:
        raise ValueError("scalar0 must be positive and scalar1 must equal one")
    packed = word0 - 1
    if packed >> 13:
        raise ValueError("reserved geometry bits must be zero")
    shape = Geometry(1 << ((packed >> 3) & 7), 1 + ((packed >> 6) & 63),
                     1 << (packed & 7), epsilon)
    return shape, (packed >> 12) & 1
