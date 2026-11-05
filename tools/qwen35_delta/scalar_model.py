"""Independent, standard-library scalar model of Delta packet v2.

Hardware-independent arithmetic reference. No production modules, NumPy,
checkpoint reads, devices, auto-run, CLI or import-time fixtures. Each arithmetic
boundary is explicit binary32; reductions use Python binary64 in fixed order.
Transcendentals use libm, independently of the proposed GPU SFU approximation.
The object represents exactly one finite bounded diagnostic sequence, not serving.
"""

from array import array
from dataclasses import dataclass
import math
import struct
import sys

from packet_contract import Geometry, PENDING, SUCCESS, NONFINITE, f32


def _raw_f32(value):
    """Preserve NaN/Inf injections, rounding ordinary inputs to IEEE binary32."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _finite(value):
    if not math.isfinite(value):
        raise ArithmeticError("nonfinite input or intermediate")
    return value


def _round(value):
    return _finite(_raw_f32(value))


def _mul(a, b):
    return _round(_finite(a) * _finite(b))


def _add(a, b):
    return _round(_finite(a) + _finite(b))


def _sigmoid(x):
    t = _round(math.exp(-abs(_finite(x))))
    denominator = _add(1.0, t)
    return _round((1.0 if x >= 0.0 else t) / denominator)


def _silu(x):
    return _mul(x, _sigmoid(x))


def _softplus(x):
    tail = _round(math.exp(-abs(_finite(x))))
    return _add(max(x, 0.0), _round(math.log1p(tail)))


def _inverse_root(x):
    root = _round(math.sqrt(_finite(x)))
    if root == 0.0:
        raise ArithmeticError("zero normalization denominator")
    return _round(1.0 / root)


def _norm_scale(values, epsilon):
    # Squaring happens before conversion to the wider accumulator.
    total = 0.0
    for x in values:
        total += _mul(x, x)
    mean = _round(total / len(values))
    return _inverse_root(_add(mean, epsilon))


@dataclass(frozen=True)
class StepResult:
    committed: bool
    active_bank: int
    tokens: int
    status: tuple
    output: tuple | None


class ScalarPacket:
    """Fixed-capacity packet model with transactional active-bank publication."""

    def __init__(self, shape: Geometry, parameters, max_tokens=8):
        if not isinstance(shape, Geometry):
            raise ValueError("explicit Geometry required")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8:
            raise ValueError("diagnostic token capacity is 1..8")
        if sys.byteorder != "little" or array("f").itemsize != 4:
            raise ValueError("packet diagnostic requires little-endian four-byte F32 storage")
        expected = shape.parameters()["bytes"] // 4
        if len(parameters) != expected:
            raise ValueError("parameter packet has wrong exact length")
        self.shape = shape
        self.parameters = array("f", (f32(x) for x in parameters))
        p = shape.parameters()
        if self.parameters[p["epsilon"]//4] != shape.epsilon:
            raise ValueError("epsilon trailer must exactly match the explicit geometry")
        if any(x > 0.0 for x in self.parameters[p["a"]//4:p["dt"]//4]):
            raise ValueError("A must contain stored non-positive -exp(A_log)")
        self.frame = array("f", [0.0]) * (shape.frame()["bytes"]//4)
        self.projections = array("f", [0.0]) * (shape.projections()["bytes"]//4)
        self.status = [PENDING] * shape.value_heads
        self.active_bank = 0
        self.tokens = 0
        self.max_tokens = max_tokens
        for head in range(shape.value_heads):
            self._write_status(head, PENDING)

    def _write_status(self, head, value):
        self.status[head] = value
        with memoryview(self.frame).cast("B") as raw:
            struct.pack_into("<I", raw, self.shape.frame()["status"]+4*head, value)

    def reset(self):
        """Clear both banks; real native reset must count its bounded host writes."""
        for i in range(2*self.shape.bank_bytes//4):
            self.frame[i] = 0.0
        self.active_bank = 0
        self.tokens = 0
        for head in range(self.shape.value_heads):
            self._write_status(head, PENDING)

    def set_projections(self, values):
        """Allow explicit nonfinite injections; the arithmetic step must refuse."""
        if len(values) != len(self.projections):
            raise ValueError("projection packet has wrong exact length")
        self.projections[:] = array("f", (_raw_f32(x) for x in values))

    def _slice(self, offset, count):
        return self.frame[offset:offset+count]

    def _get(self, offset, index):
        return self.frame[offset+index]

    def _set(self, offset, index, value):
        self.frame[offset+index] = value

    def active_state(self):
        """Explicit packed-F32 snapshot; avoid expanding ~817K cells to objects."""
        start = self.active_bank * self.shape.bank_bytes//4
        return self.frame[start:start+self.shape.bank_bytes//4]

    def step(self):
        if self.tokens >= self.max_tokens:
            raise ValueError("diagnostic token capacity exhausted before state mutation")
        g = self.shape
        p = {name: offset//4 for name, offset in g.parameters().items()}
        x = {name: offset//4 for name, offset in g.projections().items()}
        f = {name: offset//4 for name, offset in g.frame().items()}
        old = self.active_bank*g.bank_bytes//4
        new = (1-self.active_bank)*g.bank_bytes//4
        nan = math.nan
        weights, inputs = self.parameters, self.projections
        c, q, v, d, heads = (g.channels, g.key_width, g.value_width,
                            g.head_dim, g.value_heads)

        # Slot12 initializes all downstream refusal sentinels before lane work.
        for n in range(v):
            self._set(f["core"], n, nan)
        for head in range(heads):
            self._write_status(head, PENDING)
        for channel in range(c):
            history = old + 3*channel
            target = new + 3*channel
            raw = inputs[x["qkv"]+channel]
            self.frame[target:target+3] = array("f", [self.frame[history+1],
                                                    self.frame[history+2], raw])
            try:
                total = 0.0
                samples = (self.frame[history], self.frame[history+1],
                           self.frame[history+2], raw)
                for tap, value in enumerate(samples):
                    total = _add(total, _mul(value, weights[4*channel+tap]))
                convolved = _silu(total)
            except ArithmeticError:
                convolved = nan
            self._set(f["conv_qkv"], channel, convolved)

        # Slot13 group0 normalizes independently owned Q/K heads out of place.
        for which, target in ((0, f["normalized_q"]), (q, f["normalized_k"])):
            for head in range(g.key_heads):
                start = head*d
                values = self._slice(f["conv_qkv"]+which+start, d)
                try:
                    norm = _norm_scale(values, _round(g.epsilon/d))
                    reciprocal_dimension = _inverse_root(float(d))
                    result = [_mul(_mul(value, norm), reciprocal_dimension)
                              for value in values]
                except ArithmeticError:
                    result = [nan]*d
                self.frame[target+start:target+start+d] = array("f", result)

        # Slot13 group1 stores decay/beta, never partial/stale gate pairs.
        for head in range(heads):
            try:
                alpha = _add(inputs[x["alpha"]+head], weights[p["dt"]+head])
                log_decay = _mul(weights[p["a"]+head], _softplus(alpha))
                rho = _round(math.exp(log_decay))
                beta = _sigmoid(inputs[x["beta"]+head])
            except ArithmeticError:
                rho = beta = nan
            self._set(f["decay"], head, rho)
            self._set(f["beta"], head, beta)

        # Slot14 independently owns one recurrent row and one core coordinate.
        for n in range(v):
            head, coordinate = divmod(n, d)
            key_head = head % g.key_heads
            qbase = f["normalized_q"]+key_head*d
            kbase = f["normalized_k"]+key_head*d
            oldrow = old+f["bank_recurrent"]+n*d
            newrow = new+f["bank_recurrent"]+n*d
            rho = self._get(f["decay"], head)
            beta = self._get(f["beta"], head)
            try:
                prediction = 0.0
                for i in range(d):
                    decayed = _mul(self.frame[oldrow+i], rho)
                    prediction += _mul(decayed, self.frame[kbase+i])
                innovation = _add(self._get(f["conv_qkv"], 2*q+n),
                                  -_round(prediction))
                delta = _mul(innovation, beta)
                readout = 0.0
                for i in range(d):
                    decayed = _mul(self.frame[oldrow+i], rho)
                    updated = _add(decayed, _mul(self.frame[kbase+i], delta))
                    self.frame[newrow+i] = updated
                    readout += _mul(updated, self.frame[qbase+i])
                self._set(f["core"], n, _mul(_round(readout),
                                            _inverse_root(float(d))))
            except ArithmeticError:
                # Partial destination-bank state is deliberately unpublished.
                pass

        # Slot15 owns a whole head so its one status covers every coordinate.
        for head in range(heads):
            self._write_status(head, NONFINITE)
            core = self._slice(f["core"]+head*d, d)
            try:
                scale = _norm_scale(core, g.epsilon)
                for j, value in enumerate(core):
                    gate = _silu(inputs[x["z"]+head*d+j])
                    result = _mul(_mul(_mul(value, scale), weights[p["norm"]+j]), gate)
                    self.frame[new+f["bank_output"]+head*d+j] = result
                self._write_status(head, SUCCESS)
            except ArithmeticError:
                pass

        passed = all(status == SUCCESS for status in self.status)
        if passed:
            self.active_bank = 1-self.active_bank
            self.tokens += 1
            out = tuple(self.frame[new+f["bank_output"]:new+f["bank_output"]+v])
        else:
            out = None
        return StepResult(passed, self.active_bank, self.tokens, tuple(self.status), out)


def small_fixture(shape):
    """Return explicit, deterministic finite parameters and five changed inputs.

    Binary fractions keep input interpretation exact; asymmetric taps/head values
    expose history direction, head mapping, and norm/gate ordering. The caller
    decides whether and when to execute the fixture.
    """
    if not isinstance(shape, Geometry):
        raise ValueError("explicit Geometry required")
    params = [0.0] * (shape.parameters()["bytes"]//4)
    p = {name: offset//4 for name, offset in shape.parameters().items()}
    for channel in range(shape.channels):
        taps = (0.125, -0.25, 0.5, 0.75)
        for tap, value in enumerate(taps):
            params[4*channel+tap] = value * (-1 if channel % 3 == 1 else 1)
    for head in range(shape.value_heads):
        params[p["a"]+head] = -0.0 if head % 3 == 0 else -0.125*(1+head%5)
        params[p["dt"]+head] = (head%7-3)*0.125
    for j in range(shape.head_dim):
        params[p["norm"]+j] = 0.5+(j%5)*0.25
    params[p["epsilon"]] = shape.epsilon
    width = shape.projections()["bytes"]//4
    inputs = [[((i*7+token*11)%29-14)*0.0625 for i in range(width)]
              for token in range(5)]
    return params, inputs
