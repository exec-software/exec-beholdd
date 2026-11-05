"""Hardware-independent checks for the packet contract and scalar model."""

import math
import struct
import unittest

from packet_contract import Geometry, decode_scalars
from scalar_model import ScalarPacket, small_fixture, _sigmoid, _silu, _softplus


class PacketContractChecks(unittest.TestCase):
    def test_full_geometry_exact_extents_and_transport_scalars(self):
        shape = Geometry(16, 48, 128)
        self.assertEqual(shape.bank_bytes, 3293184)
        self.assertEqual(shape.parameters()["epsilon"], 164736)
        self.assertEqual(shape.parameters()["bytes"], 164740)
        self.assertEqual(shape.projections(), {"qkv": 0, "z": 40960,
                         "alpha": 65536, "beta": 65728, "bytes": 65920})
        self.assertEqual(shape.frame()["core"], 6644096)
        self.assertEqual(shape.frame()["status"], 6668672)
        self.assertEqual(shape.frame()["bytes"], 6668864)
        self.assertEqual(shape.scalars(0), (3048, 1))
        self.assertEqual(shape.scalars(1), (7144, 1))
        self.assertEqual(shape.dispatch_groups(), (160, 2, 96, 1))
        for bank in (0, 1):
            self.assertEqual(decode_scalars(*shape.scalars(bank), shape.epsilon),
                             (shape, bank))

    def test_smallest_shape_and_invalid_fields(self):
        self.assertEqual(Geometry(1, 1, 1).scalars(0), (1, 1))
        for fields in ((3, 3, 2), (2, 3, 2), (1, 49, 2), (1, 1, 3),
                       (True, 1, 1), (1, 1, 256)):
            with self.assertRaises(ValueError):
                Geometry(*fields)
        for scalars in ((0, 1), (1, 0), (1, 2), (8193, 1)):
            with self.assertRaises(ValueError):
                decode_scalars(*scalars, 1e-6)

    def test_hand_zero_key_preserves_recurrent_state_and_raw_history(self):
        shape = Geometry(1, 1, 1, 0.25)
        # Three current-only conv channels; A=-0,dt=0,norm=1,eps=0.25.
        params = [0.0, 0.0, 0.0, 1.0]*3 + [-0.0, 0.0, 1.0, 0.25]
        model = ScalarPacket(shape, params)
        model.frame[0:3] = model.frame.__class__("f", [1.0, 2.0, 3.0])
        recurrent = shape.frame()["bank_recurrent"]//4
        model.frame[recurrent] = 2.0
        model.set_projections([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        result = model.step()
        self.assertTrue(result.committed)
        self.assertEqual(result.output, (0.0,))
        base = shape.bank_bytes//4
        self.assertEqual(tuple(model.frame[base:base+3]), (2.0, 3.0, 0.0))
        self.assertEqual(model.frame[base+recurrent], 2.0)
        with memoryview(model.frame).cast("B") as raw:
            self.assertEqual(struct.unpack_from("<I", raw, shape.frame()["status"])[0], 0)

    def test_late_refusal_does_not_publish_and_retry_matches_fresh(self):
        shape = Geometry(2, 4, 2)
        params, inputs = small_fixture(shape)
        failed = ScalarPacket(shape, params)
        fresh = ScalarPacket(shape, params)
        previous = failed.active_state()
        bad = list(inputs[0])
        bad[shape.projections()["z"]//4] = math.nan
        failed.set_projections(bad)
        result = failed.step()
        self.assertFalse(result.committed)
        self.assertEqual((failed.active_bank, failed.tokens), (0, 0))
        self.assertEqual(failed.active_state(), previous)
        failed.set_projections(inputs[0])
        fresh.set_projections(inputs[0])
        self.assertEqual(failed.step(), fresh.step())
        self.assertEqual(failed.active_state(), fresh.active_state())

    def test_capacity_refusal_and_reset_replay(self):
        shape = Geometry(2, 4, 2)
        params, inputs = small_fixture(shape)
        model = ScalarPacket(shape, params, max_tokens=5)
        outputs = []
        for values in inputs:
            model.set_projections(values)
            result = model.step()
            self.assertTrue(result.committed)
            outputs.append(result.output)
        before = model.active_state()
        with self.assertRaises(ValueError):
            model.step()
        self.assertEqual(model.active_state(), before)
        model.reset()
        self.assertEqual(model.tokens, 0)
        self.assertTrue(all(value == 0.0 for value in model.active_state()))
        model.set_projections(inputs[0])
        self.assertEqual(model.step().output, outputs[0])

    def test_silu_retains_literal_f32_exp_boundary(self):
        self.assertGreater(_sigmoid(-90.0), 0.0)
        self.assertLess(_silu(-90.0), 0.0)
        # The oracle must not silently fuse z*exp(z) before rounding F32 exp.
        self.assertEqual(struct.pack("<f", _silu(-105.0)), struct.pack("<I", 0x80000000))
        # log1p must retain the least subnormal, even when t/2 would round away.
        self.assertEqual(struct.pack("<f", _softplus(-103.0)), struct.pack("<I", 1))


if __name__ == "__main__":
    unittest.main()
