from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ace_router.paths import V1_ROOT, require_write_path
from ace_router.schedules import layer_gates_for_preset, step_gates
from ace_router.sizing import best_output_size, prepare_reference_image


class ScheduleTests(unittest.TestCase):
    def test_mvp_mid16(self) -> None:
        gates = layer_gates_for_preset("mvp_mid16")
        self.assertEqual(len(gates), 30)
        self.assertEqual(sum(value > 0 for value in gates), 16)
        self.assertEqual(gates[7], 0.0)
        self.assertEqual(gates[8], 0.4)
        self.assertEqual(gates[14], 1.0)
        self.assertEqual(gates[23], 1.0)
        self.assertEqual(gates[24], 0.0)

    def test_scan_windows_are_half_open(self) -> None:
        expected = {
            "early": set(range(0, 10)),
            "mid_a": set(range(10, 20)),
            "mid_b": set(range(14, 24)),
            "late": set(range(20, 30)),
            "all": set(range(30)),
        }
        for preset, expected_indices in expected.items():
            gates = layer_gates_for_preset(preset)
            actual_indices = {
                index for index, value in enumerate(gates) if value > 0.0
            }
            self.assertEqual(actual_indices, expected_indices, preset)

    def test_paper_soft30_is_active_on_all_layers(self) -> None:
        gates = layer_gates_for_preset("paper_soft30")
        self.assertEqual(len(gates), 30)
        self.assertTrue(all(value > 0.0 for value in gates))

    def test_fifty_step_boundaries(self) -> None:
        gates = step_gates(50)
        self.assertEqual(gates[:15], [1.0] * 15)
        self.assertEqual(gates[15:30], [0.7] * 15)
        self.assertEqual(gates[30:40], [0.3] * 10)
        self.assertEqual(gates[40:], [0.0] * 10)

    def test_forty_step_boundaries(self) -> None:
        gates = step_gates(40)
        self.assertEqual(gates[:12], [1.0] * 12)
        self.assertEqual(gates[12:24], [0.7] * 12)
        self.assertEqual(gates[24:32], [0.3] * 8)
        self.assertEqual(gates[32:], [0.0] * 8)

    def test_safe_i2v_step_preset_keeps_small_terminal_gate(self) -> None:
        gates = step_gates(50, preset="safe_i2v_v1")
        self.assertEqual(gates[:15], [0.5] * 15)
        self.assertEqual(gates[15:35], [1.0] * 20)
        self.assertEqual(gates[35:45], [0.3] * 10)
        self.assertEqual(gates[45:], [0.1] * 5)

    def test_safe_t2v_step_preset_is_more_conservative_early(self) -> None:
        gates = step_gates(50, preset="safe_t2v_v1")
        self.assertEqual(gates[:15], [0.25] * 15)
        self.assertEqual(gates[15:35], [0.8] * 20)
        self.assertEqual(gates[35:45], [0.3] * 10)
        self.assertEqual(gates[45:], [0.1] * 5)


class SizingTests(unittest.TestCase):
    def test_exact_smoke_sizes(self) -> None:
        self.assertEqual(best_output_size(1280, 704, 32, 32, 640 * 352), (640, 352))
        self.assertEqual(best_output_size(704, 1280, 32, 32, 640 * 352), (352, 640))
        self.assertEqual(
            best_output_size(1280, 704, 32, 32, 1280 * 704), (1280, 704)
        )

    def test_reference_crop_preserves_orientation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.png"
            Image.new("RGB", (1536, 1024), color=(1, 2, 3)).save(path)
            prepared = prepare_reference_image(path)
            try:
                self.assertEqual(prepared.orientation, "landscape")
                self.assertEqual(prepared.prepared_size, (1280, 704))
                self.assertEqual(prepared.image.size, (1280, 704))
                self.assertEqual(prepared.crop_box[2] - prepared.crop_box[0], 1536)
                self.assertLess(prepared.crop_box[3] - prepared.crop_box[1], 1024)
            finally:
                prepared.image.close()


class PathTests(unittest.TestCase):
    def test_write_path_is_scoped_to_v1(self) -> None:
        accepted = require_write_path(V1_ROOT / "outputs" / "test.mp4")
        self.assertTrue(str(accepted).startswith(str(V1_ROOT)))
        with self.assertRaises(ValueError):
            require_write_path("/home/liuzhirui/model/Wan2.2/output.mp4")
        with self.assertRaises(ValueError):
            require_write_path("/tmp/ace-output.mp4")


if __name__ == "__main__":
    unittest.main()
