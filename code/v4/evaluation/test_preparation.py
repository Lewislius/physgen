"""CPU regressions for preprocessing restart and early failure behavior."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml

from physgen_v4.cache import (atomic_write, audit_cache, cache_error, check_manifest_config, exclude_short_views,
                             finalize_cache, preserve_manifest, save_json, save_tensor)
from tools.prepare_wisa import main as prepare
from physgen_v4.environments import python_for, subprocess_environment

torch.set_num_threads(2)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for stage in ("vae", "text", "teacher", "views"):
            (self.root / stage).mkdir()
        self.cfg = dict(max_frames=101, teacher_frames=4, teacher_size=32)
        self.view = dict(frames=5, height=32, width=32, teacher_frames=4,
                         actual_times=[0., .25, .5, .75, 1.])
        self.record = dict(index=0, caption="a ball", video_name="0.mp4", split="train")
        self.paths = dict(cache_root=str(self.root), wan_code="/unused/wan", teacher_code="/unused/teacher")
        self.manifest = dict(data=self.cfg, paths=self.paths, records=[self.record], teacher=dict(channels=1664))
        save_json(self.manifest, self.root / "manifest.json")
        save_json(self.view, self.root / "views/0000000.json")
        self.vae = dict(latent=torch.zeros(1, 48, 2, 2, 2), first=torch.zeros(1, 48, 1, 2, 2),
                        times=torch.tensor(self.view["actual_times"]), target_weight=torch.ones(1, 8),
                        video_metadata=self.view)
        save_tensor(self.vae, self.root / "vae/0000000.pt")
        save_tensor(torch.zeros(2, 4096), self.root / "text/0000000.pt")
        save_tensor(torch.zeros(1, 4096), self.root / "null_text.pt")
        save_tensor(torch.zeros(1, 8, 1664), self.root / "teacher/0000000.pt")

    def test_complete_cache_is_validated_and_finalized(self):
        result = finalize_cache(self.root, self.manifest)
        self.assertTrue(result["complete"])
        persisted = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(persisted["records"][0]["view"], self.view)

    def test_incomplete_teacher_does_not_publish_manifest(self):
        (self.root / "teacher/0000000.pt").unlink()
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            finalize_cache(self.root, self.manifest)
        self.assertEqual(before, (self.root / "manifest.json").read_bytes())

    def test_caption_change_refuses_to_mix_existing_caches(self):
        changed = copy.deepcopy(self.manifest)
        changed["records"][0]["caption"] = "a different event"
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            preserve_manifest(self.root, changed)
        self.assertEqual(before, (self.root / "manifest.json").read_bytes())

    def test_compatible_index_keeps_finalized_views(self):
        fresh = copy.deepcopy(self.manifest)
        finalize_cache(self.root, self.manifest)
        self.assertEqual(preserve_manifest(self.root, fresh)["records"][0]["view"], self.view)

    def test_changed_configuration_rejected(self):
        config = dict(data={**self.cfg, "teacher_frames": 16}, paths=self.paths)
        with self.assertRaisesRegex(ValueError, "Configuration differs"):
            check_manifest_config(self.manifest, config)

    def test_orphan_cache_is_not_adopted(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "without"):
            preserve_manifest(self.root, self.manifest)

    def test_truncated_tensor_is_not_reused(self):
        (self.root / "text/0000000.pt").write_bytes(b"PK truncated")
        self.assertIsNotNone(cache_error(self.root, "text", self.record, self.cfg))
        self.assertFalse(audit_cache(self.root, self.manifest)["complete"])

    def test_nonfinite_teacher_is_not_reused(self):
        value = torch.zeros(1, 8, 1664)
        value[0, 0, 0] = float("nan")
        save_tensor(value, self.root / "teacher/0000000.pt")
        self.assertIn("NaN/Inf", cache_error(self.root, "teacher", self.record, self.cfg))

    def test_wrong_teacher_token_count_is_rejected(self):
        save_tensor(torch.zeros(1, 6, 1664), self.root / "teacher/0000000.pt")
        self.assertIn("shape", cache_error(self.root, "teacher", self.record, self.cfg))

    def test_inconsistent_view_is_rejected(self):
        save_json({**self.view, "source_fps": 24}, self.root / "views/0000000.json")
        self.assertIn("metadata differs", cache_error(self.root, "vae", self.record, self.cfg))

    def test_null_text_must_be_valid(self):
        save_tensor(torch.zeros(1, 100), self.root / "null_text.pt")
        self.assertFalse(audit_cache(self.root, self.manifest)["complete"])

    def test_failed_atomic_save_preserves_previous_result(self):
        target = self.root / "text/0000000.pt"
        before = target.read_bytes()
        def fail(path):
            Path(path).write_bytes(b"partial write")
            raise OSError("disk full")
        with self.assertRaises(OSError):
            atomic_write(target, fail)
        self.assertEqual(before, target.read_bytes())
        self.assertFalse(list(target.parent.glob("*.tmp")))

    def test_completed_pass_does_not_initialize_cuda_or_load_encoder(self):
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(dict(data=self.cfg, paths=self.paths)))
        for stage in ("vae", "text", "teacher"):
            with self.subTest(stage=stage):
                role = 'teacher' if stage == 'teacher' else 'runtime'
                environment = subprocess_environment(role)
                environment.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
                probe = '''
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv.pop(1))
from tools.prepare_wisa import main
with patch('torch.cuda.set_device') as set_device:
    main()
    set_device.assert_not_called()
'''
                result = subprocess.run([python_for(role), '-I', '-B', '-c', probe,
                                         str(Path(__file__).resolve().parents[1]),
                                         '--config', str(config), '--pass', stage],
                                        env=environment, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads(result.stdout)["pending"], 0)
                self.assertEqual(json.loads(result.stdout)["reused"], 1)

    def test_short_clips_are_explicitly_excluded_without_changing_selection(self):
        save_json(dict(source_frames=3, frames=1), self.root / "views/0000000.json")
        before = copy.deepcopy(self.manifest)
        excluded = exclude_short_views(self.root, self.manifest, require_all_views=True)
        self.assertEqual(excluded[0]["index"], 0)
        self.assertEqual(len(self.manifest["records"]), 1)
        self.assertEqual(preserve_manifest(self.root, before)["records"][0]["excluded_reason"]["frames"], 1)
        # An entirely unusable dataset must never be considered complete.
        self.assertFalse(audit_cache(self.root, self.manifest)["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
