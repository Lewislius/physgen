import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


class RecordedRunTests(unittest.TestCase):
    def test_signal_termination_is_preserved_without_claiming_oom(self):
        spec=importlib.util.spec_from_file_location('recorded',Path(__file__).resolve().parents[1]/'tools/run_recorded.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resources.jsonl'
            status=module.run([sys.executable,'-c','import os, signal; os.kill(os.getpid(), signal.SIGKILL)'],path,.01)
            records=[json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(status,137)
        self.assertEqual(records[-1]['terminating_signal'],9)
        self.assertEqual(records[-1]['returncode'],-9)
        self.assertNotIn('oom_diagnosed',records[-1])


if __name__=='__main__':unittest.main()
