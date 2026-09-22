"""Run the actual inference preflight with all sibling v4.2 file access forbidden."""
import importlib.abc
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = str(ROOT.parent / "v4.2")


def guard_path(path):
    if isinstance(path, (str, bytes, os.PathLike)):
        value = os.path.abspath(os.fsdecode(path))
        if value == FORBIDDEN or value.startswith(FORBIDDEN + os.sep):
            raise RuntimeError(f"Standalone check caught a forbidden sibling dependency: {value}")


def audit(event, args):
    if event in ("open", "os.listdir", "os.scandir"):
        guard_path(args[0])


class RejectSiblingImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("physgen_v42", "physgen_v4"):
            raise RuntimeError(f"Standalone check caught a forbidden import: {fullname}")
        return None


if __name__ == "__main__":
    sys.addaudithook(audit)
    sys.meta_path.insert(0, RejectSiblingImport())
    original_stat = Path.stat
    def checked_stat(self, *args, **kwargs):
        guard_path(self)
        return original_stat(self, *args, **kwargs)
    Path.stat = checked_stat
    script = ROOT / "inference/infer.py"
    sys.argv = [str(script), "--phase", "check", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    print("[standalone] PASS: inference preflight completed with all v4.2 reads/imports forbidden.", flush=True)
