"""Launch one R18 A/B/C/D run; this file does not auto-submit or auto-train."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401
from physical_content.training import main

if __name__ == '__main__':
    main()
