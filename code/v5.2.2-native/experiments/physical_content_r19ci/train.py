"""Launch one R18E-H NoChunks-v2 run; it does not auto-submit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401
from physical_content.training import main

if __name__ == '__main__':
    main()
