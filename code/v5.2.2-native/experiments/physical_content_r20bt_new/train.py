"""Train one official-fresh R20 task entry for its fixed400 updates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401
from physical_content.training import main

if __name__ == '__main__':
    main()
