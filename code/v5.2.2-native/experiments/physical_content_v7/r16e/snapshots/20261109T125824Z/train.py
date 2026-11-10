from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.train import main
if __name__=='__main__':main()
