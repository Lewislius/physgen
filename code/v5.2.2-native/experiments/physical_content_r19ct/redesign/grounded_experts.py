"""Compatibility import for local architecture checks; canonical code is integrated."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physical_content.grounded_experts import SemanticPhysicalEncoder, PhysicalStateExpert, GroundedPhysicalFusion
