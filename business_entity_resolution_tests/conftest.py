import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "code" / "business_entity_resolution"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
