"""Make sibling modules importable by bare name when running pytest from the repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
