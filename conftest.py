"""Root conftest — ensures src/ is importable in all environments."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'from src.xxx import ...' works
# regardless of how pip install is configured
sys.path.insert(0, str(Path(__file__).parent))
