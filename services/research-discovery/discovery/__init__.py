"""Research Discovery internals.

The repository root goes on sys.path here, once, so every module can import
shared.utilities (the scholarly gateway, the LLM provider).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
