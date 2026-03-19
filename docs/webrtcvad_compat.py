"""Forward to the repo-level webrtcvad compatibility shim."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webrtcvad_compat import *  # noqa: F401,F403
