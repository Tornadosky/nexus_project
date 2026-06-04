"""Thin wrapper for the continuous-control NEXUS extension."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    from nexus_continuous.scripts.train_nexus_playground import main

    main()
