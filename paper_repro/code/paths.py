"""Locations for the paper bundle. Scripts in this folder import these.

The bundle is self-contained. Checkpoints are not included. The series the
figures read were copied or reduced into ``data/``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "sealed" / "nexus_matrix_review_2026-09-08_18-28.zip"
PACK = DATA / "pack"
LOGS = DATA / "logs"
FIGURES = ROOT / "figures"
