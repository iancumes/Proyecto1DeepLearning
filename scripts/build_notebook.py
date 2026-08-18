"""Reconstruye el notebook a partir de resultados ya validados."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.report import build_notebook

print(build_notebook(ROOT))

