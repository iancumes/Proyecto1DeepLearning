"""Ejecuta el notebook de entrega desde la raiz del proyecto."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.report import execute_notebook

path = ROOT / "notebooks" / "Proyecto1_MLP_Ames_IanCumes_23236.ipynb"
execute_notebook(path, ROOT)
print(path)

