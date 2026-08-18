"""Escritura atomica de artefactos y carga segura para inferencia."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import MLPConfig
from .model import RegressionMLP


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"No serializable: {type(value)}")


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(path: str | Path, model: torch.nn.Module) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, input_dim: int, config: MLPConfig) -> RegressionMLP:
    model = RegressionMLP(input_dim, config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def save_transformers(directory: str | Path, preprocessor, target_transformer) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    joblib.dump(target_transformer, directory / "target_transformer.joblib")


def load_transformers(directory: str | Path):
    directory = Path(directory)
    return joblib.load(directory / "preprocessor.joblib"), joblib.load(directory / "target_transformer.joblib")

