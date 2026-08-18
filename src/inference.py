"""Inferencia desde artefactos serializados."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_model, load_transformers, read_json
from .config import EXPECTED_FEATURES, ID_COLUMN, MLPConfig, TARGET
from .data import load_data
from .training import predict_scaled, regression_metrics

OUTPUT_COLUMN = "Prediction"


def predict_file(input_path: str | Path, artifact_dir: str | Path, mode: str = "single") -> tuple[pd.DataFrame, dict]:
    frame = load_data(input_path, require_target=False)
    labels = frame[TARGET].astype(float).copy() if TARGET in frame else None
    X = frame[EXPECTED_FEATURES]
    artifact_dir = Path(artifact_dir)
    metadata = read_json(artifact_dir / "metadata.json")
    preprocessor, target_transformer = load_transformers(artifact_dir)
    matrix = preprocessor.transform(X).astype(np.float32)
    config = MLPConfig.from_dict(metadata["model_config"])
    model_paths = [artifact_dir / "single_model.pt"]
    if mode == "ensemble":
        model_paths = [artifact_dir / "ensemble" / f"model_seed_{seed}.pt" for seed in metadata["seeds"]]
        if not model_paths:
            raise FileNotFoundError("No se encontraron pesos del ensemble.")
        missing_models = [str(path) for path in model_paths if not path.exists()]
        if missing_models:
            raise FileNotFoundError(f"Faltan pesos declarados del ensemble: {missing_models}")
    predictions = []
    for model_path in model_paths:
        model = load_model(model_path, matrix.shape[1], config)
        predictions.append(target_transformer.inverse_transform(predict_scaled(model, matrix)))
    values = np.mean(predictions, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("El modelo produjo valores no finitos.")
    output = pd.DataFrame({ID_COLUMN: frame[ID_COLUMN].to_numpy(), OUTPUT_COLUMN: np.clip(values, 0.0, None)})
    metrics = regression_metrics(labels, values) if labels is not None else {}
    return output, metrics


def apply_output_template(predictions: pd.DataFrame, template_path: str | Path) -> pd.DataFrame:
    template = pd.read_csv(template_path)
    expected_columns = [ID_COLUMN, OUTPUT_COLUMN]
    if template.columns.tolist() != expected_columns:
        raise ValueError(
            f"El template debe tener exactamente las columnas {expected_columns} en ese orden; "
            f"recibidas={template.columns.tolist()}"
        )
    if template[ID_COLUMN].isna().any() or template[ID_COLUMN].duplicated().any():
        raise ValueError("El template contiene Id nulos o duplicados.")
    prediction_ids = predictions[ID_COLUMN].tolist()
    template_ids = template[ID_COLUMN].tolist()
    if prediction_ids != template_ids:
        raise ValueError(
            "Los Id o su orden no coinciden entre pipeline_test.csv y expected_output.csv. "
            f"entrada={prediction_ids}; template={template_ids}"
        )
    output = template.copy()
    output[OUTPUT_COLUMN] = predictions[OUTPUT_COLUMN].to_numpy(dtype=float)
    return output[expected_columns]
