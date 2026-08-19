from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import nbformat

from src.config import EXPECTED_FEATURES, MLPConfig, PreprocessConfig, TARGET
from src.data import clean_frame, development_test_split, load_data, validate_schema
from src.model import RegressionMLP
from src.inference import apply_output_template, predict_file
from src.preprocessing import TargetTransformer, build_preprocessor
from src.training import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def training_frame():
    path = ROOT / "train.csv"
    if not path.exists():
        pytest.skip("train.csv no está disponible")
    return load_data(path)


def test_cleaning_removes_wrapped_apostrophes():
    frame = pd.DataFrame({"x": [" 'Wd Sdng' ", "RL", np.nan], "Alley": [np.nan, "Pave", np.nan]})
    cleaned = clean_frame(frame)
    assert cleaned.loc[0, "x"] == "Wd Sdng"
    assert cleaned.loc[0, "Alley"] == "None"


def test_schema_and_ids(training_frame):
    validate_schema(training_frame, require_target=True)
    assert list(training_frame[EXPECTED_FEATURES].columns) == EXPECTED_FEATURES
    assert training_frame["Id"].is_unique


def test_split_is_deterministic_and_disjoint(training_frame):
    first = development_test_split(training_frame, 23236)
    second = development_test_split(training_frame, 23236)
    assert first.X_dev.index.tolist() == second.X_dev.index.tolist()
    assert set(first.X_dev.index).isdisjoint(first.X_test.index)
    assert len(first.X_dev) + len(first.X_test) == len(training_frame)


def test_preprocessor_handles_unknown_category(training_frame):
    train = training_frame[EXPECTED_FEATURES].iloc[:100].copy()
    test = training_frame[EXPECTED_FEATURES].iloc[100:105].copy()
    test.loc[test.index[0], "Neighborhood"] = "CategoriaNuncaVista"
    preprocessor = build_preprocessor(train, PreprocessConfig())
    train_matrix = preprocessor.fit_transform(train)
    test_matrix = preprocessor.transform(test)
    assert train_matrix.shape[1] == test_matrix.shape[1]
    assert np.isfinite(test_matrix).all()


def test_target_roundtrip_and_metrics():
    y = np.array([100_000.0, 150_000.0, 250_000.0])
    transformer = TargetTransformer("log1p").fit(y)
    restored = transformer.inverse_transform(transformer.transform(y))
    assert np.allclose(restored, y, rtol=1e-5)
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_mlp_output_shape():
    config = MLPConfig(hidden_layers=(16, 8), batch_norm=False)
    model = RegressionMLP(12, config)
    output = model(torch.zeros(7, 12))
    assert output.shape == (7,)


def test_missing_feature_fails(training_frame, tmp_path):
    invalid = training_frame.drop(columns=["Neighborhood"])
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Faltantes"):
        load_data(path, require_target=True)


def test_smoke_artifacts_and_notebook_if_available():
    metadata = ROOT / "artifacts" / "metadata.json"
    notebook_path = ROOT / "notebooks" / "Proyecto1_MLP_Ames_IanCumes_23236.ipynb"
    if not metadata.exists() or not notebook_path.exists():
        pytest.skip("El smoke pipeline todavía no se ha ejecutado")
    notebook = nbformat.read(notebook_path, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for section in ["2.1 Análisis exploratorio", "2.2 Metodología", "2.3 Resultados", "2.4 Discusión", "2.5 Conclusiones", "2.6 Enlace"]:
        assert section in markdown
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)


def test_professor_pipeline_and_output_contract_if_available(tmp_path):
    input_path = ROOT / "pipeline_test.csv"
    template_path = ROOT / "expected_output.csv"
    artifact_dir = ROOT / "artifacts"
    if not input_path.exists() or not template_path.exists() or not (artifact_dir / "metadata.json").exists():
        pytest.skip("Los archivos del profesor o los artefactos no están disponibles")
    raw_input = pd.read_csv(input_path)
    predictions, metrics = predict_file(input_path, artifact_dir, mode="single")
    template = pd.read_csv(template_path)
    assert metrics == {}
    assert template.columns.tolist() == ["Id", "Prediction"]
    assert predictions.columns.tolist() == ["Id", "Prediction"]
    assert predictions["Id"].tolist() == raw_input["Id"].tolist()
    assert np.isfinite(predictions["Prediction"]).all()
    assert (predictions["Prediction"] > 0).all()

    if len(template) == len(predictions):
        output = apply_output_template(predictions, template_path)
        assert output["Id"].tolist() == raw_input["Id"].tolist()
    else:
        # El archivo del profesor puede ser una muestra del contrato de salida.
        assert template["Id"].tolist() == predictions["Id"].head(len(template)).tolist()

    wrong_template = template.iloc[::-1]
    wrong_path = tmp_path / "wrong_template.csv"
    wrong_template.to_csv(wrong_path, index=False)
    with pytest.raises(ValueError, match="orden no coinciden"):
        apply_output_template(predictions, wrong_path)
