"""Construccion y ejecucion segura del notebook de entrega."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from .artifacts import read_json


def _money(value: float) -> str:
    return f"USD {value:,.0f}"


def build_notebook(root: str | Path) -> Path:
    root = Path(root)
    results = root / "results"
    notebook_dir = root / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    quality = read_json(results / "data_quality.json")
    final = read_json(results / "final_metrics.json")
    selected = final["selected_candidate"]
    single = final["internal_test"]["single"]
    ensemble = final["internal_test"]["ensemble"]
    run_mode = final["run_mode"]
    optuna_summary = final.get("optuna", {"total_trials": 0, "state_counts": {}})
    predictions = pd.read_csv(results / "internal_test_predictions.csv")
    predictions["AbsoluteError"] = predictions["ResidualSingle"].abs()
    worst = predictions.nlargest(1, "AbsoluteError").iloc[0]
    predictions["PriceQuartile"] = pd.qcut(predictions["SalePrice"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    quartile_rmse = predictions.groupby("PriceQuartile", observed=True).apply(
        lambda part: float(np.sqrt(np.mean((part["SalePrice"] - part["PredictionSingle"]) ** 2))),
        include_groups=False,
    )
    ensemble_gain = 100 * (single["rmse"] - ensemble["rmse"]) / single["rmse"]

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    cells = []
    cells.append(nbformat.v4.new_markdown_cell(f"""# Proyecto 1 — Competencia de modelación con MLP

**Curso:** CC3092 Deep Learning y sistemas inteligentes  
**Estudiante:** Ian Cumes — 23236  
**Dataset:** `train.csv` suministrado por el curso  
**Modo de ejecución registrado:** `{run_mode}`

## tl;dr

Se entrenó un MLP de regresión con preprocesamiento ajustado exclusivamente dentro de cada partición. El modelo individual obtuvo **{_money(single['rmse'])} de RMSE** en el test interno aislado; el ensemble predefinido de cinco semillas obtuvo **{_money(ensemble['rmse'])}**. La configuración se eligió antes de consultar ese test. Estos valores son evidencia interna, no una promesa sobre el held-out del curso.
"""))
    cells.append(nbformat.v4.new_markdown_cell("""## Contexto y método

El objetivo es minimizar RMSE en dólares para un archivo held-out sin `SalePrice`. Solo se utilizó el dataset entregado; no se consultaron etiquetas públicas de los 292 `Id` ausentes. `Id` se conserva para trazabilidad, pero no entra al MLP.

### Supuestos clave

- La unidad de análisis es una vivienda vendida.
- El held-out conserva las mismas 80 features.
- El test interno del 20% permanece fuera de selección y se consulta una sola vez.
- El ensemble es un artefacto opcional y solo debe emplearse si el profesor lo autoriza.
"""))
    cells.append(nbformat.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import numpy as np
from IPython.display import Image, display

ROOT = Path.cwd()
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
assert (ROOT / "train.csv").exists(), "Ejecute el notebook desde la raíz del repositorio."

data = pd.read_csv(ROOT / "train.csv")
quality = json.loads((RESULTS / "data_quality.json").read_text(encoding="utf-8"))
experiments = pd.read_csv(RESULTS / "experiments.csv")
final_metrics = json.loads((RESULTS / "final_metrics.json").read_text(encoding="utf-8"))
print(f"Dataset: {data.shape[0]:,} filas × {data.shape[1]} columnas")
print(f"Huella SHA-256: {final_metrics['data_sha256']}")
"""))
    cells.append(nbformat.v4.new_markdown_cell("""## 2.1 Análisis exploratorio de datos (EDA)

La auditoría cubre dimensiones, tipos, unicidad, nulos, consistencia y outliers. Los nulos de elementos físicos ausentes se codifican como `None`; el resto se imputa dentro de cada fold. No se eliminan masivamente outliers con IQR porque varias variables son cero-infladas.
"""))
    cells.append(nbformat.v4.new_code_cell("""profile = pd.DataFrame({
    "Métrica": ["Filas", "Features numéricas", "Features categóricas", "Filas duplicadas", "Id duplicados", "Celdas nulas", "Columnas con nulos", "Asimetría SalePrice", "Asimetría log1p(SalePrice)"],
    "Valor": [quality["clean"]["rows"], quality["clean"]["numeric_features"], quality["clean"]["categorical_features"], quality["clean"]["duplicate_rows"], quality["clean"]["duplicate_ids"], quality["clean"]["total_missing"], quality["clean"]["columns_with_missing"], round(quality["clean"]["target_skew"], 3), round(quality["clean"]["log_target_skew"], 3)]
})
display(profile)
display(data[["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]].describe().T)
display(pd.DataFrame(quality["raw"]["extreme_living_area"]))
print("Categorías con apóstrofos antes de limpiar:", quality["raw"]["remaining_wrapped_apostrophes"])
print("Categorías con apóstrofos después de limpiar:", quality["clean"]["remaining_wrapped_apostrophes"])
"""))
    cells.append(nbformat.v4.new_code_cell("""for figure in ["target_distribution.png", "missingness.png", "correlations.png"]:
    display(Image(filename=str(FIGURES / figure)))
display(Image(filename=str(FIGURES / "categorical_associations.png")))
"""))
    cells.append(nbformat.v4.new_markdown_cell("""### Decisiones derivadas del EDA

- `SalePrice` presenta cola derecha; se compara target estandarizado directo contra `log1p`.
- Las variables categóricas requieren manejo explícito de categorías desconocidas.
- La ausencia de garaje, sótano o piscina es información, no un error de captura.
- Los casos extremos se conservan por defecto y se compara winsorización ajustada solo con entrenamiento.
"""))
    cells.append(nbformat.v4.new_markdown_cell(f"""## 2.2 Metodología de desarrollo

Se separó un test interno estratificado del 20%. Sobre el 80% restante se ejecutaron baselines, diez ablaciones y **{optuna_summary['total_trials']} trials acumulados de Optuna** con validación cruzada de tres folds ({optuna_summary['state_counts'].get('COMPLETE', 0)} completos y {optuna_summary['state_counts'].get('PRUNED', 0)} podados). Los finalistas se reevaluaron con cinco folds y dos semillas. Cada fold ajusta su propio imputador, encoder, escalador y transformación del target.

El MLP permite 1–4 capas, cinco activaciones, BatchNorm, Dropout, MSE o SmoothL1, cuatro optimizadores, scheduler, weight decay, clipping de gradiente y early stopping. La salida es lineal y todas las métricas se calculan después de volver a dólares.
"""))
    cells.append(nbformat.v4.new_code_cell(f"""selected = final_metrics["selected_candidate"]
print("Configuración seleccionada:")
display(pd.json_normalize(selected).T.rename(columns={{0: "valor"}}))
print("Épocas finales:", final_metrics["final_epochs"])
"""))
    cells.append(nbformat.v4.new_markdown_cell("""## 2.3 Resultados de iteraciones

La tabla conserva baselines, cambios controlados y finalistas. El RMSE de validación es la métrica de selección; el test interno no aparece en esta tabla para evitar mezclar selección con evaluación final.
"""))
    cells.append(nbformat.v4.new_code_cell("""columns = [c for c in ["id", "stage", "train_rmse", "val_rmse", "val_mae", "val_r2", "best_epoch"] if c in experiments]
display(experiments[columns].sort_values("val_rmse", na_position="last").head(20).style.format({
    "train_rmse": "{:,.0f}", "val_rmse": "{:,.0f}", "val_mae": "{:,.0f}", "val_r2": "{:.3f}"
}, na_rep="—"))
display(Image(filename=str(FIGURES / "experiment_comparison.png")))
display(Image(filename=str(FIGURES / "learning_curve.png")))
"""))
    cells.append(nbformat.v4.new_markdown_cell(f"""## 2.4 Discusión de resultados

El test interno contiene {final['internal_test']['rows']} viviendas. El MLP individual obtuvo RMSE {_money(single['rmse'])}, MAE {_money(single['mae'])} y R² {single['r2']:.3f}. El ensemble obtuvo RMSE {_money(ensemble['rmse'])}, MAE {_money(ensemble['mae'])} y R² {ensemble['r2']:.3f}.

El ensemble redujo el RMSE interno en **{ensemble_gain:.1f}%** frente al modelo individual. El mayor error absoluto individual fue la vivienda `Id={int(worst['Id'])}`: precio real {_money(worst['SalePrice'])}, predicción {_money(worst['PredictionSingle'])} y error absoluto {_money(worst['AbsoluteError'])}. El RMSE del cuartil de precios altos fue {_money(quartile_rmse['Q4'])}, evidencia de que los casos caros dominan parte importante del error cuadrático.

El gráfico de residuos permite comprobar heterocedasticidad y errores en precios altos. Dado el tamaño reducido del dataset, la incertidumbre entre folds y semillas importa: una mejora pequeña no debe interpretarse como universal. El MLP también puede quedar por debajo de métodos de boosting en datos tabulares; Ridge se conserva como control, no como candidato de competencia.
"""))
    cells.append(nbformat.v4.new_code_cell("""display(pd.DataFrame(final_metrics["internal_test"]).T[["rmse", "mae", "r2"]].style.format({"rmse": "{:,.0f}", "mae": "{:,.0f}", "r2": "{:.3f}"}))
display(Image(filename=str(FIGURES / "residuals.png")))
errors = pd.read_csv(RESULTS / "internal_test_predictions.csv")
errors["AbsoluteError"] = errors["ResidualSingle"].abs()
errors["PriceQuartile"] = pd.qcut(errors["SalePrice"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
quartiles = errors.groupby("PriceQuartile", observed=True).apply(lambda part: pd.Series({
    "Viviendas": len(part),
    "RMSE": np.sqrt(np.mean((part["SalePrice"] - part["PredictionSingle"]) ** 2)),
    "MAE": np.mean(np.abs(part["SalePrice"] - part["PredictionSingle"])),
    "Sesgo_medio": np.mean(part["SalePrice"] - part["PredictionSingle"]),
}), include_groups=False)
display(quartiles.style.format({"Viviendas": "{:.0f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}", "Sesgo_medio": "{:,.0f}"}))
details = data[["Id", "Neighborhood", "OverallQual", "GrLivArea", "YearBuilt"]]
display(errors.nlargest(10, "AbsoluteError").merge(details, on="Id")[["Id", "Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "SalePrice", "PredictionSingle", "ResidualSingle", "AbsoluteError"]].style.format({
    "SalePrice": "{:,.0f}", "PredictionSingle": "{:,.0f}", "ResidualSingle": "{:,.0f}", "AbsoluteError": "{:,.0f}"
}))
"""))
    cells.append(nbformat.v4.new_markdown_cell(f"""## 2.5 Conclusiones

1. El mejor MLP individual alcanzó **{_money(single['rmse'])} de RMSE interno** sin usar el test para seleccionar hiperparámetros.
2. El ensemble {'mejoró' if ensemble['rmse'] < single['rmse'] else 'no mejoró'} el RMSE interno frente al modelo individual; su uso queda condicionado a las reglas del profesor.
3. Las decisiones con mayor rigor son el ajuste de transformadores por fold, la evaluación final única y la persistencia conjunta de preprocesador, target y pesos.
4. Las limitaciones principales son el tamaño muestral, categorías raras y sensibilidad de RMSE a viviendas extremas.
"""))
    cells.append(nbformat.v4.new_markdown_cell("""## 2.6 Enlace al repositorio de GitHub

[Repositorio Proyecto1DeepLearning](https://github.com/iancumes/Proyecto1DeepLearning)

El README documenta instalación y ejecución. `train.csv` no se publica: debe colocarse en la raíz y coincidir con la huella registrada.
"""))
    cells.append(nbformat.v4.new_markdown_cell("""## Takeaways y reproducibilidad

El comando `python scripts/run_pipeline.py --data train.csv --budget-minutes 75 --seed 23236` reconstruye los artefactos. Para el día de competencia, coloque `pipeline_test.csv` en la raíz y ejecute `python scripts/predict.py`; la salida oficial tendrá exactamente `Id,Prediction`. `expected_output.csv` es un template opcional y `--model ensemble` queda como alternativa autorizada.
"""))
    notebook.cells = cells
    path = notebook_dir / "Proyecto1_MLP_Ames_IanCumes_23236.ipynb"
    nbformat.write(notebook, path)
    return path


def execute_notebook(path: str | Path, root: str | Path, timeout: int = 300) -> None:
    path, root = Path(path), Path(root)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(root)}}, allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)
