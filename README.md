# Proyecto 1 — MLP para Ames Housing

Implementación reproducible en PyTorch para predecir `SalePrice`. El proyecto usa exclusivamente `train.csv` entregado por el curso, mantiene un test interno aislado y guarda conjuntamente el preprocesamiento, la transformación del target y los pesos del MLP.

## Requisitos

- Python 3.11 recomendado.
- CPU con al menos 8 GB de RAM; no se necesita GPU.
- `train.csv` en la raíz del repositorio. El archivo no se versiona.

En Windows conviene crear el entorno en una ruta corta, fuera de OneDrive:

```powershell
py -3.11 -m venv C:\venvs\proyecto1dl
C:\venvs\proyecto1dl\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución completa

```powershell
python scripts/run_pipeline.py --data train.csv --budget-minutes 75 --seed 23236
```

El presupuesto comienza al cargar el CSV; la instalación no forma parte de los 75 minutos. Se puede continuar el estudio existente con `--resume`. Para verificar todo en pocos minutos:

```powershell
python scripts/run_pipeline.py --data train.csv --budget-minutes 8 --seed 23236 --smoke
python -m pytest
```

Para dedicar tiempo adicional a Optuna sin perder los trials anteriores:

```powershell
python scripts/run_pipeline.py --data train.csv --budget-minutes 20 --seed 23236 --resume --search-minutes 15 --trials 120
```

## Predicción de competencia

El formato real se reconoce con estos archivos:

- `pipeline_test.csv`: 80 features de entrada, incluida `Id`, sin `SalePrice`.
- `expected_output.csv`: ejemplo o template opcional con exactamente `Id,Prediction`.

El comando oficial solo necesita el archivo de evaluación:

```powershell
python scripts/predict.py
```

La forma explícita equivalente es:

```powershell
python scripts/predict.py --input pipeline_test.csv --output predictions.csv --model single
```

La salida contiene exactamente `Id,Prediction`, preserva el orden de entrada y usa seis decimales. Si el profesor también entrega un template con los mismos IDs, agregue `--template expected_output.csv` para validarlo. No use el template de ejemplo de cinco filas con una evaluación de distinto tamaño. Se crea `predictions.metadata.json` para auditoría, pero solo se entrega `predictions.csv`.

Ensemble opcional, únicamente si el profesor lo permite:

```powershell
python scripts/predict.py --input pipeline_test.csv --output predictions_ensemble.csv --model ensemble
```

## Estructura

```text
src/          datos, preprocesamiento, MLP, entrenamiento, búsqueda e inferencia
scripts/      pipeline completo, predicción y utilidades del notebook
tests/        pruebas unitarias e integración ligera
notebooks/    informe ejecutado con las seis secciones de la rúbrica
results/      métricas, historiales, tablas y figuras
artifacts/    preprocesador, transformación del target y pesos finales
```

Los transformadores se ajustan solo con el fold de entrenamiento. `Id` nunca entra al modelo. Las métricas se calculan en dólares después de invertir la transformación del objetivo.

## Reproducibilidad y límites

- Semilla principal: `23236`.
- Split: 80% desarrollo y 20% test interno, estratificado por deciles del target.
- Selección: ablaciones, Optuna con 3 folds y reevaluación de finalistas con 5 folds.
- Evaluación final: una única consulta al test interno después de bloquear la configuración.
- El RMSE interno no garantiza el resultado del held-out.
- No se utilizan precios externos del dataset Ames público.

El informe principal es [`notebooks/Proyecto1_MLP_Ames_IanCumes_23236.ipynb`](notebooks/Proyecto1_MLP_Ames_IanCumes_23236.ipynb). La guía ampliada de uso está fuera del repositorio, junto a la carpeta del proyecto.
