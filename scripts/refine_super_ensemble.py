"""Compara meta-modelos OOF y calibra el super-ensemble sin usar etiquetas externas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def bins(y: pd.Series) -> pd.Series:
    return pd.qcut(y.rank(method="first"), 10, labels=False, duplicates="drop")


def constrained_weights(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    scale = max(float(np.std(y)), 1.0)
    objective = lambda weights: np.mean(((y - matrix @ weights) / scale) ** 2)
    count = matrix.shape[1]
    result = minimize(
        objective, np.full(count, 1.0 / count), method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def cv_predict(model_factory, x: pd.DataFrame, y: pd.Series, splitter) -> tuple[np.ndarray, list]:
    prediction = np.zeros(len(y), dtype=float)
    fitted = []
    for train_idx, val_idx in splitter.split(x, bins(y)):
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prediction[val_idx] = model.predict(x.iloc[val_idx])
        fitted.append(model)
    return prediction, fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default=str(ROOT / "predictions_refined_stack.csv"))
    args = parser.parse_args()

    oof = pd.read_csv(ROOT / "results" / "super_ensemble_oof.csv")
    base_test = pd.read_csv(ROOT / "results" / "super_ensemble_test_predictions.csv")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if oof["Id"].tolist() != train["Id"].tolist() or base_test["Id"].tolist() != test["Id"].tolist():
        raise ValueError("Los IDs de las matrices OOF/test no coinciden con los datasets.")

    y = oof["SalePrice"].astype(float)
    base_names = [column for column in oof.columns if column not in {"Id", "SalePrice", "super_ensemble"}]
    core_names = ["lgb_2", "lgb_1", "xgb_2", "catboost", "mlp"]
    numeric_names = [column for column in train.select_dtypes(include=np.number).columns if column not in {"Id", "SalePrice"}]
    key_numeric = [
        "OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
        "GarageCars", "GarageArea", "YearBuilt", "YearRemodAdd", "FullBath", "Fireplaces",
        "LotArea", "MasVnrArea", "BsmtFinSF1", "TotRmsAbvGrd",
    ]
    train_meta = oof[base_names + ["super_ensemble"]].copy()
    test_meta = base_test[base_names + ["super_ensemble"]].copy()
    for column in numeric_names:
        train_meta[f"raw_{column}"] = train[column]
        test_meta[f"raw_{column}"] = test[column]

    splitter = StratifiedKFold(n_splits=7, shuffle=True, random_state=23999)
    candidates = []

    def register(name: str, columns: list[str], factory) -> None:
        cv, _ = cv_predict(factory, train_meta[columns], y, splitter)
        final_model = factory()
        final_model.fit(train_meta[columns], y)
        test_prediction = final_model.predict(test_meta[columns])
        candidates.append({"name": name, "cv_rmse": rmse(y, cv), "cv_prediction": cv, "test_prediction": test_prediction, "columns": columns})
        print(f"{name}: meta-CV RMSE={rmse(y, cv):,.2f}", flush=True)

    register("calibracion_lineal", ["super_ensemble"], lambda: LinearRegression())
    register("nnls_core", core_names, lambda: LinearRegression(positive=True))
    register("nnls_all", base_names, lambda: LinearRegression(positive=True))
    for alpha in (10.0, 100.0, 1000.0, 10000.0):
        register(
            f"ridge_core_a{alpha:g}", core_names,
            lambda alpha=alpha: Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]),
        )
    for epsilon in (1.2, 1.5, 1.8):
        register(
            f"huber_core_e{epsilon}", core_names,
            lambda epsilon=epsilon: Pipeline([("scale", RobustScaler()), ("model", HuberRegressor(epsilon=epsilon, alpha=1e-4, max_iter=3000))]),
        )

    combined_columns = core_names + [f"raw_{column}" for column in key_numeric]
    numeric_transform = ColumnTransformer([
        ("values", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), combined_columns),
    ])
    for alpha in (100.0, 1000.0, 10000.0):
        register(
            f"ridge_features_a{alpha:g}", combined_columns,
            lambda alpha=alpha: Pipeline([("prep", numeric_transform), ("model", Ridge(alpha=alpha))]),
        )
    for depth in (1, 2):
        for loss in ("squared_error", "huber"):
            register(
                f"gbr_core_d{depth}_{loss}", core_names,
                lambda depth=depth, loss=loss: GradientBoostingRegressor(
                    loss=loss, n_estimators=180, learning_rate=0.025, max_depth=depth,
                    min_samples_leaf=15, subsample=0.8, random_state=23236,
                ),
            )
    register(
        "rf_meta_core", core_names,
        lambda: RandomForestRegressor(
            n_estimators=700, max_depth=4, min_samples_leaf=8, max_features=0.8,
            n_jobs=4, random_state=23236,
        ),
    )

    # Validación cruzada específica del blend convexo para medir estabilidad de pesos.
    core_oof = oof[core_names].to_numpy(float)
    core_test = base_test[core_names].to_numpy(float)
    convex_cv = np.zeros(len(y), dtype=float)
    fold_weights = []
    for train_idx, val_idx in splitter.split(core_oof, bins(y)):
        weights = constrained_weights(y.iloc[train_idx].to_numpy(), core_oof[train_idx])
        convex_cv[val_idx] = core_oof[val_idx] @ weights
        fold_weights.append(weights)
    stable_weights = np.mean(fold_weights, axis=0)
    stable_weights /= stable_weights.sum()
    candidates.append({
        "name": "convex_core_stable", "cv_rmse": rmse(y, convex_cv),
        "cv_prediction": convex_cv, "test_prediction": core_test @ stable_weights,
        "columns": core_names, "weights": dict(zip(core_names, stable_weights.tolist())),
    })
    print(f"convex_core_stable: meta-CV RMSE={rmse(y, convex_cv):,.2f}", flush=True)

    best = min(candidates, key=lambda item: item["cv_rmse"])
    output = pd.DataFrame({"Id": test["Id"], "Prediction": np.clip(best["test_prediction"], 0.0, None)})
    output_path = Path(args.output)
    output.to_csv(output_path, index=False, float_format="%.6f")
    summary = {
        "scores": sorted([{"name": item["name"], "cv_rmse": item["cv_rmse"], **({"weights": item["weights"]} if "weights" in item else {})} for item in candidates], key=lambda item: item["cv_rmse"]),
        "selected": best["name"], "selected_cv_rmse": best["cv_rmse"],
        "output": str(output_path.resolve()),
    }
    (ROOT / "results" / "refined_stack_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
