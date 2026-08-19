"""Reentrena con 100% de datos los componentes con peso del ensemble campeón."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_super_ensemble import prepare_catboost
from src.config import EXPECTED_FEATURES, ID_COLUMN, PreprocessConfig, SEED, TARGET
from src.data import load_data, target_bins
from src.preprocessing import build_preprocessor


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def lgb_model(params: dict, seed: int):
    clean = {key: value for key, value in params.items() if key != "target_mode"}
    return lgb.LGBMRegressor(
        objective="regression", random_state=seed, n_jobs=4, verbosity=-1,
        deterministic=True, force_col_wise=True, **clean,
    )


def inverse(values, mode: str) -> np.ndarray:
    return np.expm1(values) if mode == "log" else np.asarray(values, dtype=float)


def target(values, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log1p(values) if mode == "log" else values


def preprocessor_config() -> PreprocessConfig:
    return PreprocessConfig(
        scaler="standard", encoding="onehot", feature_engineering=True,
        add_log_features=True, winsorize=False, min_frequency=2, target_transform="standard",
    )


def full_lgb_predictions(x, y, x_test, params: dict, seeds: list[int], artifact_dir: Path, name: str) -> np.ndarray:
    prep = build_preprocessor(x, preprocessor_config())
    matrix = prep.fit_transform(x).astype(np.float32)
    test_matrix = prep.transform(x_test).astype(np.float32)
    joblib.dump(prep, artifact_dir / f"{name}_preprocessor.joblib", compress=3)
    predictions = []
    for seed in seeds:
        model = lgb_model(params, seed)
        model.fit(matrix, target(y, params["target_mode"]))
        predictions.append(inverse(model.predict(test_matrix), params["target_mode"]))
        joblib.dump(model, artifact_dir / f"{name}_seed_{seed}.joblib", compress=3)
    return np.mean(predictions, axis=0)


def nested_bagging_weight(x, y, params: dict, direct_oof: np.ndarray, seed: int) -> tuple[float, float, float]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 700)
    inner_bag = np.zeros(len(y), dtype=float)
    for outer_number, (outer_train, outer_val) in enumerate(outer.split(x, target_bins(y)), start=1):
        x_outer, y_outer = x.iloc[outer_train].reset_index(drop=True), y.iloc[outer_train].reset_index(drop=True)
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 9000 + outer_number)
        val_predictions = []
        for inner_number, (inner_train, _) in enumerate(inner.split(x_outer, target_bins(y_outer)), start=1):
            prep = build_preprocessor(x_outer.iloc[inner_train], preprocessor_config())
            train_matrix = prep.fit_transform(x_outer.iloc[inner_train]).astype(np.float32)
            val_matrix = prep.transform(x.iloc[outer_val]).astype(np.float32)
            model = lgb_model(params, seed + outer_number * 100 + inner_number)
            model.fit(train_matrix, target(y_outer.iloc[inner_train], params["target_mode"]))
            val_predictions.append(inverse(model.predict(val_matrix), params["target_mode"]))
        inner_bag[outer_val] = np.mean(val_predictions, axis=0)
    grid = np.linspace(0.0, 1.0, 101)
    scores = [(rmse(y, alpha * direct_oof + (1.0 - alpha) * inner_bag), float(alpha)) for alpha in grid]
    best_score, direct_weight = min(scores)
    return direct_weight, rmse(y, inner_bag), best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default=str(ROOT / "predictions_full_champion.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train = load_data(args.train, require_target=True).reset_index(drop=True)
    test = load_data(args.test, require_target=False).reset_index(drop=True)
    x, y, x_test = train[EXPECTED_FEATURES], train[TARGET].astype(float), test[EXPECTED_FEATURES]
    summary = json.loads((ROOT / "results" / "super_ensemble_results.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(ROOT / "results" / "super_ensemble_oof.csv")
    base_test = pd.read_csv(ROOT / "results" / "super_ensemble_test_predictions.csv")
    weights = summary["weights"]
    scores_by_name = {item["name"]: item for item in summary["model_scores"]}
    active_lgb = [name for name, weight in weights.items() if name.startswith("lgb_") and weight > 1e-4]
    artifact_dir = ROOT / "artifacts" / "full_champion"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + offset for offset in range(12)]
    replacements = {}
    nested_results = {}

    for name in active_lgb:
        params = scores_by_name[name]["params"]
        full_prediction = full_lgb_predictions(x, y, x_test, params, seeds, artifact_dir, name)
        direct_weight, inner_rmse, nested_rmse = nested_bagging_weight(
            x, y, params, oof[name].to_numpy(float), args.seed,
        )
        # La analogía nested compara 80% directo vs. 60% bagged; en producción,
        # 100% directo vs. 80% bagged. Se usa exactamente el peso validado.
        replacements[name] = direct_weight * full_prediction + (1.0 - direct_weight) * base_test[name].to_numpy(float)
        nested_results[name] = {
            "full_direct_weight": direct_weight,
            "fold_bag_weight": 1.0 - direct_weight,
            "direct_oof_rmse": rmse(y, oof[name]),
            "inner_bag_oof_rmse": inner_rmse,
            "nested_blend_rmse": nested_rmse,
        }
        print(f"{name}: peso full={direct_weight:.2f}, nested RMSE={nested_rmse:,.2f}", flush=True)

    mlp_full = pd.read_csv(ROOT / "predictions.csv")
    if mlp_full[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("predictions.csv no corresponde al test real.")
    replacements["mlp"] = mlp_full["Prediction"].to_numpy(float)

    native_test, _ = prepare_catboost(test)
    cat_predictions = []
    for path in sorted((ROOT / "artifacts" / "catboost").glob("model_seed_*.cbm")):
        model = CatBoostRegressor()
        model.load_model(path)
        cat_predictions.append(model.predict(native_test))
    if cat_predictions:
        replacements["catboost"] = np.mean(cat_predictions, axis=0)

    component_predictions = []
    component_weights = []
    for name, weight in weights.items():
        if weight <= 1e-10:
            continue
        component_predictions.append(replacements.get(name, base_test[name].to_numpy(float)))
        component_weights.append(float(weight))
    component_weights = np.asarray(component_weights, dtype=float)
    component_weights /= component_weights.sum()
    raw_prediction = np.column_stack(component_predictions) @ component_weights

    calibrator = LinearRegression().fit(oof[["super_ensemble"]], y)
    final_prediction = calibrator.predict(raw_prediction.reshape(-1, 1))
    output_path = Path(args.output)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "Prediction": np.clip(final_prediction, 0.0, None)}).to_csv(
        output_path, index=False, float_format="%.6f",
    )
    result = {
        "weights": weights,
        "nested": nested_results,
        "calibration": {"slope": float(calibrator.coef_[0]), "intercept": float(calibrator.intercept_)},
        "full_seeds": seeds,
        "output": str(output_path.resolve()),
    }
    (ROOT / "results" / "full_champion_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
