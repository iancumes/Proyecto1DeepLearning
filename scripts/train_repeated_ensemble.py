"""Bagging de folds repetidos para los mejores modelos del super-ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_super_ensemble import build_matrix_folds, oof_matrix_model, oof_mlp, optimize_weights
from src.config import EXPECTED_FEATURES, ID_COLUMN, SEED, TARGET
from src.data import load_data, target_bins


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def meta_calibration_cv(y: pd.Series, prediction: np.ndarray, seed: int) -> tuple[float, LinearRegression]:
    splitter = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
    calibrated = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in splitter.split(prediction.reshape(-1, 1), target_bins(y)):
        model = LinearRegression().fit(prediction[train_idx].reshape(-1, 1), y.iloc[train_idx])
        calibrated[val_idx] = model.predict(prediction[val_idx].reshape(-1, 1))
    final_model = LinearRegression().fit(prediction.reshape(-1, 1), y)
    return rmse(y, calibrated), final_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default=str(ROOT / "predictions_repeated_super_ensemble.csv"))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train = load_data(args.train, require_target=True).reset_index(drop=True)
    test = load_data(args.test, require_target=False).reset_index(drop=True)
    x, y = train[EXPECTED_FEATURES], train[TARGET].astype(float)
    x_test = test[EXPECTED_FEATURES]
    prior = json.loads((ROOT / "results" / "super_ensemble_results.json").read_text(encoding="utf-8"))
    prior_oof = pd.read_csv(ROOT / "results" / "super_ensemble_oof.csv")
    prior_test = pd.read_csv(ROOT / "results" / "super_ensemble_test_predictions.csv")

    lgb_configs = [item for item in prior["model_scores"] if item["name"].startswith("lgb_")][:3]
    xgb_configs = [item for item in prior["model_scores"] if item["name"].startswith("xgb_")][:1]
    selected = [("lgb", item) for item in lgb_configs] + [("xgb", item) for item in xgb_configs]
    artifact_dir = ROOT / "artifacts" / "repeated_ensemble"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof_candidates: dict[str, np.ndarray] = {}
    test_candidates: dict[str, np.ndarray] = {}
    scores = []

    fold_seeds = [args.seed + 700 + repeat * 1000 for repeat in range(args.repeats)]
    for kind, item in selected:
        name = f"repeated_{item['name']}"
        repeated_oof, repeated_test = [], []
        repeat_scores = []
        for repeat, fold_seed in enumerate(fold_seeds, start=1):
            folds = build_matrix_folds(x, y, x_test, folds=5, seed=fold_seed)
            oof, test_prediction, models = oof_matrix_model(
                kind, item["params"], folds, y, args.seed + repeat * 10000 + len(repeated_oof) * 100,
            )
            repeated_oof.append(oof)
            repeated_test.append(test_prediction)
            repeat_scores.append(rmse(y, oof))
            joblib.dump(models, artifact_dir / f"{name}_repeat_{repeat}.joblib", compress=3)
            print(f"{name} repetición {repeat}: RMSE={repeat_scores[-1]:,.2f}", flush=True)
        averaged_oof = np.mean(repeated_oof, axis=0)
        averaged_test = np.mean(repeated_test, axis=0)
        oof_candidates[name] = averaged_oof
        test_candidates[name] = averaged_test
        scores.append({"name": name, "rmse": rmse(y, averaged_oof), "repeat_rmse": repeat_scores, "params": item["params"]})
        print(f"{name} promedio: RMSE={rmse(y, averaged_oof):,.2f}", flush=True)

    mlp_oofs, mlp_tests, mlp_scores = [], [], []
    for repeat, fold_seed in enumerate(fold_seeds, start=1):
        oof, test_prediction = oof_mlp(x, y, x_test, fold_seed)
        mlp_oofs.append(oof)
        mlp_tests.append(test_prediction)
        mlp_scores.append(rmse(y, oof))
        print(f"MLP repetición {repeat}: RMSE={mlp_scores[-1]:,.2f}", flush=True)
    oof_candidates["repeated_mlp"] = np.mean(mlp_oofs, axis=0)
    test_candidates["repeated_mlp"] = np.mean(mlp_tests, axis=0)
    scores.append({"name": "repeated_mlp", "rmse": rmse(y, oof_candidates["repeated_mlp"]), "repeat_rmse": mlp_scores})

    # CatBoost aporta diversidad; se reutiliza la versión OOF ya validada.
    oof_candidates["catboost"] = prior_oof["catboost"].to_numpy(float)
    test_candidates["catboost"] = prior_test["catboost"].to_numpy(float)
    scores.append({"name": "catboost", "rmse": rmse(y, oof_candidates["catboost"])})

    names = list(oof_candidates)
    matrix = np.column_stack([oof_candidates[name] for name in names])
    test_matrix = np.column_stack([test_candidates[name] for name in names])
    weights = optimize_weights(y.to_numpy(), matrix)
    blend_oof = matrix @ weights
    blend_test = test_matrix @ weights
    calibration_cv_rmse, calibrator = meta_calibration_cv(y, blend_oof, args.seed + 999)
    uncalibrated_rmse = rmse(y, blend_oof)
    if calibration_cv_rmse < uncalibrated_rmse:
        final_test = calibrator.predict(blend_test.reshape(-1, 1))
        calibration = {"used": True, "cv_rmse": calibration_cv_rmse, "intercept": float(calibrator.intercept_), "slope": float(calibrator.coef_[0])}
    else:
        final_test = blend_test
        calibration = {"used": False, "cv_rmse": calibration_cv_rmse}

    output_path = Path(args.output)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "Prediction": np.clip(final_test, 0.0, None)}).to_csv(
        output_path, index=False, float_format="%.6f",
    )
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, **oof_candidates, "blend": blend_oof}).to_csv(
        ROOT / "results" / "repeated_ensemble_oof.csv", index=False,
    )
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], **test_candidates, "blend": blend_test}).to_csv(
        ROOT / "results" / "repeated_ensemble_test_predictions.csv", index=False,
    )
    summary = {
        "fold_seeds": fold_seeds,
        "scores": sorted(scores, key=lambda item: item["rmse"]),
        "weights": dict(zip(names, weights.tolist())),
        "blend_oof_rmse": uncalibrated_rmse,
        "calibration": calibration,
        "output": str(output_path.resolve()),
    }
    (ROOT / "results" / "repeated_ensemble_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
