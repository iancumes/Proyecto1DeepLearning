"""Optimiza especialistas LightGBM con transformaciones potencia del target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_super_ensemble import build_matrix_folds
from src.config import EXPECTED_FEATURES, ID_COLUMN, SEED, TARGET
from src.data import load_data

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def forward_target(values, power: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log1p(values) if power == 0.0 else np.power(values, power)


def inverse_target(values, power: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if power == 0.0:
        return np.expm1(values)
    return np.power(np.clip(values, 0.0, None), 1.0 / power)


def make_model(params: dict, seed: int):
    clean = {key: value for key, value in params.items() if key != "target_power"}
    return lgb.LGBMRegressor(
        random_state=seed, n_jobs=4, verbosity=-1, deterministic=True,
        force_col_wise=True, **clean,
    )


def evaluate(params: dict, folds, y: pd.Series, seed: int, trial=None) -> float:
    scores = []
    power = params["target_power"]
    for number, fold in enumerate(folds, start=1):
        model = make_model(params, seed + number)
        model.fit(fold.x_train, forward_target(y.iloc[fold.train_idx], power))
        prediction = inverse_target(model.predict(fold.x_val), power)
        scores.append(rmse(y.iloc[fold.val_idx], prediction))
        if trial is not None:
            trial.report(float(np.mean(scores)), number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def predict_oof(params: dict, folds, y: pd.Series, seed: int):
    oof = np.zeros(len(y), dtype=float)
    test_predictions, models = [], []
    power = params["target_power"]
    for number, fold in enumerate(folds, start=1):
        model = make_model(params, seed + number * 101)
        model.fit(fold.x_train, forward_target(y.iloc[fold.train_idx], power))
        oof[fold.val_idx] = inverse_target(model.predict(fold.x_val), power)
        test_predictions.append(inverse_target(model.predict(fold.x_test), power))
        models.append(model)
    return oof, np.mean(test_predictions, axis=0), models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--test", required=True)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train = load_data(args.train, require_target=True).reset_index(drop=True)
    test = load_data(args.test, require_target=False).reset_index(drop=True)
    x, y, x_test = train[EXPECTED_FEATURES], train[TARGET].astype(float), test[EXPECTED_FEATURES]
    tuning_folds = build_matrix_folds(x, y, x_test, folds=4, seed=args.seed + 321)
    storage_path = ROOT / "results" / "power_lgb_optuna.db"
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = "ames_power_lgb_v1"
    if not args.resume:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 2, 6)
        params = {
            "target_power": trial.suggest_categorical("target_power", [0.0, 0.20, 0.35, 0.50, 0.70, 1.0]),
            "objective": trial.suggest_categorical("objective", ["regression", "huber", "fair"]),
            "n_estimators": trial.suggest_int("n_estimators", 500, 2200),
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.075, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, min(48, 2 ** max_depth)),
            "max_depth": max_depth,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 35),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 30.0, log=True),
        }
        trial.set_user_attr("model_params", params)
        return evaluate(params, tuning_folds, y, args.seed + trial.number * 37, trial)

    def callback(study, trial):
        complete = len([item for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE])
        if complete and complete % 10 == 0:
            print(f"Power-LGB: {complete} completos, mejor RMSE={study.best_value:,.2f}", flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[callback], gc_after_trial=True)
    complete = sorted(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
        key=lambda trial: trial.value,
    )
    final_folds = build_matrix_folds(x, y, x_test, folds=5, seed=args.seed + 700)
    artifact_dir = ROOT / "artifacts" / "power_lgb"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof_data, test_data, results = {}, {}, []
    for rank, trial in enumerate(complete[:args.finalists], start=1):
        params = trial.user_attrs["model_params"]
        oof, prediction, models = predict_oof(params, final_folds, y, args.seed + rank * 1000)
        name = f"power_lgb_{rank}"
        oof_data[name], test_data[name] = oof, prediction
        score = rmse(y, oof)
        results.append({"name": name, "rmse": score, "tuning_rmse": trial.value, "params": params})
        joblib.dump(models, artifact_dir / f"{name}_models.joblib", compress=3)
        print(f"{name}: power={params['target_power']} loss={params['objective']} RMSE 5-fold={score:,.2f}", flush=True)

    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, **oof_data}).to_csv(
        ROOT / "results" / "power_lgb_oof.csv", index=False,
    )
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], **test_data}).to_csv(
        ROOT / "results" / "power_lgb_test_predictions.csv", index=False,
    )
    summary = {"best_tuning_rmse": study.best_value, "results": sorted(results, key=lambda item: item["rmse"])}
    (ROOT / "results" / "power_lgb_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
