"""Búsqueda y stacking OOF de modelos tabulares para el held-out de Ames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize, nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import read_json
from src.config import EXPECTED_FEATURES, ID_COLUMN, MLPConfig, PreprocessConfig, SEED, TARGET
from src.data import load_data, target_bins
from src.model import RegressionMLP
from src.preprocessing import TargetTransformer, build_preprocessor
from src.training import fit_fixed_epochs, predict_scaled

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class MatrixFold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def transform_target(y: pd.Series | np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    return np.log1p(values) if mode == "log" else values


def inverse_target(values: np.ndarray, mode: str) -> np.ndarray:
    return np.expm1(values) if mode == "log" else np.asarray(values, dtype=float)


def build_matrix_folds(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, folds: int, seed: int) -> list[MatrixFold]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cache = []
    config = PreprocessConfig(
        scaler="standard", encoding="onehot", feature_engineering=True,
        add_log_features=True, winsorize=False, min_frequency=2, target_transform="standard",
    )
    for train_idx, val_idx in splitter.split(x, target_bins(y)):
        preprocessor = build_preprocessor(x.iloc[train_idx], config)
        train_matrix = preprocessor.fit_transform(x.iloc[train_idx]).astype(np.float32)
        cache.append(MatrixFold(
            train_idx=train_idx,
            val_idx=val_idx,
            x_train=train_matrix,
            x_val=preprocessor.transform(x.iloc[val_idx]).astype(np.float32),
            x_test=preprocessor.transform(x_test).astype(np.float32),
        ))
    return cache


def estimator(kind: str, params: dict, seed: int):
    clean = {key: value for key, value in params.items() if key != "target_mode"}
    if kind == "xgb":
        return XGBRegressor(
            objective="reg:squarederror", tree_method="hist", random_state=seed,
            n_jobs=4, verbosity=0, **clean,
        )
    if kind == "lgb":
        return lgb.LGBMRegressor(
            objective="regression", random_state=seed, n_jobs=4, verbosity=-1,
            deterministic=True, force_col_wise=True, **clean,
        )
    if kind == "extra":
        return ExtraTreesRegressor(random_state=seed, n_jobs=4, **clean)
    if kind == "ridge":
        return Ridge(**clean)
    raise ValueError(f"Modelo desconocido: {kind}")


def suggest(trial: optuna.Trial, kind: str) -> dict:
    target_mode = trial.suggest_categorical("target_mode", ["raw", "log"])
    if kind == "xgb":
        return {
            "target_mode": target_mode,
            "n_estimators": trial.suggest_int("n_estimators", 350, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 80.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }
    if kind == "lgb":
        max_depth = trial.suggest_int("max_depth", 3, 8)
        return {
            "target_mode": target_mode,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.07, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, min(64, 2 ** max_depth)),
            "max_depth": max_depth,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 45),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 80.0, log=True),
        }
    if kind == "extra":
        depth_choice = trial.suggest_categorical("max_depth_choice", [0, 12, 18, 25, 35])
        return {
            "target_mode": target_mode,
            "n_estimators": 700,
            "max_depth": None if depth_choice == 0 else depth_choice,
            "max_features": trial.suggest_float("max_features", 0.35, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
        }
    raise ValueError(kind)


def score_config(kind: str, params: dict, folds: list[MatrixFold], y: pd.Series, seed: int, trial: optuna.Trial | None = None) -> float:
    scores = []
    mode = params["target_mode"]
    for fold_number, fold in enumerate(folds, start=1):
        model = estimator(kind, params, seed + fold_number)
        model.fit(fold.x_train, transform_target(y.iloc[fold.train_idx], mode))
        prediction = inverse_target(model.predict(fold.x_val), mode)
        scores.append(rmse(y.iloc[fold.val_idx], prediction))
        if trial is not None:
            trial.report(float(np.mean(scores)), fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def tune(kind: str, folds: list[MatrixFold], y: pd.Series, trials: int, seed: int, storage: Path, resume: bool) -> optuna.Study:
    name = f"ames_super_{kind}_v1"
    storage_url = f"sqlite:///{storage.as_posix()}"
    if not resume:
        try:
            optuna.delete_study(study_name=name, storage=storage_url)
        except KeyError:
            pass
    study = optuna.create_study(
        study_name=name, storage=storage_url, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, kind)
        trial.set_user_attr("model_params", params)
        return score_config(kind, params, folds, y, seed + trial.number * 17, trial)

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = len([item for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE])
        if completed and completed % 5 == 0:
            print(f"{kind}: {completed} trials completos, mejor RMSE={study.best_value:,.2f}", flush=True)

    if trials > 0:
        study.optimize(objective, n_trials=trials, callbacks=[progress], gc_after_trial=True)
    return study


def top_configs(study: optuna.Study, count: int) -> list[dict]:
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    selected = []
    for trial in sorted(complete, key=lambda item: item.value):
        params = trial.user_attrs["model_params"]
        signature = json.dumps(params, sort_keys=True)
        if signature not in {json.dumps(item, sort_keys=True) for item in selected}:
            selected.append(params)
        if len(selected) == count:
            break
    return selected


def config_signature(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def oof_matrix_model(kind: str, params: dict, folds: list[MatrixFold], y: pd.Series, seed: int) -> tuple[np.ndarray, np.ndarray, list]:
    oof = np.zeros(len(y), dtype=float)
    test_predictions, models = [], []
    mode = params["target_mode"]
    for fold_number, fold in enumerate(folds, start=1):
        model = estimator(kind, params, seed + fold_number * 101)
        model.fit(fold.x_train, transform_target(y.iloc[fold.train_idx], mode))
        oof[fold.val_idx] = inverse_target(model.predict(fold.x_val), mode)
        test_predictions.append(inverse_target(model.predict(fold.x_test), mode))
        models.append(model)
    return oof, np.mean(test_predictions, axis=0), models


def predictions_from_matrix_models(params: dict, folds: list[MatrixFold], y: pd.Series, models: list) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    test_predictions = []
    mode = params["target_mode"]
    if len(models) != len(folds):
        raise ValueError("La cantidad de modelos guardados no coincide con los folds.")
    for fold, model in zip(folds, models):
        oof[fold.val_idx] = inverse_target(model.predict(fold.x_val), mode)
        test_predictions.append(inverse_target(model.predict(fold.x_test), mode))
    return oof, np.mean(test_predictions, axis=0)


def prepare_catboost(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = frame.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    x["TotalSF"] = x["TotalBsmtSF"].fillna(0) + x["1stFlrSF"].fillna(0) + x["2ndFlrSF"].fillna(0)
    x["TotalBathrooms"] = x["FullBath"].fillna(0) + 0.5 * x["HalfBath"].fillna(0) + x["BsmtFullBath"].fillna(0) + 0.5 * x["BsmtHalfBath"].fillna(0)
    x["TotalPorchSF"] = x[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].fillna(0).sum(axis=1)
    x["HouseAgeAtSale"] = (x["YrSold"] - x["YearBuilt"]).clip(lower=0)
    x["YearsSinceRemodel"] = (x["YrSold"] - x["YearRemodAdd"]).clip(lower=0)
    x["GarageAgeAtSale"] = (x["YrSold"] - x["GarageYrBlt"]).clip(lower=0)
    for column in ("MSSubClass", "MoSold"):
        x[column] = x[column].astype("Int64").astype("string")
    categorical = x.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in categorical:
        x[column] = x[column].fillna("__MISSING__").astype(str)
    return x, categorical


def oof_catboost(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, seed: int, folds: int = 5) -> tuple[np.ndarray, np.ndarray, list]:
    native_x, categorical = prepare_catboost(x)
    native_test, _ = prepare_catboost(x_test)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_predictions, models = [], []
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(native_x, target_bins(y)), start=1):
        model = CatBoostRegressor(
            iterations=900, depth=6, learning_rate=0.03, l2_leaf_reg=7.0,
            random_strength=0.5, bagging_temperature=0.5, loss_function="RMSE",
            random_seed=seed + fold_number, thread_count=-1, verbose=False, allow_writing_files=False,
        )
        model.fit(native_x.iloc[train_idx], y.iloc[train_idx], cat_features=categorical)
        oof[val_idx] = model.predict(native_x.iloc[val_idx])
        test_predictions.append(model.predict(native_test))
        models.append(model)
    return oof, np.mean(test_predictions, axis=0), models


def predictions_from_catboost_models(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, seed: int, models: list) -> tuple[np.ndarray, np.ndarray]:
    native_x, _ = prepare_catboost(x)
    native_test, _ = prepare_catboost(x_test)
    splitter = StratifiedKFold(n_splits=len(models), shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_predictions = []
    for (_, val_idx), model in zip(splitter.split(native_x, target_bins(y)), models):
        oof[val_idx] = model.predict(native_x.iloc[val_idx])
        test_predictions.append(model.predict(native_test))
    return oof, np.mean(test_predictions, axis=0)


def oof_mlp(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    metadata = read_json(ROOT / "artifacts" / "metadata.json")
    preprocess_config = PreprocessConfig(**metadata["preprocess_config"])
    model_config = MLPConfig.from_dict(metadata["model_config"])
    epochs = int(metadata["final_epochs"])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_predictions = []
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(x, target_bins(y)), start=1):
        preprocessor = build_preprocessor(x.iloc[train_idx], preprocess_config)
        train_matrix = preprocessor.fit_transform(x.iloc[train_idx]).astype(np.float32)
        val_matrix = preprocessor.transform(x.iloc[val_idx]).astype(np.float32)
        test_matrix = preprocessor.transform(x_test).astype(np.float32)
        target = TargetTransformer(preprocess_config.target_transform).fit(y.iloc[train_idx])
        val_predictions, fold_test = [], []
        for offset in (0, 1):
            model, _ = fit_fixed_epochs(
                train_matrix, target.transform(y.iloc[train_idx]), model_config,
                seed + fold_number * 101 + offset, epochs,
            )
            val_predictions.append(target.inverse_transform(predict_scaled(model, val_matrix)))
            fold_test.append(target.inverse_transform(predict_scaled(model, test_matrix)))
        oof[val_idx] = np.mean(val_predictions, axis=0)
        test_predictions.append(np.mean(fold_test, axis=0))
    return oof, np.mean(test_predictions, axis=0)


def optimize_weights(y: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
    count = prediction_matrix.shape[1]
    scale = max(float(np.std(y)), 1.0)
    objective = lambda weights: np.mean(((y - prediction_matrix @ weights) / scale) ** 2)
    best = None
    starts = [np.full(count, 1.0 / count)] + [np.eye(count)[index] for index in range(count)]
    for start in starts:
        result = minimize(
            objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        fallback, _ = nnls(prediction_matrix / scale, y / scale)
        if fallback.sum() <= 0:
            raise RuntimeError("No se pudieron optimizar los pesos del ensemble.")
        return fallback / fallback.sum()
    weights = np.clip(best.x, 0.0, 1.0)
    return weights / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default=str(ROOT / "predictions_super_ensemble.csv"))
    parser.add_argument("--xgb-trials", type=int, default=45)
    parser.add_argument("--lgb-trials", type=int, default=45)
    parser.add_argument("--extra-trials", type=int, default=18)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-models", action="store_true", help="Reutiliza finalistas ya entrenados")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--xgb-finalists", type=int, default=3)
    parser.add_argument("--lgb-finalists", type=int, default=5)
    parser.add_argument("--extra-finalists", type=int, default=2)
    args = parser.parse_args()

    train = load_data(args.train, require_target=True).reset_index(drop=True)
    test = load_data(args.test, require_target=False).reset_index(drop=True)
    x = train[EXPECTED_FEATURES]
    y = train[TARGET].astype(float)
    x_test = test[EXPECTED_FEATURES]
    results_dir = ROOT / "results"
    artifact_dir = ROOT / "artifacts" / "super_ensemble"
    results_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Preparando folds sin leakage...", flush=True)
    tuning_folds = build_matrix_folds(x, y, x_test, folds=3, seed=args.seed)
    studies = {}
    storage = results_dir / "super_ensemble_optuna.db"
    for kind, trials in (("xgb", args.xgb_trials), ("lgb", args.lgb_trials), ("extra", args.extra_trials)):
        studies[kind] = tune(kind, tuning_folds, y, trials, args.seed, storage, args.resume)
        print(f"Ganador {kind}: RMSE 3-fold={studies[kind].best_value:,.2f}", flush=True)
    if args.tune_only:
        return

    print("Reevaluando finalistas sobre los mismos 5 folds...", flush=True)
    final_folds = build_matrix_folds(x, y, x_test, folds=5, seed=args.seed + 700)
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    model_scores = []
    for kind, count in (("xgb", args.xgb_finalists), ("lgb", args.lgb_finalists), ("extra", args.extra_finalists)):
        for rank, params in enumerate(top_configs(studies[kind], count), start=1):
            name = f"{kind}_{rank}"
            model_path = artifact_dir / f"{name}_{config_signature(params)}_fold_models.joblib"
            if args.reuse_models and model_path.exists():
                models = joblib.load(model_path)
                oof, test_pred = predictions_from_matrix_models(params, final_folds, y, models)
            else:
                oof, test_pred, models = oof_matrix_model(kind, params, final_folds, y, args.seed + rank * 1000)
                joblib.dump(models, model_path, compress=3)
            oof_predictions[name] = oof
            test_predictions[name] = test_pred
            score = rmse(y, oof)
            model_scores.append({"name": name, "rmse": score, "params": params})
            print(f"{name}: RMSE OOF={score:,.2f}", flush=True)

    # Controles lineales con target crudo y logarítmico.
    for mode, alpha in (("raw", 30.0), ("log", 10.0)):
        name = f"ridge_{mode}"
        params = {"target_mode": mode, "alpha": alpha}
        model_path = artifact_dir / f"{name}_fold_models.joblib"
        if args.reuse_models and model_path.exists():
            models = joblib.load(model_path)
            oof, test_pred = predictions_from_matrix_models(params, final_folds, y, models)
        else:
            oof, test_pred, models = oof_matrix_model("ridge", params, final_folds, y, args.seed + 5000)
            joblib.dump(models, model_path, compress=3)
        oof_predictions[name] = oof
        test_predictions[name] = test_pred
        model_scores.append({"name": name, "rmse": rmse(y, oof), "params": params})
        print(f"{name}: RMSE OOF={rmse(y, oof):,.2f}", flush=True)

    print("Generando CatBoost OOF...", flush=True)
    cat_paths = [artifact_dir / f"catboost_fold_{index}.cbm" for index in range(1, 6)]
    if args.reuse_models and all(path.exists() for path in cat_paths):
        cat_models = []
        for path in cat_paths:
            model = CatBoostRegressor()
            model.load_model(path)
            cat_models.append(model)
        cat_oof, cat_test = predictions_from_catboost_models(x, y, x_test, args.seed + 700, cat_models)
    else:
        cat_oof, cat_test, cat_models = oof_catboost(x, y, x_test, args.seed + 700)
        for index, model in enumerate(cat_models, start=1):
            model.save_model(artifact_dir / f"catboost_fold_{index}.cbm")
    oof_predictions["catboost"] = cat_oof
    test_predictions["catboost"] = cat_test
    model_scores.append({"name": "catboost", "rmse": rmse(y, cat_oof), "params": {"iterations": 900, "depth": 6, "target_mode": "raw"}})
    print(f"catboost: RMSE OOF={rmse(y, cat_oof):,.2f}", flush=True)

    print("Generando MLP OOF...", flush=True)
    mlp_oof, mlp_test = oof_mlp(x, y, x_test, args.seed + 700)
    oof_predictions["mlp"] = mlp_oof
    test_predictions["mlp"] = mlp_test
    model_scores.append({"name": "mlp", "rmse": rmse(y, mlp_oof), "params": {"source": "T109", "fold_seeds": 2}})
    print(f"mlp: RMSE OOF={rmse(y, mlp_oof):,.2f}", flush=True)

    names = list(oof_predictions)
    oof_matrix = np.column_stack([oof_predictions[name] for name in names])
    test_matrix = np.column_stack([test_predictions[name] for name in names])
    weights = optimize_weights(y.to_numpy(), oof_matrix)
    blend_oof = oof_matrix @ weights
    blend_test = np.clip(test_matrix @ weights, 0.0, None)
    blend_score = rmse(y, blend_oof)

    output = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), "Prediction": blend_test})
    output_path = Path(args.output)
    output.to_csv(output_path, index=False, float_format="%.6f")
    oof_frame = pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, **oof_predictions, "super_ensemble": blend_oof})
    oof_frame.to_csv(results_dir / "super_ensemble_oof.csv", index=False)
    test_frame = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], **test_predictions, "super_ensemble": blend_test})
    test_frame.to_csv(results_dir / "super_ensemble_test_predictions.csv", index=False)
    weights_dict = {name: float(weight) for name, weight in zip(names, weights)}
    summary = {
        "seed": args.seed,
        "rows_train": len(train), "rows_test": len(test),
        "model_scores": sorted(model_scores, key=lambda item: item["rmse"]),
        "weights": weights_dict,
        "super_ensemble_oof_rmse": blend_score,
        "best_base_oof_rmse": min(item["rmse"] for item in model_scores),
        "output": str(output_path.resolve()),
        "optuna": {kind: {"trials": len(study.trials), "best_value": study.best_value, "best_params": study.best_trial.user_attrs["model_params"]} for kind, study in studies.items()},
    }
    (results_dir / "super_ensemble_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"model_scores": summary["model_scores"], "weights": weights_dict, "super_ensemble_oof_rmse": blend_score, "output": summary["output"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
