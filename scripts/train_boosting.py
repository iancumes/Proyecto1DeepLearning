"""Compara CatBoost, entrena el mejor ensemble y genera un blend de competencia."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ID_COLUMN, SEED, TARGET
from src.data import development_test_split, load_data, target_bins


CONFIGS = [
    {"name": "cb_log_d5", "log_target": True, "depth": 5, "learning_rate": 0.040, "l2_leaf_reg": 5.0, "random_strength": 0.4, "bagging_temperature": 0.5},
    {"name": "cb_log_d6", "log_target": True, "depth": 6, "learning_rate": 0.030, "l2_leaf_reg": 7.0, "random_strength": 0.5, "bagging_temperature": 0.5},
    {"name": "cb_raw_d6", "log_target": False, "depth": 6, "learning_rate": 0.030, "l2_leaf_reg": 7.0, "random_strength": 0.5, "bagging_temperature": 0.5},
]


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = frame.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    x["TotalSF"] = x["TotalBsmtSF"].fillna(0) + x["1stFlrSF"].fillna(0) + x["2ndFlrSF"].fillna(0)
    x["TotalBathrooms"] = x["FullBath"].fillna(0) + 0.5 * x["HalfBath"].fillna(0) + x["BsmtFullBath"].fillna(0) + 0.5 * x["BsmtHalfBath"].fillna(0)
    x["TotalPorchSF"] = x[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].fillna(0).sum(axis=1)
    x["HouseAgeAtSale"] = (x["YrSold"] - x["YearBuilt"]).clip(lower=0)
    x["YearsSinceRemodel"] = (x["YrSold"] - x["YearRemodAdd"]).clip(lower=0)
    x["GarageAgeAtSale"] = (x["YrSold"] - x["GarageYrBlt"]).clip(lower=0)
    for nominal_numeric in ("MSSubClass", "MoSold"):
        x[nominal_numeric] = x[nominal_numeric].astype("Int64").astype("string")
    categorical = x.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in categorical:
        x[column] = x[column].fillna("__MISSING__").astype(str)
    return x, categorical


def inverse_target(values: np.ndarray, use_log: bool) -> np.ndarray:
    return np.expm1(values) if use_log else values


def model_for(config: dict, seed: int, iterations: int, use_best_model: bool) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=config["depth"],
        learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"],
        random_strength=config["random_strength"],
        bagging_temperature=config["bagging_temperature"],
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
        use_best_model=use_best_model,
    )


def cross_validate(x: pd.DataFrame, y: pd.Series, categorical: list[str], config: dict, folds: int, seed: int) -> tuple[dict, np.ndarray]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    scores, best_iterations = [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, target_bins(y)), start=1):
        target_train = np.log1p(y.iloc[train_idx]) if config["log_target"] else y.iloc[train_idx]
        target_val = np.log1p(y.iloc[val_idx]) if config["log_target"] else y.iloc[val_idx]
        model = model_for(config, seed + fold, iterations=800, use_best_model=True)
        model.fit(
            x.iloc[train_idx], target_train,
            cat_features=categorical,
            eval_set=(x.iloc[val_idx], target_val),
            early_stopping_rounds=70,
        )
        prediction = inverse_target(model.predict(x.iloc[val_idx]), config["log_target"])
        oof[val_idx] = prediction
        scores.append(float(mean_squared_error(y.iloc[val_idx], prediction) ** 0.5))
        best_iterations.append(max(1, int(model.get_best_iteration()) + 1))
    return {
        "name": config["name"], "folds": folds,
        "mean_rmse": float(np.mean(scores)), "std_rmse": float(np.std(scores)),
        "fold_rmse": scores, "median_iterations": int(np.median(best_iterations)),
        "config": config,
    }, oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--test", required=True)
    parser.add_argument("--mlp-predictions", default=str(ROOT / "predictions.csv"))
    parser.add_argument("--output", default=str(ROOT / "predictions_best_blend.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train = load_data(args.train, require_target=True)
    test = load_data(args.test, require_target=False)
    x, categorical = prepare_features(train)
    x_test, test_categorical = prepare_features(test)
    if categorical != test_categorical or x.columns.tolist() != x_test.columns.tolist():
        raise ValueError("Las features preparadas de train y test no coinciden.")
    y = train[TARGET].astype(float).reset_index(drop=True)
    x = x.reset_index(drop=True)

    coarse_results = []
    for config in CONFIGS:
        result, _ = cross_validate(x, y, categorical, config, folds=3, seed=args.seed)
        coarse_results.append(result)
        print(f"3-fold {result['name']}: RMSE={result['mean_rmse']:,.2f} ± {result['std_rmse']:,.2f}")
    finalists = sorted(coarse_results, key=lambda row: row["mean_rmse"])[:1]

    final_results = []
    final_oof = {}
    for finalist in finalists:
        config = finalist["config"]
        result, oof = cross_validate(x, y, categorical, config, folds=5, seed=args.seed + 101)
        final_results.append(result)
        final_oof[result["name"]] = oof
        print(f"5-fold {result['name']}: RMSE={result['mean_rmse']:,.2f} ± {result['std_rmse']:,.2f}")
    winner = min(final_results, key=lambda row: row["mean_rmse"])
    winner_config = winner["config"]
    winner_oof = final_oof[winner["name"]]
    calibrator = LinearRegression().fit(winner_oof.reshape(-1, 1), y)
    calibrated_oof = calibrator.predict(winner_oof.reshape(-1, 1))
    calibrated_rmse = float(mean_squared_error(y, calibrated_oof) ** 0.5)

    # Evaluación comparable en el mismo test interno usado por el MLP.
    split = development_test_split(train, args.seed)
    x_dev, cat_dev = prepare_features(split.X_dev)
    x_internal, _ = prepare_features(split.X_test)
    internal_model = model_for(winner_config, args.seed, winner["median_iterations"], use_best_model=False)
    internal_y = np.log1p(split.y_dev) if winner_config["log_target"] else split.y_dev
    internal_model.fit(x_dev, internal_y, cat_features=cat_dev)
    internal_cat = inverse_target(internal_model.predict(x_internal), winner_config["log_target"])
    internal_cat = calibrator.predict(internal_cat.reshape(-1, 1))
    internal_saved = pd.read_csv(ROOT / "results" / "internal_test_predictions.csv")
    internal_frame = pd.DataFrame({ID_COLUMN: split.X_test[ID_COLUMN].to_numpy(), "CatBoost": internal_cat})
    internal_frame = internal_frame.merge(internal_saved[[ID_COLUMN, TARGET, "PredictionEnsemble"]], on=ID_COLUMN, validate="one_to_one")
    weights = np.linspace(0.0, 1.0, 21)
    blend_scores = []
    for cat_weight in weights:
        blended = cat_weight * internal_frame["CatBoost"] + (1.0 - cat_weight) * internal_frame["PredictionEnsemble"]
        blend_scores.append((float(mean_squared_error(internal_frame[TARGET], blended) ** 0.5), float(cat_weight)))
    blend_rmse, cat_weight = min(blend_scores)
    cat_internal_rmse = float(mean_squared_error(internal_frame[TARGET], internal_frame["CatBoost"]) ** 0.5)

    artifact_dir = ROOT / "artifacts" / "catboost"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed, args.seed + 1, args.seed + 2, args.seed + 3, args.seed + 4]
    test_predictions = []
    for seed in seeds:
        model = model_for(winner_config, seed, winner["median_iterations"], use_best_model=False)
        full_target = np.log1p(y) if winner_config["log_target"] else y
        model.fit(x, full_target, cat_features=categorical)
        model.save_model(artifact_dir / f"model_seed_{seed}.cbm")
        test_predictions.append(inverse_target(model.predict(x_test), winner_config["log_target"]))
    cat_prediction = np.mean(test_predictions, axis=0)
    cat_prediction = calibrator.predict(cat_prediction.reshape(-1, 1))
    cat_prediction = np.clip(cat_prediction, 0.0, None)
    cat_output = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), "Prediction": cat_prediction})
    cat_output.to_csv(ROOT / "predictions_catboost.csv", index=False, float_format="%.6f")

    mlp_output = pd.read_csv(args.mlp_predictions)
    if mlp_output.columns.tolist() != [ID_COLUMN, "Prediction"] or mlp_output[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Las predicciones MLP no corresponden al test indicado.")
    blend_prediction = cat_weight * cat_prediction + (1.0 - cat_weight) * mlp_output["Prediction"].to_numpy(float)
    blend_output = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), "Prediction": np.clip(blend_prediction, 0.0, None)})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blend_output.to_csv(output_path, index=False, float_format="%.6f")
    joblib.dump(calibrator, artifact_dir / "calibrator.joblib")

    summary = {
        "coarse_results": coarse_results,
        "final_results": final_results,
        "winner": winner,
        "oof_calibrated_rmse": calibrated_rmse,
        "internal_catboost_rmse": cat_internal_rmse,
        "internal_mlp_ensemble_rmse": float(mean_squared_error(internal_frame[TARGET], internal_frame["PredictionEnsemble"]) ** 0.5),
        "internal_blend_rmse": blend_rmse,
        "catboost_weight": cat_weight,
        "mlp_weight": 1.0 - cat_weight,
        "seeds": seeds,
        "test_rows": len(test),
        "output": str(output_path.resolve()),
    }
    (ROOT / "results" / "boosting_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["winner", "oof_calibrated_rmse", "internal_catboost_rmse", "internal_mlp_ensemble_rmse", "internal_blend_rmse", "catboost_weight", "mlp_weight", "output"]}, indent=2))


if __name__ == "__main__":
    main()
