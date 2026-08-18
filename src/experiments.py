"""Ablaciones, baselines, validacion cruzada y busqueda Optuna."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import MLPConfig, PreprocessConfig, SEED
from .data import target_bins
from .preprocessing import TargetTransformer, build_preprocessor
from .training import regression_metrics, train_mlp


@dataclass(frozen=True)
class Candidate:
    name: str
    preprocess: PreprocessConfig
    model: MLPConfig

    def to_dict(self) -> dict:
        return {"name": self.name, "preprocess": self.preprocess.to_dict(), "model": self.model.to_dict()}

    @classmethod
    def from_dict(cls, values: dict) -> "Candidate":
        return cls(
            values["name"], PreprocessConfig(**values["preprocess"]), MLPConfig.from_dict(values["model"])
        )


def ablation_catalog(smoke: bool = False) -> list[Candidate]:
    epochs, patience = (8, 3) if smoke else (140, 18)
    base_model = dict(max_epochs=epochs, patience=patience)
    return [
        Candidate("A01_numerico", PreprocessConfig(numeric_only=True, feature_engineering=False, add_log_features=False), MLPConfig(hidden_layers=(128, 64), batch_norm=False, dropout=0.0, **base_model)),
        Candidate("A02_onehot_base", PreprocessConfig(feature_engineering=False, add_log_features=False), MLPConfig(hidden_layers=(128, 64), batch_norm=False, dropout=0.0, **base_model)),
        Candidate("A03_target_log", PreprocessConfig(feature_engineering=False, add_log_features=False, target_transform="log1p"), MLPConfig(hidden_layers=(128, 64), **base_model)),
        Candidate("A04_robust", PreprocessConfig(scaler="robust", feature_engineering=False, add_log_features=False), MLPConfig(hidden_layers=(128, 64), **base_model)),
        Candidate("A05_profundo", PreprocessConfig(), MLPConfig(hidden_layers=(256, 128, 64), dropout=0.15, **base_model)),
        Candidate("A06_ancho", PreprocessConfig(), MLPConfig(hidden_layers=(512, 256), dropout=0.15, **base_model)),
        Candidate("A07_gelu_adamw", PreprocessConfig(), MLPConfig(hidden_layers=(256, 128), activation="gelu", **base_model)),
        Candidate("A08_dropout_sin_bn", PreprocessConfig(), MLPConfig(hidden_layers=(256, 128), batch_norm=False, dropout=0.3, **base_model)),
        Candidate("A09_smooth_l1", PreprocessConfig(), MLPConfig(hidden_layers=(256, 128), loss="smooth_l1", **base_model)),
        Candidate("A10_ordinal_winsor", PreprocessConfig(encoding="ordinal", scaler="robust", winsorize=True), MLPConfig(hidden_layers=(256, 128), activation="silu", **base_model)),
    ]


def baseline_results(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> list[dict]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=target_bins(y)
    )
    preprocessor = build_preprocessor(X_train, PreprocessConfig(feature_engineering=True, add_log_features=True))
    train_matrix = preprocessor.fit_transform(X_train)
    val_matrix = preprocessor.transform(X_val)
    rows = []
    for name, estimator in [("B01_DummyMean", DummyRegressor()), ("B02_Ridge", Ridge(alpha=10.0))]:
        estimator.fit(train_matrix, y_train)
        train_metrics = regression_metrics(y_train, estimator.predict(train_matrix))
        val_metrics = regression_metrics(y_val, estimator.predict(val_matrix))
        rows.append({
            "id": name, "stage": "baseline", "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"], "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"], "best_epoch": 0,
        })
    return rows


def evaluate_holdout_candidate(candidate: Candidate, X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[dict, dict]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=target_bins(y)
    )
    preprocessor = build_preprocessor(X_train, candidate.preprocess)
    X_train_matrix = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_matrix = preprocessor.transform(X_val).astype(np.float32)
    target = TargetTransformer(candidate.preprocess.target_transform).fit(y_train)
    result = train_mlp(X_train_matrix, target.transform(y_train), X_val_matrix, y_val, target, candidate.model, seed)
    row = {
        "id": candidate.name, "stage": "ablation", "train_rmse": result.history["train_rmse"][result.best_epoch - 1],
        "val_rmse": result.metrics["rmse"], "val_mae": result.metrics["mae"],
        "val_r2": result.metrics["r2"], "best_epoch": result.best_epoch,
        "input_dim": int(X_train_matrix.shape[1]), "candidate": candidate.to_dict(),
    }
    return row, result.history


def evaluate_candidate_cv(
    candidate: Candidate, X: pd.DataFrame, y: pd.Series, folds: int, seeds: list[int],
    progress: Callable[[float], None] | None = None,
) -> dict:
    fold_rows, representative_history = [], None
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        bins = target_bins(y)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, bins), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            preprocessor = build_preprocessor(X_train, candidate.preprocess)
            X_train_matrix = preprocessor.fit_transform(X_train).astype(np.float32)
            X_val_matrix = preprocessor.transform(X_val).astype(np.float32)
            target = TargetTransformer(candidate.preprocess.target_transform).fit(y_train)
            result = train_mlp(
                X_train_matrix, target.transform(y_train), X_val_matrix, y_val, target,
                candidate.model, seed + fold,
            )
            if representative_history is None:
                representative_history = result.history
            fold_rows.append({"seed": seed, "fold": fold, "best_epoch": result.best_epoch, **result.metrics})
            if progress:
                progress(float(np.mean([row["rmse"] for row in fold_rows])))
    rmses = [row["rmse"] for row in fold_rows]
    return {
        "candidate": candidate.to_dict(), "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "mean_mae": float(np.mean([row["mae"] for row in fold_rows])),
        "mean_r2": float(np.mean([row["r2"] for row in fold_rows])),
        "median_best_epoch": int(np.median([row["best_epoch"] for row in fold_rows])),
        "folds": fold_rows, "representative_history": representative_history,
    }


def suggest_candidate(trial, smoke: bool = False) -> Candidate:
    layer_count = trial.suggest_int("layers", 1, 2 if smoke else 4)
    widths = tuple(trial.suggest_categorical(f"width_{i}", [32, 64, 128, 256, 512]) for i in range(layer_count))
    preprocess = PreprocessConfig(
        scaler=trial.suggest_categorical("scaler", ["standard", "robust"]),
        encoding=trial.suggest_categorical("encoding", ["onehot", "ordinal"]),
        feature_engineering=trial.suggest_categorical("feature_engineering", [True, False]),
        add_log_features=trial.suggest_categorical("add_log_features", [True, False]),
        winsorize=trial.suggest_categorical("winsorize", [True, False]),
        min_frequency=trial.suggest_categorical("min_frequency", [None, 2, 5]),
        target_transform=trial.suggest_categorical("target_transform", ["standard", "log1p"]),
    )
    model = MLPConfig(
        hidden_layers=widths,
        activation=trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu", "silu", "tanh"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.45),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop", "sgd"]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        momentum=trial.suggest_float("momentum", 0.5, 0.95),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        max_epochs=8 if smoke else 200,
        patience=3 if smoke else 22,
        loss=trial.suggest_categorical("loss", ["mse", "smooth_l1"]),
        scheduler=trial.suggest_categorical("scheduler", ["none", "plateau", "cosine"]),
    )
    return Candidate(f"T{trial.number:03d}", preprocess, model)


def run_optuna(
    X: pd.DataFrame, y: pd.Series, storage_path: str, timeout_seconds: float,
    n_trials: int, seed: int, smoke: bool, resume: bool,
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{storage_path.replace(chr(92), '/')}"
    study_name = "ames_mlp_v1"
    if not resume:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    for stale_trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study._storage.set_trial_state_values(stale_trial._trial_id, optuna.trial.TrialState.PRUNED)

    def objective(trial):
        candidate = suggest_candidate(trial, smoke=smoke)
        trial.set_user_attr("candidate", candidate.to_dict())
        report_step = 0

        def report(value: float):
            nonlocal report_step
            trial.report(value, step=report_step)
            report_step += 1
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = evaluate_candidate_cv(candidate, X, y, folds=2 if smoke else 3, seeds=[seed], progress=report)
        trial.set_user_attr("median_best_epoch", result["median_best_epoch"])
        trial.set_user_attr("std_rmse", result["std_rmse"])
        return result["mean_rmse"]

    study.optimize(objective, n_trials=n_trials, timeout=max(1.0, timeout_seconds), catch=(RuntimeError, ValueError))
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda trial: trial.value)
    return study, completed
