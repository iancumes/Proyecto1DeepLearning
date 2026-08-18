"""Ejecuta el proyecto de principio a fin con un presupuesto global."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import save_model, save_transformers, write_json
from src.config import EXPECTED_FEATURES, MLPConfig, SEED
from src.data import clean_frame, development_test_split, load_data, quality_report, sha256_file, validate_schema
from src.experiments import Candidate, ablation_catalog, baseline_results, evaluate_candidate_cv, evaluate_holdout_candidate, run_optuna
from src.model import count_parameters
from src.preprocessing import TargetTransformer, build_preprocessor
from src.report import build_notebook, execute_notebook
from src.training import fit_fixed_epochs, predict_scaled, regression_metrics
from src.visualization import save_eda_figures, save_experiment_figure, save_learning_curve, save_residual_figures


class Budget:
    def __init__(self, minutes: float):
        self.started = time.monotonic()
        self.seconds = minutes * 60

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)


def status(message: str, budget: Budget) -> None:
    print(f"[{budget.elapsed / 60:6.1f} min | quedan {budget.remaining / 60:5.1f}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--budget-minutes", type=float, default=75)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=60, help="Trials nuevos de Optuna en esta ejecución")
    parser.add_argument("--search-minutes", type=float, default=None, help="Tiempo máximo dedicado solo a Optuna")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = Budget(args.budget_minutes)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    artifacts_dir = ROOT / "artifacts"
    results_dir.mkdir(exist_ok=True); figures_dir.mkdir(exist_ok=True); artifacts_dir.mkdir(exist_ok=True)

    status("Validando y auditando el dataset", budget)
    raw = pd.read_csv(data_path)
    validate_schema(raw, require_target=True)
    frame = clean_frame(raw)
    quality = {"raw": quality_report(raw), "clean": quality_report(frame)}
    write_json(results_dir / "data_quality.json", quality)
    save_eda_figures(raw, figures_dir)
    split = development_test_split(frame, args.seed)

    status("Calculando baselines", budget)
    rows = baseline_results(split.X_dev, split.y_dev, args.seed)
    histories: dict[str, dict] = {}
    candidates_by_name = {candidate.name: candidate for candidate in ablation_catalog(args.smoke)}
    status(f"Ejecutando {len(candidates_by_name)} ablaciones", budget)
    for index, candidate in enumerate(candidates_by_name.values(), start=1):
        if not args.smoke and budget.remaining < 4 * 60 and index > 6:
            status("Se omiten ablaciones restantes para proteger el entrenamiento final", budget)
            break
        row, history = evaluate_holdout_candidate(candidate, split.X_dev, split.y_dev, args.seed)
        rows.append(row); histories[candidate.name] = history
        status(f"{candidate.name}: RMSE validación = {row['val_rmse']:,.0f}", budget)

    status("Iniciando búsqueda Optuna", budget)
    if args.smoke:
        search_seconds, n_trials = min(30, max(5, budget.remaining / 3)), 2
    else:
        default_search = min(35 * 60, max(60, budget.remaining - 23 * 60))
        requested_search = args.search_minutes * 60 if args.search_minutes is not None else default_search
        search_seconds = min(requested_search, max(60, budget.remaining - 5 * 60))
        n_trials = args.trials
    study, completed = run_optuna(
        split.X_dev, split.y_dev, str(results_dir / "optuna.db"), search_seconds,
        n_trials, args.seed, args.smoke, args.resume,
    )
    study.trials_dataframe().to_csv(results_dir / "optuna_trials.csv", index=False)
    status(f"Optuna completó {len(completed)} trials válidos", budget)

    finalists: list[Candidate] = []
    previous_metrics_path = results_dir / "final_metrics.json"
    if args.resume and previous_metrics_path.exists():
        from src.artifacts import read_json
        finalists.append(Candidate.from_dict(read_json(previous_metrics_path)["selected_candidate"]))
    for trial in completed[:3]:
        candidate = Candidate.from_dict(trial.user_attrs["candidate"])
        signature = json.dumps(candidate.to_dict(), sort_keys=True)
        if all(json.dumps(existing.to_dict(), sort_keys=True) != signature for existing in finalists):
            finalists.append(candidate)
    if not finalists:
        ablation_rows = [row for row in rows if row["stage"] == "ablation"]
        best_row = min(ablation_rows, key=lambda row: row["val_rmse"])
        finalists = [candidates_by_name[best_row["id"]]]

    finalist_results = []
    finalist_limit = 1 if args.smoke or budget.remaining < 4 * 60 else min(4, len(finalists))
    folds = 2 if args.smoke else 5
    seeds = [args.seed] if args.smoke or budget.remaining < 4 * 60 else [args.seed, args.seed + 1]
    status(f"Reevaluando {finalist_limit} finalistas con {folds} folds y {len(seeds)} semilla(s)", budget)
    for candidate in finalists[:finalist_limit]:
        result = evaluate_candidate_cv(candidate, split.X_dev, split.y_dev, folds=folds, seeds=seeds)
        finalist_results.append(result)
        rows.append({
            "id": f"F_{candidate.name}", "stage": "finalist", "train_rmse": np.nan,
            "val_rmse": result["mean_rmse"], "val_mae": result["mean_mae"],
            "val_r2": result["mean_r2"], "best_epoch": result["median_best_epoch"],
            "std_rmse": result["std_rmse"], "candidate": candidate.to_dict(),
        })
        status(f"{candidate.name}: CV RMSE = {result['mean_rmse']:,.0f} ± {result['std_rmse']:,.0f}", budget)
    selected_result = min(finalist_results, key=lambda result: result["mean_rmse"])
    selected = Candidate.from_dict(selected_result["candidate"])
    final_epochs = max(5, selected_result["median_best_epoch"])

    status("Evaluando una sola vez en test interno", budget)
    dev_preprocessor = build_preprocessor(split.X_dev, selected.preprocess)
    X_dev_matrix = dev_preprocessor.fit_transform(split.X_dev).astype(np.float32)
    X_test_matrix = dev_preprocessor.transform(split.X_test).astype(np.float32)
    dev_target = TargetTransformer(selected.preprocess.target_transform).fit(split.y_dev)
    internal_predictions, dev_models = [], []
    ensemble_seeds = [args.seed + offset for offset in range(5)]
    for seed in ensemble_seeds:
        model, _ = fit_fixed_epochs(X_dev_matrix, dev_target.transform(split.y_dev), selected.model, seed, final_epochs)
        dev_models.append(model)
        internal_predictions.append(dev_target.inverse_transform(predict_scaled(model, X_test_matrix)))
    single_prediction = internal_predictions[0]
    ensemble_prediction = np.mean(internal_predictions, axis=0)
    internal_single = regression_metrics(split.y_test, single_prediction)
    internal_ensemble = regression_metrics(split.y_test, ensemble_prediction)
    pd.DataFrame({
        "Id": split.X_test["Id"].to_numpy(), "SalePrice": split.y_test.to_numpy(),
        "PredictionSingle": single_prediction, "PredictionEnsemble": ensemble_prediction,
        "ResidualSingle": split.y_test.to_numpy() - single_prediction,
    }).to_csv(results_dir / "internal_test_predictions.csv", index=False)

    status("Entrenando y guardando modelos finales con todo train.csv", budget)
    final_preprocessor = build_preprocessor(frame[EXPECTED_FEATURES], selected.preprocess)
    X_full = final_preprocessor.fit_transform(frame[EXPECTED_FEATURES]).astype(np.float32)
    final_target = TargetTransformer(selected.preprocess.target_transform).fit(frame["SalePrice"])
    save_transformers(artifacts_dir, final_preprocessor, final_target)
    ensemble_dir = artifacts_dir / "ensemble"; ensemble_dir.mkdir(exist_ok=True)
    full_losses = {}
    parameter_count = None
    for index, seed in enumerate(ensemble_seeds):
        model, losses = fit_fixed_epochs(X_full, final_target.transform(frame["SalePrice"]), selected.model, seed, final_epochs)
        parameter_count = count_parameters(model)
        path = artifacts_dir / "single_model.pt" if index == 0 else ensemble_dir / f"model_seed_{seed}.pt"
        save_model(path, model)
        if index == 0:
            save_model(ensemble_dir / f"model_seed_{seed}.pt", model)
        full_losses[str(seed)] = losses

    experiment_records = []
    for row in rows:
        record = {k: v for k, v in row.items() if k != "candidate"}
        record["config_json"] = json.dumps(row.get("candidate"), ensure_ascii=False) if row.get("candidate") else ""
        experiment_records.append(record)
    experiment_frame = pd.DataFrame(experiment_records)
    experiment_frame.to_csv(results_dir / "experiments.csv", index=False)
    write_json(results_dir / "histories.json", histories)
    write_json(results_dir / "finalists.json", finalist_results)
    write_json(results_dir / "full_training_losses.json", full_losses)
    final_metrics = {
        "run_mode": "smoke" if args.smoke else "full",
        "data_sha256": sha256_file(data_path), "seed": args.seed,
        "optuna": {
            "total_trials": len(study.trials),
            "state_counts": pd.Series([trial.state.name for trial in study.trials]).value_counts().to_dict(),
        },
        "selected_candidate": selected.to_dict(), "final_epochs": final_epochs,
        "parameter_count": parameter_count, "input_dim": int(X_full.shape[1]),
        "internal_test": {"rows": int(len(split.y_test)), "single": internal_single, "ensemble": internal_ensemble},
        "cv": {k: v for k, v in selected_result.items() if k not in {"representative_history", "folds", "candidate"}},
        "elapsed_minutes_before_report": budget.elapsed / 60,
    }
    write_json(results_dir / "final_metrics.json", final_metrics)
    metadata = {
        "data_sha256": final_metrics["data_sha256"], "feature_columns": EXPECTED_FEATURES,
        "input_dim": int(X_full.shape[1]), "model_config": selected.model.to_dict(),
        "preprocess_config": selected.preprocess.to_dict(), "target": "SalePrice",
        "final_epochs": final_epochs, "seeds": ensemble_seeds,
    }
    write_json(artifacts_dir / "metadata.json", metadata)

    save_experiment_figure(experiment_frame, figures_dir / "experiment_comparison.png")
    learning_history = selected_result["representative_history"]
    save_learning_curve(learning_history, figures_dir / "learning_curve.png")
    save_residual_figures(split.y_test, single_prediction, figures_dir / "residuals.png")

    status("Construyendo y ejecutando el notebook", budget)
    notebook_path = build_notebook(ROOT)
    execute_notebook(notebook_path, ROOT, timeout=300)
    final_metrics["elapsed_minutes_total"] = budget.elapsed / 60
    final_metrics["within_budget"] = budget.elapsed <= budget.seconds
    write_json(results_dir / "final_metrics.json", final_metrics)
    status(f"Pipeline completo. Notebook: {notebook_path}", budget)
    print(json.dumps({"single_rmse": internal_single["rmse"], "ensemble_rmse": internal_ensemble["rmse"], "within_budget": final_metrics["within_budget"]}, indent=2))


if __name__ == "__main__":
    main()
