"""Figuras estaticas y honestas para el notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2563A6"
ORANGE = "#D97706"
INK = "#263238"
GRID = "#D8DEE4"


def _style(ax, title: str, subtitle: str):
    ax.set_title(title, loc="left", fontsize=13, color=INK, weight="bold", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#586069", va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save_eda_figures(frame: pd.DataFrame, directory: str | Path) -> None:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(frame["SalePrice"], bins=35, color=BLUE, edgecolor="white")
    _style(axes[0], "Distribución de SalePrice", "Precio de venta en USD; n = 1,168")
    axes[0].set_xlabel("USD"); axes[0].set_ylabel("Viviendas")
    axes[1].hist(np.log1p(frame["SalePrice"]), bins=35, color=ORANGE, edgecolor="white")
    _style(axes[1], "Distribución de log1p(SalePrice)", "Transformación comparada durante el modelado")
    axes[1].set_xlabel("log(1 + USD)"); axes[1].set_ylabel("Viviendas")
    fig.tight_layout(); fig.savefig(directory / "target_distribution.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    missing = frame.isna().mean().sort_values(ascending=True)
    missing = missing[missing > 0].tail(15) * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(missing.index, missing.values, color=BLUE)
    _style(ax, "Variables con mayor proporción de nulos", "Porcentaje sobre 1,168 viviendas; varios nulos representan ausencia física")
    ax.set_xlabel("Filas con valor nulo (%)")
    fig.tight_layout(); fig.savefig(directory / "missingness.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    numeric = frame.select_dtypes(include=np.number)
    corr = numeric.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(corr.index, corr.values, color=BLUE)
    _style(ax, "Correlación numérica absoluta con SalePrice", "Correlación de Pearson; relación no implica causalidad")
    ax.set_xlabel("|correlación|"); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(directory / "correlations.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    associations = {}
    for column in frame.select_dtypes(exclude=np.number).columns:
        subset = frame[[column, "SalePrice"]].copy()
        subset[column] = subset[column].fillna("Missing").astype(str)
        overall = subset["SalePrice"].mean()
        grouped = subset.groupby(column)["SalePrice"].agg(["mean", "count"])
        numerator = float((grouped["count"] * (grouped["mean"] - overall) ** 2).sum())
        denominator = float(((subset["SalePrice"] - overall) ** 2).sum())
        associations[column] = numerator / denominator if denominator else 0.0
    association = pd.Series(associations).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(association.index, association.values, color=ORANGE)
    _style(ax, "Asociación categórica con SalePrice", "η² descriptivo sobre 1,168 viviendas; asociación no implica causalidad")
    ax.set_xlabel("Proporción descriptiva de varianza entre categorías (η²)"); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(directory / "categorical_associations.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def save_experiment_figure(results: pd.DataFrame, path: str | Path) -> None:
    data = results.dropna(subset=["val_rmse"]).sort_values("val_rmse").head(12).sort_values("val_rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["id"], data["val_rmse"], color=BLUE)
    _style(ax, "RMSE de validación por experimento", "Menor es mejor; la columna stage identifica holdout o validación cruzada")
    ax.set_xlabel("RMSE (USD)")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def save_learning_curve(history: dict, path: str | Path) -> None:
    epochs = np.arange(1, len(history["val_rmse"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(epochs, history["train_rmse"], label="Entrenamiento", color=BLUE)
    ax.plot(epochs, history["val_rmse"], label="Validación", color=ORANGE)
    _style(ax, "Curvas de aprendizaje del modelo seleccionado", "RMSE en USD por época; el checkpoint conserva el mínimo de validación")
    ax.set_xlabel("Época"); ax.set_ylabel("RMSE (USD)"); ax.legend(frameon=False)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def save_residual_figures(y_true, y_pred, path: str | Path) -> None:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_pred, residuals, s=22, alpha=0.65, color=BLUE, edgecolor="none")
    axes[0].axhline(0, color=INK, linewidth=1)
    _style(axes[0], "Residuos frente a predicción", "Test interno aislado; residuo = real − predicción")
    axes[0].set_xlabel("Predicción (USD)"); axes[0].set_ylabel("Residuo (USD)")
    axes[1].scatter(y_true, y_pred, s=22, alpha=0.65, color=ORANGE, edgecolor="none")
    lower, upper = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([lower, upper], [lower, upper], color=INK, linewidth=1, linestyle="--")
    _style(axes[1], "Precio real frente a predicción", "Test interno aislado; línea diagonal representa predicción perfecta")
    axes[1].set_xlabel("Precio real (USD)"); axes[1].set_ylabel("Predicción (USD)")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
