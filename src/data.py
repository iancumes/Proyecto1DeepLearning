"""Carga, limpieza, auditoria y split sin fuga de datos."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EXPECTED_FEATURES, ID_COLUMN, SEED, TARGET

SEMANTIC_ABSENCE_COLUMNS = {
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature",
}


@dataclass(frozen=True)
class DataSplit:
    X_dev: pd.DataFrame
    X_test: pd.DataFrame
    y_dev: pd.Series
    y_test: pd.Series


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_string_value(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1].strip()
    return value


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(clean_string_value)
    for column in SEMANTIC_ABSENCE_COLUMNS.intersection(cleaned.columns):
        cleaned[column] = cleaned[column].fillna("None")
    return cleaned


def validate_schema(frame: pd.DataFrame, require_target: bool) -> None:
    expected = set(EXPECTED_FEATURES)
    actual_features = set(frame.columns) - {TARGET}
    missing = sorted(expected - actual_features)
    extra = sorted(actual_features - expected)
    if missing or extra:
        raise ValueError(f"Esquema invalido. Faltantes={missing}; adicionales={extra}")
    if require_target and TARGET not in frame:
        raise ValueError(f"El dataset de entrenamiento debe incluir {TARGET}.")
    if not require_target and TARGET in frame:
        return
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("Id debe ser completo y unico.")
    if require_target:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError("SalePrice debe contener precios positivos y finitos.")


def load_data(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_schema(frame, require_target=require_target)
    return clean_frame(frame)


def target_bins(y: pd.Series, bins: int = 10) -> pd.Series:
    return pd.qcut(y.rank(method="first"), q=min(bins, len(y)), labels=False, duplicates="drop")


def development_test_split(frame: pd.DataFrame, seed: int = SEED) -> DataSplit:
    X = frame[EXPECTED_FEATURES].copy()
    y = frame[TARGET].astype(float).copy()
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, shuffle=True, stratify=target_bins(y)
    )
    return DataSplit(X_dev, X_test, y_dev, y_test)


def quality_report(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include=np.number)
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))).sum()
    apostrophe_count = 0
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype(str)
        apostrophe_count += int((values.str.startswith("'") | values.str.endswith("'")).sum())
    def absent(column: str) -> pd.Series:
        return frame[column].isna() | frame[column].eq("None")
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "numeric_features": int(len(numeric.columns) - (TARGET in numeric.columns)),
        "categorical_features": int(len(frame.select_dtypes(exclude=np.number).columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "total_missing": int(frame.isna().sum().sum()),
        "columns_with_missing": int((frame.isna().sum() > 0).sum()),
        "remaining_wrapped_apostrophes": apostrophe_count,
        "target_mean": float(frame[TARGET].mean()),
        "target_median": float(frame[TARGET].median()),
        "target_std": float(frame[TARGET].std()),
        "target_min": float(frame[TARGET].min()),
        "target_max": float(frame[TARGET].max()),
        "target_skew": float(frame[TARGET].skew()),
        "log_target_skew": float(np.log1p(frame[TARGET]).skew()),
        "extreme_living_area": frame.loc[
            frame["GrLivArea"] > 4000, ["Id", "GrLivArea", TARGET]
        ].to_dict(orient="records"),
        "consistency_checks": {
            "garage_absence_mismatches": int((absent("GarageType") != frame["GarageCars"].eq(0)).sum()),
            "basement_absence_mismatches": int((absent("BsmtQual") != frame["TotalBsmtSF"].eq(0)).sum()),
            "fireplace_absence_mismatches": int((absent("FireplaceQu") != frame["Fireplaces"].eq(0)).sum()),
            "pool_absence_mismatches": int((absent("PoolQC") != frame["PoolArea"].eq(0)).sum()),
        },
        "top_iqr_outliers": {k: int(v) for k, v in outliers.sort_values(ascending=False).head(15).items()},
        "missing_rates": {k: float(v) for k, v in frame.isna().mean().sort_values(ascending=False).items() if v > 0},
    }
