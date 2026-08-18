"""Transformaciones de features y objetivo ajustadas exclusivamente con entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from .config import ID_COLUMN, PreprocessConfig

ORDINAL_CATEGORIES = {
    "ExterQual": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["Missing", "None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["Missing", "None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["Missing", "None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["Missing", "None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["Missing", "None", "Po", "Fa", "TA", "Gd", "Ex"],
}

LOG_CANDIDATES = [
    "LotArea", "LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "MiscVal",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, enabled: bool = True, add_log_features: bool = True):
        self.enabled = enabled
        self.add_log_features = add_log_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if ID_COLUMN in result:
            result = result.drop(columns=ID_COLUMN)
        if self.enabled:
            result["HouseAge"] = (result["YrSold"] - result["YearBuilt"]).clip(lower=0)
            result["RemodAge"] = (result["YrSold"] - result["YearRemodAdd"]).clip(lower=0)
            result["GarageAge"] = (result["YrSold"] - result["GarageYrBlt"]).clip(lower=0)
            result["TotalSF"] = result["TotalBsmtSF"] + result["1stFlrSF"] + result["2ndFlrSF"]
            result["TotalBathrooms"] = (
                result["FullBath"] + 0.5 * result["HalfBath"]
                + result["BsmtFullBath"] + 0.5 * result["BsmtHalfBath"]
            )
            result["TotalPorchSF"] = (
                result["WoodDeckSF"] + result["OpenPorchSF"] + result["EnclosedPorch"]
                + result["3SsnPorch"] + result["ScreenPorch"]
            )
            result["QualGrLivInteraction"] = result["OverallQual"] * result["GrLivArea"]
            for name, source in {
                "HasGarage": "GarageArea", "HasBasement": "TotalBsmtSF", "HasFireplace": "Fireplaces",
                "HasPool": "PoolArea", "HasSecondFloor": "2ndFlrSF",
            }.items():
                result[name] = (result[source].fillna(0) > 0).astype(float)
        if self.add_log_features:
            for column in LOG_CANDIDATES:
                if column in result:
                    values = pd.to_numeric(result[column], errors="coerce").clip(lower=0)
                    result[f"log1p_{column}"] = np.log1p(values)
        return result


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


@dataclass
class TargetTransformer:
    mode: str = "standard"

    def fit(self, y):
        values = np.asarray(y, dtype=float)
        base = np.log1p(values) if self.mode == "log1p" else values
        self.mean_ = float(base.mean())
        self.scale_ = float(base.std()) or 1.0
        return self

    def transform(self, y) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        base = np.log1p(values) if self.mode == "log1p" else values
        return ((base - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, y) -> np.ndarray:
        base = np.asarray(y, dtype=float) * self.scale_ + self.mean_
        values = np.expm1(base) if self.mode == "log1p" else base
        return np.clip(values, 0.0, None)


def build_preprocessor(X: pd.DataFrame, config: PreprocessConfig) -> Pipeline:
    engineer = FeatureEngineer(config.feature_engineering, config.add_log_features)
    preview = engineer.transform(X.iloc[:2])
    numeric_columns = list(preview.select_dtypes(include=np.number).columns)
    categorical_columns = [] if config.numeric_only else list(preview.select_dtypes(exclude=np.number).columns)

    scaler = RobustScaler() if config.scaler == "robust" else StandardScaler()
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if config.winsorize:
        numeric_steps.insert(0, ("clip", QuantileClipper()))
    numeric_steps.append(("scaler", scaler))

    transformers = [("numeric", Pipeline(numeric_steps), numeric_columns)]
    ordinal_columns = [c for c in categorical_columns if c in ORDINAL_CATEGORIES] if config.encoding == "ordinal" else []
    nominal_columns = [c for c in categorical_columns if c not in ordinal_columns]
    if ordinal_columns:
        ordinal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OrdinalEncoder(
                categories=[ORDINAL_CATEGORIES[c] for c in ordinal_columns],
                handle_unknown="use_encoded_value", unknown_value=-1,
            )),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("ordinal", ordinal_pipe, ordinal_columns))
    if nominal_columns:
        onehot = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32,
            min_frequency=config.min_frequency,
        )
        nominal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", onehot),
        ])
        transformers.append(("categorical", nominal_pipe, nominal_columns))
    columns = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("features", engineer), ("columns", columns)])

