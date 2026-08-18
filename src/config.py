"""Configuracion central y contratos del dataset Ames Housing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SEED = 23236
TARGET = "SalePrice"
ID_COLUMN = "Id"

EXPECTED_FEATURES = [
    "Id", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley",
    "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond", "Foundation", "BsmtQual",
    "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir",
    "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual",
    "TotRmsAbvGrd", "Functional", "Fireplaces", "FireplaceQu", "GarageType", "GarageYrBlt",
    "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond", "PavedDrive",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType",
    "SaleCondition",
]


@dataclass(frozen=True)
class PreprocessConfig:
    numeric_only: bool = False
    scaler: str = "standard"
    encoding: str = "onehot"
    feature_engineering: bool = True
    add_log_features: bool = True
    winsorize: bool = False
    min_frequency: int | None = 2
    target_transform: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MLPConfig:
    hidden_layers: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = True
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    batch_size: int = 64
    max_epochs: int = 220
    patience: int = 25
    min_delta: float = 1e-5
    loss: str = "mse"
    scheduler: str = "plateau"
    gradient_clip: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hidden_layers"] = list(self.hidden_layers)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MLPConfig":
        clean = dict(values)
        clean["hidden_layers"] = tuple(clean["hidden_layers"])
        return cls(**clean)

