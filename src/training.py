"""Entrenamiento, evaluacion y prediccion determinista del MLP."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import MLPConfig
from .model import RegressionMLP
from .preprocessing import TargetTransformer


@dataclass
class TrainingResult:
    state_dict: dict[str, torch.Tensor]
    history: dict[str, list[float]]
    best_epoch: int
    metrics: dict[str, float]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_optimizer(model: nn.Module, config: MLPConfig):
    common = {"params": model.parameters(), "lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "adam":
        return torch.optim.Adam(**common)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(**common)
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(**common, momentum=min(config.momentum, 0.9))
    if config.optimizer == "sgd":
        return torch.optim.SGD(**common, momentum=config.momentum, nesterov=config.momentum > 0)
    raise ValueError(f"Optimizador no soportado: {config.optimizer}")


def make_loss(config: MLPConfig):
    return nn.SmoothL1Loss(beta=0.5) if config.loss == "smooth_l1" else nn.MSELoss()


@torch.no_grad()
def predict_scaled(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    outputs = []
    for start in range(0, len(tensor), batch_size):
        outputs.append(model(tensor[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def train_mlp(
    X_train: np.ndarray, y_train_scaled: np.ndarray, X_val: np.ndarray, y_val,
    target_transformer: TargetTransformer, config: MLPConfig, seed: int,
) -> TrainingResult:
    set_deterministic(seed)
    model = RegressionMLP(X_train.shape[1], config)
    optimizer = make_optimizer(model, config)
    criterion = make_loss(config)
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    else:
        scheduler = None
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=min(config.batch_size, len(X_train)), shuffle=True, generator=generator, drop_last=False,
    )
    history = {"train_loss": [], "train_rmse": [], "val_rmse": [], "learning_rate": []}
    best_rmse = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            loss.backward()
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_pred = target_transformer.inverse_transform(predict_scaled(model, X_train))
        val_pred = target_transformer.inverse_transform(predict_scaled(model, X_val))
        train_true = target_transformer.inverse_transform(y_train_scaled)
        train_rmse = regression_metrics(train_true, train_pred)["rmse"]
        val_rmse = regression_metrics(y_val, val_pred)["rmse"]
        history["train_loss"].append(float(np.mean(losses)))
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step(val_rmse) if config.scheduler == "plateau" else scheduler.step()
        if val_rmse < best_rmse - config.min_delta:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    val_pred = target_transformer.inverse_transform(predict_scaled(model, X_val))
    return TrainingResult(best_state, history, best_epoch, regression_metrics(y_val, val_pred))


def fit_fixed_epochs(X: np.ndarray, y_scaled: np.ndarray, config: MLPConfig, seed: int, epochs: int):
    set_deterministic(seed)
    model = RegressionMLP(X.shape[1], config)
    optimizer = make_optimizer(model, config)
    criterion = make_loss(config)
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    else:
        scheduler = None
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y_scaled, dtype=torch.float32)),
        batch_size=min(config.batch_size, len(X)), shuffle=True, generator=generator,
    )
    losses = []
    for _ in range(max(1, epochs)):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        if scheduler is not None:
            scheduler.step(epoch_loss) if config.scheduler == "plateau" else scheduler.step()
    return model, losses
