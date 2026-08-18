"""MLP parametrizable para regresion."""

from __future__ import annotations

import torch
from torch import nn

from .config import MLPConfig


def activation(name: str) -> nn.Module:
    choices = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name not in choices:
        raise ValueError(f"Activacion no soportada: {name}")
    return choices[name]()


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, config: MLPConfig):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_layers:
            linear = nn.Linear(previous, width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

