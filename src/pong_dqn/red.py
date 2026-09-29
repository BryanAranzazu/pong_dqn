"""Red convolucional de Mnih et al. (2015), "Nature DQN".

    entrada  (4, 84, 84) uint8 -> float / 255
    conv1    32 filtros 8x8, stride 4  -> (32, 20, 20)   ReLU
    conv2    64 filtros 4x4, stride 2  -> (64,  9,  9)   ReLU
    conv3    64 filtros 3x3, stride 1  -> (64,  7,  7)   ReLU
    aplanar                            -> 3136
    fc       512                                         ReLU
    salida   n_acciones (sin activacion: los valores Q no estan acotados)

Tamano de salida de una convolucion sin relleno: (L - k) / s + 1.
    84 -> (84 - 8) / 4 + 1 = 20 -> (20 - 4) / 2 + 1 = 9 -> (9 - 3) / 1 + 1 = 7
"""
from __future__ import annotations

import torch
from torch import nn


class QNetworkCNN(nn.Module):
    def __init__(self, n_acciones: int, n_frames: int = 4) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.cabeza = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_acciones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.convs(x.float() / 255.0))


def resumen(red: nn.Module, n_frames: int = 4, lado: int = 84) -> list[tuple[str, tuple, int]]:
    """(capa, forma de salida, parametros) de cada capa con pesos."""
    filas = []
    x = torch.zeros(1, n_frames, lado, lado)
    x = x / 255.0
    for modulo in [*red.convs, *red.cabeza]:
        x = modulo(x)
        n = sum(p.numel() for p in modulo.parameters())
        if n:
            filas.append((modulo.__class__.__name__, tuple(x.shape[1:]), n))
    return filas
