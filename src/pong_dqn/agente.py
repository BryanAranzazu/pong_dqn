"""Agente DQN (y Double DQN opcional) para entradas de imagen."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from pong_dqn.red import QNetworkCNN


def elegir_dispositivo(pedido: str = "auto") -> torch.device:
    if pedido != "auto":
        return torch.device(pedido)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Hiper:
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    capacidad: int = 100_000
    inicio_aprendizaje: int = 10_000  # pasos con politica aleatoria antes del primer gradiente
    sync_pasos: int = 1_000           # la red objetivo se copia cada N pasos de entorno
    aprender_cada: int = 1            # un paso de gradiente cada N pasos de entorno
    eps_inicio: float = 1.0
    eps_fin: float = 0.01
    eps_pasos: int = 250_000          # decaimiento lineal de eps_inicio a eps_fin
    doble: bool = False
    clip_grad: float = 10.0
    # eps de Adam. El valor por defecto de PyTorch (1e-8) hace que Adam normalice
    # gradientes diminutos a pasos de tamano ~lr en direcciones de ruido; en Pong,
    # donde casi todos los lotes tienen recompensa 0, eso mato todas las unidades
    # de la tercera convolucion en menos de 50.000 pasos (ver README, seccion 7).
    # 1.5e-4 es el valor de Rainbow (Hessel et al., 2018).
    adam_eps: float = 1.5e-4


class AgenteDQN:
    def __init__(self, n_acciones: int, hp: Hiper | None = None, dispositivo: str = "auto") -> None:
        self.hp = hp or Hiper()
        self.n_acciones = n_acciones
        self.dev = elegir_dispositivo(dispositivo)
        self.q = QNetworkCNN(n_acciones).to(self.dev)
        self.q_obj = QNetworkCNN(n_acciones).to(self.dev)
        self.q_obj.load_state_dict(self.q.state_dict())
        self.q_obj.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=self.hp.lr, eps=self.hp.adam_eps)
        self.perdida = nn.SmoothL1Loss()
        self.pasos = 0      # pasos de entorno acumulados
        self.episodios = 0

    # ── politica ──────────────────────────────────────────────────────

    def epsilon(self) -> float:
        h = self.hp
        if self.pasos >= h.eps_pasos:
            return h.eps_fin
        frac = self.pasos / h.eps_pasos
        return h.eps_inicio + frac * (h.eps_fin - h.eps_inicio)

    @torch.no_grad()
    def valores_q(self, obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(obs), device=self.dev).unsqueeze(0)
        return self.q(t).squeeze(0).cpu().numpy()

    def actuar(self, obs: np.ndarray, *, epsilon: float | None = None) -> int:
        eps = self.epsilon() if epsilon is None else epsilon
        if random.random() < eps:
            return random.randrange(self.n_acciones)
        return int(self.valores_q(obs).argmax())

    # ── aprendizaje ───────────────────────────────────────────────────

    def objetivo(self, r: torch.Tensor, s2: torch.Tensor, term: torch.Tensor) -> torch.Tensor:
        """y = r + gamma * Q_obj(s', a*) * (1 - terminal)."""
        with torch.no_grad():
            if self.hp.doble:
                # Double DQN: la red en linea elige a*, la objetivo la evalua.
                a_star = self.q(s2).argmax(dim=1, keepdim=True)
                q2 = self.q_obj(s2).gather(1, a_star).squeeze(1)
            else:
                q2 = self.q_obj(s2).max(dim=1).values
            return r + self.hp.gamma * q2 * (1.0 - term)

    def aprender(self, lote) -> float:
        s, a, r, s2, term = lote
        d = self.dev
        s = torch.as_tensor(s, device=d)
        s2 = torch.as_tensor(s2, device=d)
        a = torch.as_tensor(a, device=d).unsqueeze(1)
        r = torch.as_tensor(r, device=d)
        term = torch.as_tensor(term, dtype=torch.float32, device=d)

        q = self.q(s).gather(1, a).squeeze(1)
        y = self.objetivo(r, s2, term)
        loss = self.perdida(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.hp.clip_grad)
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def salud(self, estados: np.ndarray) -> dict:
        """Diagnostico de colapso: unidades vivas de conv3 y cuanto varia Q entre estados.

        Si ninguna unidad de la ultima convolucion se activa para ningun estado, la red
        devuelve la misma salida para toda entrada: ya no ve el juego.
        """
        x = torch.as_tensor(estados, device=self.dev).float() / 255.0
        h = self.q.convs(x)
        q = self.q.cabeza(h)
        return {
            "conv3_vivas": float((h > 0).any(dim=0).float().mean().item()),
            "q_std_estados": float(q.std(dim=0).mean().item()),
        }

    def sincronizar(self) -> None:
        self.q_obj.load_state_dict(self.q.state_dict())

    # ── persistencia ──────────────────────────────────────────────────

    def guardar(self, ruta: Path, **extra) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hp": asdict(self.hp),
                "n_acciones": self.n_acciones,
                "q": self.q.state_dict(),
                "q_obj": self.q_obj.state_dict(),
                "opt": self.opt.state_dict(),
                "pasos": self.pasos,
                "episodios": self.episodios,
                **extra,
            },
            ruta,
        )

    @classmethod
    def cargar(cls, ruta: Path, dispositivo: str = "auto") -> AgenteDQN:
        d = torch.load(Path(ruta), map_location="cpu", weights_only=False)
        ag = cls(d["n_acciones"], Hiper(**d["hp"]), dispositivo)
        ag.q.load_state_dict(d["q"])
        ag.q_obj.load_state_dict(d.get("q_obj", d["q"]))
        ag.opt.load_state_dict(d["opt"])
        ag.pasos, ag.episodios = d["pasos"], d["episodios"]
        return ag
