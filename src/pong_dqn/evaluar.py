"""Evaluacion con la politica congelada y semillas fijas."""
from __future__ import annotations

import numpy as np

from pong_dqn.agente import AgenteDQN
from pong_dqn.entorno import crear_entorno

SEMILLA_EVAL = 10_000
MAX_PASOS_EVAL = 20_000  # tope de seguridad por si la politica entra en un bucle


def evaluar(
    agente: AgenteDQN,
    n: int = 10,
    *,
    sticky: float = 0.0,
    epsilon: float = 0.0,
    semilla_base: int = SEMILLA_EVAL,
) -> dict:
    env = crear_entorno(sticky=sticky)
    rs, largos = [], []
    for i in range(n):
        obs, _ = env.reset(seed=semilla_base + i)
        total, t, fin = 0.0, 0, False
        while not fin and t < MAX_PASOS_EVAL:
            obs, r, term, trunc, _ = env.step(agente.actuar(obs, epsilon=epsilon))
            total += r
            t += 1
            fin = term or trunc
        rs.append(total)
        largos.append(t)
    env.close()
    rs = np.asarray(rs)
    return {
        "n": n,
        "media": float(rs.mean()),
        "desv": float(rs.std()),
        "min": float(rs.min()),
        "max": float(rs.max()),
        "ganados": int((rs > 0).sum()),
        "pasos_medios": float(np.mean(largos)),
        "recompensas": rs.tolist(),
    }
