"""Análisis de los valores Q del mejor modelo (sección 6 del README).

Reproduce con código las tres mediciones de la reflexión de resultados, que hasta ahora
no tenían script:

1. Ritmo de anotación: pasos entre un punto a favor y el siguiente en un partido
   determinista (¿la política repite la misma jugada?).
2. Acciones redundantes: diferencia media |Q(s, a) − Q(s, a')| entre acciones con el
   mismo efecto (NOOP/FIRE, RIGHT/RIGHTFIRE, LEFT/LEFTFIRE) frente a acciones con efecto
   distinto.
3. Sobreestimación: Q medio de las acciones elegidas frente al retorno descontado que el
   agente obtuvo realmente desde esos mismos estados, G_t = Σ γ^k r_{t+k}.

Uso:
    python scripts/analisis_q.py --modelo saves/dqn_v2/mejor.pt --etiqueta dqn_v2
    python scripts/analisis_q.py --modelo saves/dqn_v2_mac/mejor.pt --etiqueta dqn_v2_mac

Salida: resultados/analisis_q/<etiqueta>.json (y un resumen en pantalla).
Usa las mismas semillas de evaluación que pong_dqn.evaluar (10.000 + i) y ε = 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pong_dqn.agente import AgenteDQN  # noqa: E402
from pong_dqn.entorno import crear_entorno  # noqa: E402
from pong_dqn.evaluar import MAX_PASOS_EVAL, SEMILLA_EVAL  # noqa: E402

# Grupos de acciones con el mismo efecto en Pong (sección 2.4 del README).
GRUPOS = [(0, 1), (2, 4), (3, 5)]
PARES_IGUALES = list(GRUPOS)
PARES_DISTINTOS = [
    (i, j)
    for i, j in combinations(range(6), 2)
    if not any(i in g and j in g for g in GRUPOS)
]


def retorno_descontado(recompensas: np.ndarray, gamma: float) -> np.ndarray:
    """G_t = r_t + γ G_{t+1}, calculado hacia atrás. El partido termina de verdad
    (21 puntos), así que no hay bootstrap al final."""
    g = np.zeros_like(recompensas, dtype=np.float64)
    acumulado = 0.0
    for t in range(len(recompensas) - 1, -1, -1):
        acumulado = recompensas[t] + gamma * acumulado
        g[t] = acumulado
    return g


def jugar(agente: AgenteDQN, semilla: int, sticky: float) -> dict:
    env = crear_entorno(sticky=sticky)
    obs, _ = env.reset(seed=semilla)
    qs, acciones, rs = [], [], []
    fin, t = False, 0
    while not fin and t < MAX_PASOS_EVAL:
        q = np.asarray(agente.valores_q(obs), dtype=np.float64)
        a = int(q.argmax())
        obs, r, term, trunc, _ = env.step(a)
        qs.append(q)
        acciones.append(a)
        rs.append(float(np.sign(r)))
        fin = term or trunc
        t += 1
    env.close()
    return {"q": np.stack(qs), "a": np.asarray(acciones), "r": np.asarray(rs)}


def analizar(agente: AgenteDQN, n: int, sticky: float) -> dict:
    gamma = agente.hp.gamma
    q_elegidas, retornos = [], []
    dif_iguales, dif_distintas = [], []
    intervalos, puntajes = [], []
    uso_acciones = np.zeros(6, dtype=np.int64)

    for i in range(n):
        p = jugar(agente, SEMILLA_EVAL + i, sticky)
        q, a, r = p["q"], p["a"], p["r"]

        q_elegidas.append(q[np.arange(len(a)), a])
        retornos.append(retorno_descontado(r, gamma))

        dif_iguales.append(np.mean([np.abs(q[:, x] - q[:, y]) for x, y in PARES_IGUALES]))
        dif_distintas.append(np.mean([np.abs(q[:, x] - q[:, y]) for x, y in PARES_DISTINTOS]))

        pasos_punto = np.flatnonzero(r > 0)
        if len(pasos_punto) > 1:
            intervalos.extend(np.diff(pasos_punto).tolist())
        puntajes.append(float(r.sum()))
        uso_acciones += np.bincount(a, minlength=6)

    q_medio = float(np.concatenate(q_elegidas).mean())
    g_medio = float(np.concatenate(retornos).mean())
    intervalos = np.asarray(intervalos)
    valores, conteos = np.unique(intervalos, return_counts=True) if len(intervalos) else ([], [])
    frecuentes = sorted(zip(valores, conteos, strict=True), key=lambda x: -x[1])[:5]

    return {
        "partidos": n,
        "sticky": sticky,
        "gamma": gamma,
        "recompensa_media": float(np.mean(puntajes)),
        "ritmo_anotacion": {
            "intervalo_medio": float(intervalos.mean()) if len(intervalos) else None,
            "intervalo_min": int(intervalos.min()) if len(intervalos) else None,
            "intervalo_max": int(intervalos.max()) if len(intervalos) else None,
            "intervalos_mas_frecuentes": {
                int(v): int(c) for v, c in frecuentes
            },
        },
        "acciones_redundantes": {
            "dif_q_mismo_efecto": float(np.mean(dif_iguales)),
            "dif_q_efecto_distinto": float(np.mean(dif_distintas)),
            "uso_de_cada_accion": uso_acciones.tolist(),
        },
        "sobreestimacion": {
            "q_medio_accion_elegida": q_medio,
            "retorno_descontado_medio": g_medio,
            "sobreestimacion_relativa": (q_medio - g_medio) / abs(g_medio) if g_medio else None,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modelo", default="saves/dqn_v2/mejor.pt")
    p.add_argument("--etiqueta", default="dqn_v2")
    p.add_argument("--n", type=int, default=5, help="partidos a analizar")
    p.add_argument("--sticky", type=float, default=0.0)
    p.add_argument("--dispositivo", default="auto")
    a = p.parse_args()

    agente = AgenteDQN.cargar(Path(a.modelo), a.dispositivo)
    res = {"modelo": a.modelo, "pasos_entrenados": agente.pasos,
           **analizar(agente, a.n, a.sticky)}

    sufijo = "" if a.sticky == 0 else f"_sticky{a.sticky:.2f}"
    salida = Path("resultados/analisis_q") / f"{a.etiqueta}{sufijo}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    ra, ar, so = res["ritmo_anotacion"], res["acciones_redundantes"], res["sobreestimacion"]
    print(f"Modelo {a.modelo} ({res['pasos_entrenados']:,} pasos), {a.n} partidos, "
          f"sticky {a.sticky}")
    print(f"  recompensa media              {res['recompensa_media']:+.2f}")
    print(f"  pasos entre puntos a favor    media {ra['intervalo_medio']}, "
          f"min {ra['intervalo_min']}, max {ra['intervalo_max']}")
    print(f"  |ΔQ| mismo efecto / distinto  {ar['dif_q_mismo_efecto']:.4f} / "
          f"{ar['dif_q_efecto_distinto']:.4f}")
    print(f"  Q elegido / retorno real      {so['q_medio_accion_elegida']:.3f} / "
          f"{so['retorno_descontado_medio']:.3f} "
          f"(sobreestimación {100 * so['sobreestimacion_relativa']:.1f} %)")
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
