"""Curva de robustez frente a acciones pegajosas.

Lee las evaluaciones finales de cada corrida (resultados/<corrida>/final_sticky*.json y
resultados/robustez_sticky/<corrida>_sticky*.json), calcula para cada nivel de
estocasticidad la media, un intervalo bootstrap del 95 % y los partidos ganados, y
compara las dos semillas con la prueba U de Mann-Whitney y la probabilidad de mejora.

Uso:
    uv run python scripts/robustez.py dqn_v2 dqn_v2_mac
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RES = Path("resultados")
SALIDA = RES / "robustez_sticky"
NOMBRES = {"dqn_v2": "semilla 1 (Colab)", "dqn_v2_mac": "semilla 0 (Mac)"}


def cargar(corrida: str) -> dict[float, np.ndarray]:
    datos = {}
    for ruta in list((RES / corrida).glob("final_sticky*.json")) + list(
        SALIDA.glob(f"{corrida}_sticky*.json")
    ):
        d = json.loads(ruta.read_text())
        datos[round(float(d["sticky"]), 2)] = np.asarray(d["recompensas"], dtype=float)
    return dict(sorted(datos.items()))


def ic_bootstrap(x: np.ndarray, b: int = 10_000, semilla: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(semilla)
    medias = x[rng.integers(0, len(x), (b, len(x)))].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def mann_whitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """U de x frente a y (empates cuentan 0,5) y p bilateral por aproximación normal."""
    u = float(sum((xi > y).sum() + 0.5 * (xi == y).sum() for xi in x))
    n1, n2 = len(x), len(y)
    todos = np.concatenate([x, y])
    _, conteos = np.unique(todos, return_counts=True)
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - (conteos**3 - conteos).sum() / (n * (n - 1)))
    if var <= 0:
        return u, 1.0
    z = (u - n1 * n2 / 2) / np.sqrt(var)
    from math import erfc, sqrt

    return u, float(erfc(abs(z) / sqrt(2)))


def main(corridas: list[str]) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = {c: cargar(c) for c in corridas}
    filas = []
    for c, por_nivel in datos.items():
        for s, x in por_nivel.items():
            lo, hi = ic_bootstrap(x)
            filas.append({"corrida": c, "sticky": s, "n": len(x), "media": round(x.mean(), 2),
                          "ic95_inf": round(lo, 2), "ic95_sup": round(hi, 2),
                          "ganados": int((x > 0).sum()), "min": x.min(), "max": x.max()})
    with (SALIDA / "resumen.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    if len(corridas) == 2:
        a, b = corridas
        print(f"\nComparación {a} frente a {b} por nivel de estocasticidad")
        comparaciones = []
        for s in sorted(set(datos[a]) & set(datos[b])):
            x, y = datos[a][s], datos[b][s]
            # Excluir el nivel determinista degenerado de la familia de contrastes.
            if np.all(x == x[0]) and np.all(y == x[0]):
                continue
            u, p = mann_whitney(x, y)
            comparaciones.append({"sticky": s, "U": u, "p": p,
                                  "probabilidad_mejora": u / (len(x) * len(y))})
        orden = sorted(range(len(comparaciones)), key=lambda i: comparaciones[i]["p"])
        acumulado = 0.0
        for rango, i in enumerate(orden):
            acumulado = max(acumulado, (len(orden) - rango) * comparaciones[i]["p"])
            comparaciones[i]["p_holm"] = min(1.0, acumulado)
        if comparaciones:
            with (SALIDA / "comparacion_semillas.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(comparaciones[0]))
                w.writeheader()
                w.writerows(comparaciones)
        for fila in comparaciones:
            print(fila)
        print("Probabilidad de mejora = P(X > Y) + 0.5 * P(X = Y).")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, por_nivel in datos.items():
        s = np.array(list(por_nivel))
        m = np.array([x.mean() for x in por_nivel.values()])
        ic = np.array([ic_bootstrap(x) for x in por_nivel.values()])
        ax.plot(s, m, "o-", label=NOMBRES.get(c, c))
        ax.fill_between(s, ic[:, 0], ic[:, 1], alpha=0.2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(-20.27, color="purple", ls=":", lw=1, label="agente aleatorio (−20,27)")
    ax.axvline(0.25, color="black", ls="--", lw=0.8, label="protocolo de Machado et al. (0,25)")
    ax.set(xlabel="probabilidad de acción pegajosa", ylabel="recompensa media (30 partidos)",
           title="Robustez del mejor modelo frente a la estocasticidad del emulador",
           ylim=(-22, 22))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(SALIDA / "curva_robustez.png", dpi=150)
    print(f"\nTabla en {SALIDA / 'resumen.csv'} y figura en {SALIDA / 'curva_robustez.png'}")
    for fila in filas:
        print(fila)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dqn_v2", "dqn_v2_mac"])
