"""Curva de entrenamiento y evaluaciones de una o varias corridas.

Uso: python scripts/graficar.py [etiqueta ...]   (por defecto: dqn)
Salida: resultados/curva_<etiquetas>.png
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path.cwd() / "resultados"  # se corre desde la carpeta de trabajo (repo o Drive)
VENTANA = 20
COLORES = ["#1f5f8b", "#b5541c", "#3a7d44", "#7a3e9d"]
NOMBRES = {
    "colapso_adam_eps_1e-8": "eps de Adam 1e-8 (colapso)",
    "dqn_v2": "semilla 1 (Colab, T4)",
    "dqn_v2_mac": "semilla 0 (Mac, MPS)",
}


def leer(ruta):
    if not ruta.exists():
        return []
    with ruta.open() as f:
        return list(csv.DictReader(f))


def main(etiquetas):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for et, color in zip(etiquetas, COLORES, strict=False):
        ep = leer(RES / et / "episodios.csv")
        ev = leer(RES / et / "evaluaciones.csv")
        if ep:
            x = np.array([int(r["paso"]) for r in ep])
            y = np.array([float(r["recompensa"]) for r in ep])
            ax1.plot(x, y, color=color, alpha=0.15, lw=0.8)
            if len(y) >= VENTANA:
                mm = np.convolve(y, np.ones(VENTANA) / VENTANA, mode="valid")
                etiq = NOMBRES.get(et, et)
                ax1.plot(x[VENTANA - 1:], mm, color=color, lw=2, label=etiq)
        if ev:
            x = [int(r["paso"]) for r in ev]
            m = [float(r["media"]) for r in ev]
            d = [float(r["desv"]) for r in ev]
            ax2.errorbar(x, m, yerr=d, color=color, marker="o", capsize=4, lw=2,
                         label=NOMBRES.get(et, et))
    for ax in (ax1, ax2):
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(21, color="0.3", ls=":", lw=1)
        ax.axhline(-21, color="0.3", ls=":", lw=1)
        ax.set_xlabel("pasos del agente (1 paso = 4 cuadros)")
        ax.set_ylim(-22.5, 22.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    ax1.set_ylabel("recompensa por episodio (puntos a favor - en contra)")
    ax1.set_title(f"Entrenamiento (con exploracion, media movil de {VENTANA} episodios)")
    ax2.set_ylabel("recompensa media")
    ax2.set_title("Evaluacion (politica congelada, semillas fijas)")
    fig.suptitle("Pong: DQN desde pixeles", fontsize=13)
    fig.tight_layout()
    out = RES / f"curva_{'_'.join(etiquetas)}.png"
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dqn"])
