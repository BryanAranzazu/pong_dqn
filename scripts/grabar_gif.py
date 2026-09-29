"""GIF de un episodio del agente entrenado, junto a lo que ve la red.

Uso: python scripts/grabar_gif.py [--modelo saves/dqn/mejor.pt] [--pasos 600]
Salida: resultados/partida.gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from pong_dqn.agente import AgenteDQN  # noqa: E402
from pong_dqn.entorno import crear_entorno  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="saves/dqn/mejor.pt")
    p.add_argument("--pasos", type=int, default=600)
    p.add_argument("--semilla", type=int, default=10_000)
    a = p.parse_args()

    ag = AgenteDQN.cargar(RAIZ / a.modelo, "cpu")
    env = crear_entorno(render_mode="rgb_array")
    obs, _ = env.reset(seed=a.semilla)
    cuadros, marcador = [], [0, 0]
    for t in range(a.pasos):
        obs, r, term, trunc, _ = env.step(ag.actuar(obs, epsilon=0.0))
        marcador[0 if r > 0 else 1] += int(r != 0)
        if t % 2 == 0:
            color = Image.fromarray(env.render()).resize((160, 210))
            tira = np.hstack(list(obs))  # las 4 observaciones apiladas, lado a lado
            vista = Image.fromarray(tira).convert("RGB").resize((336 * 210 // 84, 210))
            lienzo = Image.new("RGB", (160 + 8 + vista.width, 210), "white")
            lienzo.paste(color, (0, 0))
            lienzo.paste(vista, (168, 0))
            cuadros.append(lienzo)
        if term or trunc:
            break
    env.close()
    out = RAIZ / "resultados" / "partida.gif"
    cuadros[0].save(out, save_all=True, append_images=cuadros[1:], duration=40, loop=0)
    print(out, f"| {len(cuadros)} cuadros | marcador agente {marcador[0]} - rival {marcador[1]}")


if __name__ == "__main__":
    main()
