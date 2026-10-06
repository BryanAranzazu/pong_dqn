"""Linea de comandos: pong-dqn <inspeccionar|entrenar|evaluar|benchmark>."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def cmd_inspeccionar(_a) -> None:
    import numpy as np

    from pong_dqn.entorno import crear_entorno
    from pong_dqn.red import QNetworkCNN, resumen

    env = crear_entorno()
    obs, _ = env.reset(seed=0)
    base = env.unwrapped
    print("Observacion cruda del emulador :", base.observation_space)
    print("Observacion que ve el agente   :", env.observation_space)
    print("  forma", obs.shape, "dtype", obs.dtype, "rango", int(obs.min()), "-", int(obs.max()))
    print("Acciones:", env.action_space, base.get_action_meanings())
    print("Tope de pasos (cuadros del emulador):", env.spec.max_episode_steps if env.spec else None)
    rs = []
    for _ in range(300):
        obs, r, te, tr, _ = env.step(env.action_space.sample())
        if r:
            rs.append(r)
        if te or tr:
            break
    print("Recompensas distintas de cero en 300 pasos aleatorios:", sorted(set(rs)), "n =", len(rs))
    print("\nRed:")
    total = 0
    for nombre, forma, n in resumen(QNetworkCNN(int(env.action_space.n))):
        total += n
        print(f"  {nombre:8s} salida {str(forma):16s} parametros {n:>10,}")
    print(f"  total {total:,}")
    np.save("docs/ejemplo_observacion.npy", obs)


def cmd_entrenar(a) -> None:
    from pong_dqn.agente import Hiper
    from pong_dqn.entrenar import entrenar

    hp = Hiper(doble=a.doble, aprender_cada=a.aprender_cada, lr=a.lr, eps_pasos=a.eps_pasos,
               capacidad=a.capacidad, adam_eps=a.adam_eps)
    entrenar(a.pasos, etiqueta=a.etiqueta, hp=hp, dispositivo=a.dispositivo, semilla=a.semilla,
             eval_cada=a.eval_cada, n_eval=a.n_eval, reanudar=a.reanudar, sticky=a.sticky)


def cmd_evaluar(a) -> None:
    from pong_dqn.agente import AgenteDQN
    from pong_dqn.evaluar import evaluar

    ag = AgenteDQN.cargar(Path(a.modelo), a.dispositivo)
    t = time.time()
    m = evaluar(ag, a.n, sticky=a.sticky, epsilon=a.epsilon)
    m.update(modelo=a.modelo, pasos_entrenados=ag.pasos, sticky=a.sticky, epsilon=a.epsilon,
             segundos=round(time.time() - t, 1))
    print(json.dumps({k: v for k, v in m.items() if k != "recompensas"}, indent=2))
    print("recompensas:", m["recompensas"])
    if a.salida:
        Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(a.salida).write_text(json.dumps(m, indent=2))


def cmd_benchmark(a) -> None:
    """Mide pasos por segundo en esta maquina para estimar cuanto tardara."""
    from pong_dqn.agente import AgenteDQN
    from pong_dqn.buffer import BufferFrames
    from pong_dqn.entorno import crear_entorno

    env = crear_entorno()
    ag = AgenteDQN(int(env.action_space.n), dispositivo=a.dispositivo)
    buf = BufferFrames(20_000)
    obs, _ = env.reset(seed=0)
    buf.iniciar_episodio(obs)
    for _ in range(2_000):
        act = env.action_space.sample()
        obs, r, te, tr, _ = env.step(act)
        buf.agregar(act, r, te, obs)
        if te or tr:
            obs, _ = env.reset()
            buf.iniciar_episodio(obs)
    t, n = time.time(), 0
    while time.time() - t < a.segundos:
        act = ag.actuar(obs, epsilon=0.0)
        obs, r, te, tr, _ = env.step(act)
        buf.agregar(act, r, te, obs)
        ag.aprender(buf.muestrear(32))
        n += 1
        if te or tr:
            obs, _ = env.reset()
            buf.iniciar_episodio(obs)
    pps = n / (time.time() - t)
    print(f"Dispositivo {ag.dev}: {pps:.0f} pasos/s con un gradiente por paso")
    for m in (500_000, 1_000_000, 1_500_000):
        print(f"  {m:>9,} pasos  ~ {m / pps / 3600:.1f} h (+ ~10 % por las evaluaciones)")


def main() -> None:
    p = argparse.ArgumentParser(prog="pong-dqn")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspeccionar", help="espacios, dtype, recompensas y resumen de la red")

    e = sub.add_parser("entrenar", help="entrena y evalua periodicamente")
    e.add_argument("--pasos", type=int, default=1_000_000)
    e.add_argument("--etiqueta", default="dqn")
    e.add_argument("--doble", action="store_true", help="Double DQN")
    e.add_argument("--aprender-cada", type=int, default=1)
    e.add_argument("--lr", type=float, default=1e-4)
    e.add_argument("--eps-pasos", type=int, default=250_000)
    e.add_argument("--capacidad", type=int, default=100_000)
    e.add_argument("--adam-eps", type=float, default=1.5e-4)
    e.add_argument("--eval-cada", type=int, default=50_000)
    e.add_argument("--n-eval", type=int, default=10)
    e.add_argument("--semilla", type=int, default=0)
    e.add_argument("--dispositivo", default="auto")
    e.add_argument("--reanudar", action="store_true")
    e.add_argument("--sticky", type=float, default=0.0,
                   help="acciones pegajosas al entrenar (0.25 = Machado et al., 2018)")

    v = sub.add_parser("evaluar", help="evalua un modelo guardado")
    v.add_argument("--modelo", default="saves/dqn/mejor.pt")
    v.add_argument("--n", type=int, default=30)
    v.add_argument("--sticky", type=float, default=0.0)
    v.add_argument("--epsilon", type=float, default=0.0)
    v.add_argument("--dispositivo", default="auto")
    v.add_argument("--salida", default="")

    b = sub.add_parser("benchmark", help="estima la velocidad de entrenamiento")
    b.add_argument("--segundos", type=float, default=30)
    b.add_argument("--dispositivo", default="auto")

    a = p.parse_args()
    {"inspeccionar": cmd_inspeccionar, "entrenar": cmd_entrenar,
     "evaluar": cmd_evaluar, "benchmark": cmd_benchmark}[a.cmd](a)


if __name__ == "__main__":
    main()
