"""Bucle de entrenamiento con evaluacion periodica y guardado del mejor modelo.

Tres archivos por corrida, en resultados/<etiqueta>/:
  episodios.csv     una fila por episodio de entrenamiento (con exploracion)
  evaluaciones.csv  una fila por punto de control (politica congelada)
y los pesos en saves/<etiqueta>/: mejor.pt (por evaluacion) y ultimo.pt.

El criterio para quedarse con un modelo es la EVALUACION, no la recompensa de
entrenamiento: la de entrenamiento mezcla la calidad de la politica con el nivel
de exploracion, que cambia todo el tiempo.
"""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from pong_dqn.agente import AgenteDQN, Hiper
from pong_dqn.buffer import BufferFrames
from pong_dqn.entorno import crear_entorno
from pong_dqn.evaluar import evaluar


def _csv(ruta: Path, cabecera: list[str]):
    nuevo = not ruta.exists()
    f = ruta.open("a", newline="")
    w = csv.writer(f)
    if nuevo:
        w.writerow(cabecera)
    return f, w


def entrenar(
    pasos_totales: int,
    *,
    etiqueta: str = "dqn",
    hp: Hiper | None = None,
    dispositivo: str = "auto",
    semilla: int = 0,
    eval_cada: int = 50_000,
    n_eval: int = 10,
    guardar_cada: int = 50_000,
    reanudar: bool = False,
    raiz: Path = Path("."),
) -> AgenteDQN:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    dir_res = raiz / "resultados" / etiqueta
    dir_saves = raiz / "saves" / etiqueta
    dir_res.mkdir(parents=True, exist_ok=True)
    dir_saves.mkdir(parents=True, exist_ok=True)

    env = crear_entorno(sticky=0.0)
    n_acc = int(env.action_space.n)

    ultimo = dir_saves / "ultimo.pt"
    if reanudar and ultimo.exists():
        agente = AgenteDQN.cargar(ultimo, dispositivo)
        print(f"Reanudando desde el paso {agente.pasos:,}. El buffer arranca vacio y se "
              f"rellena {agente.hp.inicio_aprendizaje:,} pasos antes de volver a aprender.")
    else:
        agente = AgenteDQN(n_acc, hp, dispositivo)
    h = agente.hp
    buffer = BufferFrames(h.capacidad, semilla=semilla)

    mejor = -float("inf")
    f_ev_path = dir_res / "evaluaciones.csv"
    if f_ev_path.exists():
        with f_ev_path.open() as f:
            medias = [float(r["media"]) for r in csv.DictReader(f)]
        mejor = max(medias, default=mejor)

    f_ep, w_ep = _csv(dir_res / "episodios.csv",
                      ["paso", "episodio", "recompensa", "longitud", "epsilon", "perdida_media",
                       "pasos_por_seg"])
    f_ev, w_ev = _csv(f_ev_path, ["paso", "episodio", "media", "desv", "min", "max", "ganados",
                                  "n", "pasos_medios", "segundos", "conv3_vivas",
                                  "q_std_estados"])

    # Estados fijos para vigilar si la red colapsa (misma muestra en toda la corrida).
    env_diag = crear_entorno()
    o, _ = env_diag.reset(seed=12_345)
    estados_diag = []
    for t in range(2_000):
        o, _, te, tr, _ = env_diag.step(env_diag.action_space.sample())
        if t % 8 == 0:
            estados_diag.append(o)
        if te or tr:
            o, _ = env_diag.reset()
    env_diag.close()
    estados_diag = np.stack(estados_diag)

    print(f"Dispositivo: {agente.dev} | acciones: {n_acc} | "
          f"parametros: {sum(p.numel() for p in agente.q.parameters()):,}")
    print(f"Hiperparametros: {h}")

    obs, _ = env.reset(seed=semilla + agente.episodios)
    buffer.iniciar_episodio(obs)
    arranque = agente.pasos
    visto_buffer = 0  # pasos desde que se (re)lleno el buffer en esta sesion
    r_ep, largo, perdidas = 0.0, 0, []
    t0 = t_ult = time.time()
    pasos_ult = agente.pasos

    try:
        while agente.pasos < pasos_totales:
            a = agente.actuar(obs)
            obs2, r, term, trunc, _ = env.step(a)
            # Pong ya paga +1 / -1 por punto; el recorte de Mnih et al. es aqui un no-op,
            # se deja por fidelidad al protocolo y por si se cambia de juego.
            buffer.agregar(a, float(np.sign(r)), bool(term), obs2)
            obs = obs2
            r_ep += r
            largo += 1
            agente.pasos += 1
            visto_buffer += 1

            if visto_buffer >= h.inicio_aprendizaje and agente.pasos % h.aprender_cada == 0:
                perdidas.append(agente.aprender(buffer.muestrear(h.batch_size)))
            if agente.pasos % h.sync_pasos == 0:
                agente.sincronizar()

            if term or trunc:
                agente.episodios += 1
                ahora = time.time()
                pps = (agente.pasos - pasos_ult) / max(ahora - t_ult, 1e-9)
                t_ult, pasos_ult = ahora, agente.pasos
                w_ep.writerow([agente.pasos, agente.episodios, r_ep, largo,
                               round(agente.epsilon(), 4),
                               round(float(np.mean(perdidas)), 6) if perdidas else "",
                               round(pps, 1)])
                f_ep.flush()
                if agente.episodios % 10 == 0:
                    resto = (pasos_totales - agente.pasos) / max(pps, 1e-9) / 3600
                    print(f"paso {agente.pasos:>9,} | ep {agente.episodios:>5} | "
                          f"recompensa {r_ep:+5.0f} | eps {agente.epsilon():.3f} | "
                          f"{pps:6.0f} pasos/s | faltan ~{resto:.1f} h", flush=True)
                obs, _ = env.reset()
                buffer.iniciar_episodio(obs)
                r_ep, largo, perdidas = 0.0, 0, []

            if agente.pasos % eval_cada == 0:
                te = time.time()
                m = evaluar(agente, n_eval)
                seg = time.time() - te
                sal = agente.salud(estados_diag)
                w_ev.writerow([agente.pasos, agente.episodios, m["media"], round(m["desv"], 3),
                               m["min"], m["max"], m["ganados"], m["n"],
                               round(m["pasos_medios"], 1), round(seg, 1),
                               round(sal["conv3_vivas"], 4), round(sal["q_std_estados"], 6)])
                f_ev.flush()
                marca = ""
                if m["media"] > mejor:
                    mejor = m["media"]
                    agente.guardar(dir_saves / "mejor.pt", eval_media=mejor)
                    marca = "  <- nuevo mejor, guardado"
                print(f"== EVALUACION paso {agente.pasos:,}: {m['media']:+.2f} +/- "
                      f"{m['desv']:.2f} | ganados {m['ganados']}/{m['n']} ({seg:.0f} s){marca}",
                      flush=True)
                print(f"   salud de la red: conv3 vivas {sal['conv3_vivas']:.1%} | "
                      f"variacion de Q entre estados {sal['q_std_estados']:.5f}", flush=True)
                if sal["conv3_vivas"] < 0.02 or sal["q_std_estados"] < 1e-5:
                    print("   ALERTA: la red parece colapsada (salida casi constante). "
                          "Conviene detener y revisar.", flush=True)

            if agente.pasos % guardar_cada == 0:
                agente.guardar(ultimo)
    except KeyboardInterrupt:
        print("\nInterrumpido: guardando ultimo.pt para poder reanudar con --reanudar.")
    finally:
        agente.guardar(ultimo)
        f_ep.close()
        f_ev.close()
        env.close()
        horas = (time.time() - t0) / 3600
        print(f"Fin de la sesion: {agente.pasos - arranque:,} pasos en {horas:.2f} h. "
              f"Mejor evaluacion: {mejor:+.2f}")
    return agente
