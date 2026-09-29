"""Construccion del entorno de Pong con el preprocesamiento de Mnih et al. (2015).

Cadena de envoltorios, del emulador hacia el agente:

    ALE/Pong-v5 (210x160x3 RGB, uint8, 60 cuadros/s)
      -> AtariPreprocessing: salta 4 cuadros repitiendo la accion, toma el maximo
         pixel a pixel de los dos ultimos (quita el parpadeo del emulador),
         pasa a escala de grises y reescala a 84x84; hasta 30 no-ops al reiniciar
      -> FrameStackObservation: apila las 4 ultimas observaciones -> (4, 84, 84)

El agente ve un tensor uint8 de forma (4, 84, 84). La conversion a float y la
division entre 255 se hacen dentro de la red, para guardar el buffer en uint8
(4 veces menos memoria que float32).
"""
from __future__ import annotations

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
N_FRAMES = 4
LADO = 84


def crear_entorno(
    *,
    sticky: float = 0.0,
    render_mode: str | None = None,
    n_frames: int = N_FRAMES,
) -> gym.Env:
    """Pong con el preprocesamiento estandar de DQN.

    `sticky` es la probabilidad de que el emulador repita la accion anterior en
    vez de la elegida (Machado et al., 2018). v5 trae 0.25 por defecto; se
    entrena con 0.0 y se evalua con ambos valores para medir robustez.
    """
    env = gym.make(
        ENV_ID,
        frameskip=1,  # el salto lo hace AtariPreprocessing, con max-pooling
        repeat_action_probability=sticky,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=LADO,
        terminal_on_life_loss=False,  # Pong no tiene vidas
        grayscale_obs=True,
        scale_obs=False,  # se mantiene uint8
    )
    return FrameStackObservation(env, n_frames)
