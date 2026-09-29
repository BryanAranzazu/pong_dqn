"""El buffer debe reconstruir EXACTAMENTE las pilas que entrego el entorno.

Se fuerza que el buffer de la vuelta (capacidad chica) y que haya muchos cortes
de episodio (reinicios cada 20 a 60 pasos), que es donde un buffer por cuadros
suele equivocarse: pilas que mezclan dos episodios o que leen cuadros ya
sobrescritos.
"""
import random

import numpy as np

from pong_dqn.buffer import BufferFrames
from pong_dqn.entorno import crear_entorno


def test_reconstruccion_exacta_con_vuelta_y_cortes():
    random.seed(0)
    env = crear_entorno()
    buf = BufferFrames(capacidad=257, semilla=0)
    verdad = {}
    obs, _ = env.reset(seed=0)
    buf.iniciar_episodio(obs)
    corte = random.randint(20, 60)
    t = 0
    for _ in range(1_500):
        a = env.action_space.sample()
        obs2, r, term, trunc, _ = env.step(a)
        i = buf._ultimo
        buf.agregar(a, float(r), bool(term), obs2)
        verdad[i] = (obs.copy(), a, float(r), obs2.copy(), bool(term))
        obs = obs2
        t += 1
        if term or trunc or t >= corte:
            obs, _ = env.reset()
            buf.iniciar_episodio(obs)
            corte, t = random.randint(20, 60), 0
    assert buf.lleno, "la prueba debe forzar la vuelta del buffer circular"

    idx = buf._indices_validos(2_000)
    for i in np.unique(idx):
        s, a, r, s2, term = verdad[int(i)]
        np.testing.assert_array_equal(buf.pila(int(i)), s)
        np.testing.assert_array_equal(buf.pila((int(i) + 1) % buf.cap), s2)
        assert buf.acciones[i] == a and buf.recompensas[i] == r and buf.terminal[i] == term


def test_muestreo_formas_y_tipos():
    env = crear_entorno()
    buf = BufferFrames(capacidad=500, semilla=1)
    obs, _ = env.reset(seed=1)
    buf.iniciar_episodio(obs)
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, te, tr, _ = env.step(a)
        buf.agregar(a, r, te, obs)
    s, a, r, s2, term = buf.muestrear(32)
    assert s.shape == s2.shape == (32, 4, 84, 84)
    assert s.dtype == np.uint8 and a.dtype == np.int64 and term.dtype == bool
