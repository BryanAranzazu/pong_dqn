import numpy as np
import torch

from pong_dqn.agente import AgenteDQN, Hiper


def test_objetivo_corta_bootstrap_en_terminal():
    ag = AgenteDQN(6, Hiper(), "cpu")
    r = torch.tensor([1.0, -1.0])
    s2 = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    y = ag.objetivo(r, s2, torch.tensor([1.0, 1.0]))
    assert torch.allclose(y, r)
    y2 = ag.objetivo(r, s2, torch.tensor([0.0, 0.0]))
    assert not torch.allclose(y2, r)


def test_double_dqn_evalua_con_red_objetivo():
    ag = AgenteDQN(6, Hiper(doble=True), "cpu")
    s2 = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
    y = ag.objetivo(torch.zeros(4), s2, torch.zeros(4))
    a_star = ag.q(s2).argmax(1, keepdim=True)
    esperado = 0.99 * ag.q_obj(s2).gather(1, a_star).squeeze(1)
    assert torch.allclose(y, esperado)


def test_epsilon_lineal():
    ag = AgenteDQN(6, Hiper(eps_pasos=100), "cpu")
    assert ag.epsilon() == 1.0
    ag.pasos = 50
    assert abs(ag.epsilon() - 0.505) < 1e-9
    ag.pasos = 10_000
    assert ag.epsilon() == 0.01


def test_guardar_y_cargar(tmp_path):
    ag = AgenteDQN(6, Hiper(), "cpu")
    ag.pasos, ag.episodios = 123, 4
    obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    ag.guardar(tmp_path / "m.pt")
    ag2 = AgenteDQN.cargar(tmp_path / "m.pt", "cpu")
    assert (ag2.pasos, ag2.episodios) == (123, 4)
    np.testing.assert_allclose(ag.valores_q(obs), ag2.valores_q(obs), rtol=1e-6)
