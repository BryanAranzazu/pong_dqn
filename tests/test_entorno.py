import pytest

from pong_dqn.entorno import crear_entorno


@pytest.mark.parametrize("sticky", [0.0, 0.25])
def test_sticky_llega_al_emulador(sticky):
    env = crear_entorno(sticky=sticky)
    ale = env.unwrapped.ale
    assert ale.getFloat("repeat_action_probability") == pytest.approx(sticky)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84)
    env.close()
