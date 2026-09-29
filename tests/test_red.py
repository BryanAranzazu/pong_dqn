import torch

from pong_dqn.red import QNetworkCNN, resumen


def test_formas_y_parametros():
    red = QNetworkCNN(6)
    x = torch.randint(0, 256, (5, 4, 84, 84), dtype=torch.uint8)
    assert red(x).shape == (5, 6)
    filas = resumen(red)
    formas = [f for _, f, _ in filas]
    assert formas[:3] == [(32, 20, 20), (64, 9, 9), (64, 7, 7)]
    assert sum(n for *_, n in filas) == 1_687_206
