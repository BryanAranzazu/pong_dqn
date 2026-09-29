"""Replay buffer que guarda cada cuadro una sola vez.

Guardar (s, s') como pilas de 4 cuadros costaria 2 x 4 x 84 x 84 = 56 KB por
transicion: 5,6 GB para 100.000 transiciones. Pero dos pilas consecutivas
comparten 3 de sus 4 cuadros, asi que basta guardar el cuadro nuevo de cada
paso (7 KB) y reconstruir las pilas al muestrear: 700 MB para 100.000.

Convenciones:
  - `frames[i]` es el cuadro mas reciente del estado i.
  - La transicion que sale del estado i (accion, recompensa, terminal) se guarda
    en la posicion i y su estado siguiente es i+1.
  - `inicio[i]` marca el primer estado de un episodio. Al reconstruir una pila
    hacia atras no se cruza esa marca: se repite el primer cuadro, igual que
    hace FrameStackObservation al reiniciar (padding_type="reset").
  - El ultimo estado de un episodio no tiene transicion saliente (`valido`=False).
"""
from __future__ import annotations

import numpy as np


class BufferFrames:
    def __init__(
        self,
        capacidad: int = 100_000,
        forma: tuple[int, int] = (84, 84),
        n_frames: int = 4,
        semilla: int | None = None,
    ) -> None:
        self.cap = capacidad
        self.k = n_frames
        self.frames = np.zeros((capacidad, *forma), dtype=np.uint8)
        self.acciones = np.zeros(capacidad, dtype=np.int64)
        self.recompensas = np.zeros(capacidad, dtype=np.float32)
        self.terminal = np.zeros(capacidad, dtype=bool)
        self.valido = np.zeros(capacidad, dtype=bool)
        self.inicio = np.zeros(capacidad, dtype=bool)
        self.ptr = 0  # proxima posicion a escribir
        self.lleno = False
        self._ultimo: int | None = None
        self.rng = np.random.default_rng(semilla)

    # ── escritura ─────────────────────────────────────────────────────

    def _escribir(self, frame: np.ndarray, inicio: bool) -> int:
        i = self.ptr
        self.frames[i] = frame
        self.inicio[i] = inicio
        self.valido[i] = False
        self.ptr = (i + 1) % self.cap
        if self.ptr == 0:
            self.lleno = True
        return i

    def iniciar_episodio(self, obs: np.ndarray) -> None:
        """Registra la observacion devuelta por reset()."""
        self._ultimo = self._escribir(obs[-1], inicio=True)

    def agregar(self, accion: int, recompensa: float, terminal: bool, obs_sig: np.ndarray) -> None:
        """Registra un paso: la transicion desde el ultimo estado hacia `obs_sig`.

        `terminal` debe ser `terminated`, no `terminated or truncated`: al
        truncar el estado siguiente si tiene futuro y hay que hacer bootstrap.
        """
        i = self._ultimo
        if i is None:
            raise RuntimeError("llamar a iniciar_episodio() antes de agregar()")
        self.acciones[i] = accion
        self.recompensas[i] = recompensa
        self.terminal[i] = terminal
        self._ultimo = self._escribir(obs_sig[-1], inicio=False)
        self.valido[i] = True

    # ── lectura ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.cap if self.lleno else self.ptr

    def pila(self, j: int) -> np.ndarray:
        """Reconstruye el estado j como pila (k, 84, 84)."""
        idx = [j]
        cur = j
        for _ in range(self.k - 1):
            if not self.inicio[cur]:
                cur = (cur - 1) % self.cap
            idx.append(cur)
        return self.frames[idx[::-1]]

    def _indices_validos(self, n: int) -> np.ndarray:
        tope = len(self)
        elegidos: list[int] = []
        while len(elegidos) < n:
            cand = self.rng.integers(0, tope, size=2 * n)
            ok = self.valido[cand]
            if self.lleno:
                # Zona de escritura: la pila de i (o de i+1) leeria cuadros ya
                # sobrescritos por datos nuevos, o el siguiente aun no existe.
                d = (cand - self.ptr) % self.cap
                ok &= (d > 2) & (d != self.cap - 1)
            elegidos.extend(cand[ok].tolist())
        return np.asarray(elegidos[:n])

    def muestrear(self, n: int):
        idx = self._indices_validos(n)
        s = np.stack([self.pila(i) for i in idx])
        s2 = np.stack([self.pila((i + 1) % self.cap) for i in idx])
        return s, self.acciones[idx], self.recompensas[idx], s2, self.terminal[idx]
