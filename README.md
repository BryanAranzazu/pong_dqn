# Pong con Deep Q-Network desde píxeles

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.python-version)
[![Licencia Apache 2.0](https://img.shields.io/badge/licencia-Apache%202.0-green.svg)](LICENSE)
[![Citar este trabajo](https://img.shields.io/badge/citar-CITATION.cff-orange.svg)](CITATION.cff)

**Aprender a jugar viendo solo la pantalla.** Taller 2 de la Unidad 3, curso Simulación y Aprendizaje por Refuerzo, Maestría en Inteligencia Artificial, Universidad de La Sabana (Chía, Colombia), periodo 2026-2.

Autores: Leonar Socarrás Molina (leonarsomo@unisabana.edu.co), John Jairo Serrano Cifuentes (johnseci@unisabana.edu.co), Brezhnev Joya Miranda (brezhnevjomi@unisabana.edu.co), Gabriel Alonso Lizano Alvarado (gabriellial@unisabana.edu.co), Bryan Johann Aranzazu Medina (bryanarme@unisabana.edu.co), ORCID [0000-0003-0601-9151](https://orcid.org/0000-0003-0601-9151). Docente: Emilio Muñoz Pérez.

> **BORRADOR.** Las secciones 5, 6 y la parte final de la 7 se completan con los
> números del entrenamiento. Todo lo marcado como `PENDIENTE` debe desaparecer
> antes de entregar.

## Resumen

`PENDIENTE` (se escribe al final, con los resultados).

---

## 1. Por qué este ambiente

El enunciado pide un ambiente no trabajado en clase. En el encuentro sincrónico del
28 de septiembre el ejemplo de la clase fue LunarLander, con una implementación DQN
de referencia, así que ese quedó descartado. El profesor sugirió los juegos de Atari
para quien quisiera un reto mayor, porque obligan a trabajar con una red
convolucional.

Pong cumple eso y además tiene tres propiedades útiles para el taller:

- **La observación es una imagen.** Hay que decidir cómo preprocesarla, y la
  decisión de apilar cuadros deja de ser opcional (sección 2.3).
- **El estado no es markoviano cuadro a cuadro.** Es el caso de libro para discutir
  la propiedad de Markov, que fue justamente el ejemplo usado en clase.
- **Es el juego de Atari más estudiado con DQN**, así que hay una referencia clara de
  lo que significa "resuelto": ganar los partidos 21 a algo, con recompensa cercana
  a +21.

---

## 2. Acciones y observaciones

Todo lo que sigue se obtuvo del propio entorno con `pong-dqn inspeccionar`.

### 2.1 Observación del emulador (antes del preprocesamiento)

`Box(0, 255, (210, 160, 3), uint8)`: la pantalla RGB completa de la consola, 210
filas por 160 columnas, 60 cuadros por segundo.

### 2.2 Observación que recibe el agente (después del preprocesamiento)

`Box(0, 255, (4, 84, 84), uint8)`: los **4 cuadros más recientes**, en escala de
grises y reescalados a 84 × 84.

| eje | tamaño | significado |
|-----|--------|-------------|
| 0 | 4 | cuadros consecutivos, del más viejo al más nuevo |
| 1 | 84 | filas |
| 2 | 84 | columnas |

Se guarda como `uint8` y la red la convierte a `float32` dividiendo entre 255
dentro de su `forward`. Así el replay buffer ocupa la cuarta parte de memoria que
si se guardara en `float32`.

### 2.3 Preprocesamiento y por qué cada paso

| paso | qué hace | por qué |
|------|----------|---------|
| salto de 4 cuadros | repite la acción 4 cuadros y devuelve solo el último | 4 cuadros son 1/15 de segundo; decidir a 60 Hz no aporta y multiplica por 4 el cómputo |
| máximo de los 2 últimos | toma el máximo píxel a píxel de los cuadros 3 y 4 | el emulador no dibuja todos los objetos en todos los cuadros (parpadeo); sin esto la pelota puede desaparecer de la observación |
| escala de grises | 3 canales → 1 | en Pong el color no distingue nada que la intensidad no distinga ya |
| reescalado a 84 × 84 | 210 × 160 → 84 × 84 | reduce 5 veces los píxeles; la pelota sigue siendo visible |
| hasta 30 no-ops al inicio | un número aleatorio de pasos sin hacer nada tras `reset()` | el emulador es determinista; sin esto todos los episodios arrancan idénticos y el agente puede memorizar una secuencia en vez de aprender a jugar |
| **apilar 4 cuadros** | la observación son los 4 últimos cuadros | ver abajo |

**Por qué apilar cuadros.** Un solo cuadro muestra *dónde* está la pelota pero no
*hacia dónde va* ni a qué velocidad. Con una sola imagen, la misma posición de la
pelota puede requerir subir la paleta (viene hacia mí) o quedarse quieto (se aleja),
así que la observación no basta para decidir: el proceso no es markoviano y el
supuesto sobre el que descansa toda la ecuación de Bellman se rompe. Con cuatro
cuadros la red puede inferir dirección y velocidad a partir del desplazamiento
entre ellos.

La comparación con LunarLander, el ambiente de la clase, lo deja claro: allí el
vector de estado ya trae las velocidades como componentes explícitas, así que un solo
paso de observación es markoviano y apilar no aporta nada. En Pong las velocidades
no están en ningún lado excepto en la diferencia entre cuadros.

### 2.4 Espacio de acción

`Discrete(6)`, `dtype=int64`.

| acción | nombre | efecto en Pong |
|--------|--------|----------------|
| 0 | NOOP | no mover la paleta |
| 1 | FIRE | igual que NOOP (el saque es automático) |
| 2 | RIGHT | mover la paleta hacia arriba |
| 3 | LEFT | mover la paleta hacia abajo |
| 4 | RIGHTFIRE | igual que RIGHT |
| 5 | LEFTFIRE | igual que LEFT |

Hay **tres efectos distintos repartidos en seis acciones**. Se comprobó
directamente: con la misma semilla, 200 pasos de FIRE producen exactamente los mismos
cuadros que 200 de NOOP, y lo mismo RIGHTFIRE frente a RIGHT. Se dejan las seis porque
es el conjunto mínimo que expone el entorno por defecto, y porque la redundancia es
en sí una prueba: si el agente aprende bien, los valores Q de las acciones
equivalentes deberían converger a valores parecidos.

### 2.5 Recompensas

- **+1** cuando el agente anota un punto.
- **−1** cuando lo anota el rival (la IA del juego).
- **0** en cualquier otro paso.

La recompensa por episodio es `puntos a favor − puntos en contra`, en el rango
**[−21, +21]**. Es una recompensa **escasa**: en un episodio aleatorio de unos 950
pasos solo unos 20 pasos pagan algo distinto de cero.

**Línea base aleatoria** (30 episodios, semillas fijas): **−20,27 ± 0,85**, con 948
pasos por episodio en promedio. El agente aleatorio anota como mucho 3 puntos.

Mnih et al. (2015) recortan las recompensas a {−1, 0, +1} para usar los mismos
hiperparámetros en los 49 juegos. En Pong ese recorte no cambia nada porque la
recompensa ya está en ese conjunto; el código lo aplica igual por fidelidad al
protocolo.

### 2.6 Terminación del episodio

- `terminated = True` cuando alguno de los dos llega a **21 puntos**. Es un final
  real: no hay futuro que descontar.
- `truncated = True` a los **108.000 cuadros del emulador** (27.000 pasos del
  agente, unos 30 minutos de juego). Es un corte administrativo y el estado sí tiene
  continuación.

El buffer guarda `terminated`, no `terminated or truncated`, para que la ecuación de
Bellman siga haciendo bootstrap cuando el episodio se corta por tiempo.

### 2.7 Acciones pegajosas

`ALE/Pong-v5` trae por defecto `repeat_action_probability = 0.25`: con probabilidad
1/4 el emulador ignora la acción elegida y repite la anterior (Machado et al., 2018).
Esto se introdujo para que un agente no pueda explotar el determinismo del emulador.
Aquí se **entrena sin acciones pegajosas** (0,0), como en el DQN original, y se
**evalúa con ambos valores** para medir si la política depende de ese determinismo.

---

## 3. Flujo de entrenamiento

Diagrama del ciclo: carpeta `esquemas/`.

```
inicializar Q (red en línea) y Q_obj (copia), buffer vacío
obs ← env.reset()                                  (4, 84, 84) uint8
repetir hasta 1.000.000 de pasos:
  ┌─ 1. ACCIÓN ε-greedy
  │     ε decae linealmente de 1,0 a 0,01 en los primeros 250.000 pasos
  │     con prob. ε → acción al azar entre las 6; si no → argmax_a Q(obs)[a]
  ├─ 2. PASO: obs2, r, terminated, truncated ← env.step(a)
  ├─ 3. ALMACENAR: se guarda solo el cuadro nuevo de obs2, más (a, sign(r), terminated)
  ├─ 4. APRENDER (desde el paso 10.000, en cada paso):
  │     muestrear 32 transiciones; reconstruir s y s' como pilas de 4 cuadros
  │     y = r + γ · max_a' Q_obj(s')[a'] · (1 − terminated)       ← Bellman
  │     pérdida de Huber entre Q(s)[a] e y; recorte de gradiente a norma 10; Adam
  ├─ 5. SINCRONIZAR: cada 1.000 pasos, Q_obj ← Q
  ├─ 6. FIN DE EPISODIO: registrar recompensa, reset(), marcar inicio en el buffer
  └─ 7. EVALUAR cada 50.000 pasos: 10 partidos con ε = 0 y semillas fijas;
        si la media supera la mejor anterior, guardar saves/dqn/mejor.pt
```

### 3.1 Los componentes del ciclo DQN

**Replay buffer.** Rompe la correlación entre transiciones consecutivas (en Pong,
cientos de cuadros seguidos casi idénticos) y permite reutilizar cada transición en
muchos lotes. Capacidad de 100.000 transiciones.

**ε-greedy.** Exploración uniforme que decae por **pasos**, no por episodios,
porque la duración de un episodio de Pong cambia mucho a medida que el agente mejora
(unos 800 a 950 pasos perdiendo casi todos los puntos, más de 2.000 en partidos
disputados).

**Red objetivo.** Una copia congelada de la red que produce el `max_a' Q(s', a')`
del objetivo. Sin ella la red perseguiría un blanco que se mueve con cada paso de
gradiente.

**Actualización de Bellman.** `y = r + γ · max_a' Q_obj(s', a') · (1 − terminated)`,
con γ = 0,99.

### 3.2 Particularidades de este ambiente y cómo se atienden

**a) El buffer guarda cuadros, no pilas.** Guardar `s` y `s'` como pilas de 4
cuadros costaría 56 KB por transición: **5,6 GB** para 100.000 transiciones. Pero dos
pilas consecutivas comparten 3 de sus 4 cuadros, así que el buffer guarda solo el
cuadro nuevo de cada paso (7 KB) y reconstruye las pilas al muestrear: **700 MB**.
La reconstrucción tiene dos trampas: no mezclar cuadros de dos partidos distintos
(al inicio de un episodio se repite el primer cuadro, como hace el propio entorno) y
no leer cuadros que el buffer circular ya sobrescribió. `tests/test_buffer.py`
compara, para cientos de transiciones, cada pila reconstruida contra la que entregó
el entorno, forzando que el buffer dé la vuelta y que haya cortes de episodio
frecuentes.

**b) Recompensa escasa y tardía.** Con −1 por punto perdido y nada más, al principio
el único aprendizaje posible es "esto terminó mal", sin saber qué acción de los
últimos cientos de pasos lo causó. γ = 0,99 da un horizonte efectivo de unos 100
pasos, suficiente para conectar el movimiento de la paleta con el punto unos 20 a 40
pasos después.

**c) 10.000 pasos aleatorios antes del primer gradiente.** Para que los primeros
lotes no salgan de un buffer con tres jugadas.

**d) La red objetivo se sincroniza por pasos.** Por la misma razón que ε decae por
pasos: la longitud del episodio no es constante.

**e) Se evalúa aparte y se guarda el mejor por evaluación.** La recompensa de
entrenamiento mezcla la calidad de la política con el nivel de exploración. El
modelo que se reporta es el de mejor evaluación con política congelada, no el último.

---

## 4. La red neuronal

### 4.1 Arquitectura

La red de Mnih et al. (2015). Las convoluciones extraen posición y movimiento de
paletas y pelota; las capas densas convierten eso en un valor por acción.

| capa | tipo | filtros / neuronas | kernel | stride | salida | parámetros |
|------|------|--------------------|--------|--------|--------|-----------:|
| entrada | uint8 / 255 | | | | 4 × 84 × 84 | 0 |
| conv1 | Conv2d + ReLU | 32 | 8 × 8 | 4 | 32 × 20 × 20 | 8.224 |
| conv2 | Conv2d + ReLU | 64 | 4 × 4 | 2 | 64 × 9 × 9 | 32.832 |
| conv3 | Conv2d + ReLU | 64 | 3 × 3 | 1 | 64 × 7 × 7 | 36.928 |
| aplanar | | | | | 3.136 | 0 |
| fc | Linear + ReLU | 512 | | | 512 | 1.606.144 |
| salida | Linear | 6 | | | 6 | 3.078 |
| | | | | | **total** | **1.687.206** |

Tamaño de salida de cada convolución sin relleno: `(L − k) / s + 1`, que da
84 → 20 → 9 → 7.

**Por qué este diseño.**

- **La primera convolución es grande y con stride 4** porque la imagen tiene mucha
  redundancia espacial: tras el reescalado la pelota ocupa apenas uno o dos píxeles,
  y un kernel de 8 × 8 con paso 4 la captura sin gastar cómputo en el fondo.
- **Los 4 cuadros entran como 4 canales** de la primera convolución. Así cada
  filtro ve los cuatro instantes a la vez y puede aprender detectores de movimiento
  (un borde que se desplaza entre canales).
- **No hay pooling.** El pooling da invariancia a la posición, y en Pong la posición
  exacta de la pelota respecto de la paleta es justo lo que importa. Los strides
  reducen resolución sin descartar dónde está cada cosa.
- **El 95 % de los parámetros está en la capa de 3.136 a 512**, que es donde se
  combina la información espacial de toda la pantalla.
- **La salida no tiene activación**: los valores Q son retornos esperados y pueden
  ser negativos.

### 4.2 Hiperparámetros

| hiperparámetro | valor | razón |
|---|---|---|
| optimizador | Adam, lr = 1e-4 | el valor de la implementación de referencia de Pong en Lapan (2020); la red es 90 veces más grande que la de LunarLander y un paso grande mueve demasiados pesos a la vez |
| γ | 0,99 | horizonte de ~100 pasos, suficiente para ligar la jugada con el punto |
| tamaño de lote | 32 | el de Mnih et al. (2015) |
| capacidad del buffer | 100.000 | 700 MB con el buffer por cuadros; entre 40 y 100 partidos de historia según lo que duren |
| pasos antes de aprender | 10.000 | que el primer lote no salga de un buffer casi vacío |
| gradientes por paso | 1 | más eficiente en muestras que 1 cada 4 cuando el presupuesto es de 1 millón de pasos y no de 50 |
| sincronización del objetivo | cada 1.000 pasos | ver sección 3.2d |
| ε | 1,0 → 0,01 lineal en 250.000 pasos | un cuarto del presupuesto explorando |
| pérdida | Huber | acota el gradiente cuando el error es grande, que es el efecto del recorte de error de Mnih et al. |
| recorte de gradiente | norma 10 | segunda barrera contra actualizaciones grandes |
| presupuesto | 1.000.000 de pasos (4 millones de cuadros) | ver sección 5 |

---

## 5. Resultados

`PENDIENTE`: curva de entrenamiento, tabla de evaluaciones por punto de control,
evaluación final con 30 partidos con y sin acciones pegajosas, GIF de una partida.

---

## 6. Reflexión sobre los resultados

`PENDIENTE`.

---

## 7. Reflexión sobre lo que más costó

**Cambiar de ambiente a mitad del taller.** El trabajo empezó en LunarLander. En la
clase del 28 de septiembre ese mismo ambiente fue el ejemplo resuelto en vivo, lo
que lo sacaba del requisito de "ambiente no trabajado en clase". Lo aprendido allí
(sección 8) se trasladó, pero la red, el preprocesamiento y el buffer hubo que
hacerlos de nuevo.

**Diseñar el buffer por cuadros sin equivocarse.** Es la parte del código con más
formas de fallar en silencio. Un error de índices no lanza excepciones: produce
pilas que mezclan el último cuadro de un partido con el primero del siguiente, o que
leen un cuadro recién sobrescrito, y el agente simplemente aprende peor. La única
defensa fue una prueba que reconstruye cientos de pilas y las compara, píxel a píxel,
con las que entregó el entorno.

`PENDIENTE`: dificultades del entrenamiento largo.

---

## 8. Antes de Pong: lo que se aprendió en LunarLander

En el Taller 1 ([leonarsomo/mountain_car](https://github.com/leonarsomo/mountain_car))
vimos que entrenar de más degradaba la tabla de Q-Learning en vez de afinarla, y por eso
separamos allí la evaluación final del entrenamiento. Antes de cambiar de ambiente en este
taller entrenamos DQN y Double DQN en LunarLander-v3, y tres resultados de ese trabajo
definieron cómo evaluamos aquí:

1. **La recompensa de entrenamiento no sirve como criterio de parada.** El mejor
   modelo estuvo en el episodio 600 y no en el 800; con la curva de entrenamiento
   como guía se habría entregado el peor.
2. **Diez episodios de evaluación no bastan.** Con 10 episodios DQN parecía 38 puntos
   mejor que Double DQN; con 20 la diferencia desapareció.
3. **Dos corridas idénticas pueden separarse mucho.** Tres corridas de la misma
   configuración dieron 57, 229 y 149 a los 400 episodios.

Por eso aquí la evaluación es independiente del entrenamiento, con semillas fijas y
política congelada, y la evaluación final usa 30 partidos.

---

## 9. Cómo reproducir

```bash
uv sync
uv run pong-dqn inspeccionar     # espacios, dtype, recompensas y resumen de la red
uv run pong-dqn benchmark        # estima cuánto tardará el entrenamiento en esta máquina
uv run pytest -q                 # 7 pruebas, incluida la del buffer

# Entrenamiento (en macOS, caffeinate evita que el equipo se duerma)
caffeinate -i uv run pong-dqn entrenar --pasos 1000000 --etiqueta dqn
# Si se interrumpe, se retoma con:
caffeinate -i uv run pong-dqn entrenar --pasos 1000000 --etiqueta dqn --reanudar

# Evaluación final y gráficas
uv run pong-dqn evaluar --modelo saves/dqn/mejor.pt --n 30
uv run pong-dqn evaluar --modelo saves/dqn/mejor.pt --n 30 --sticky 0.25
uv run python scripts/graficar.py dqn
uv run python scripts/grabar_gif.py --modelo saves/dqn/mejor.pt
```

## 10. Estructura

```
src/pong_dqn/entorno.py   Pong con el preprocesamiento de Mnih et al. (2015)
src/pong_dqn/buffer.py    replay buffer que guarda cada cuadro una sola vez
src/pong_dqn/red.py       red convolucional de Nature DQN
src/pong_dqn/agente.py    ε-greedy, objetivo de Bellman (DQN y Double DQN), persistencia
src/pong_dqn/entrenar.py  bucle de entrenamiento, evaluación periódica, mejor modelo
src/pong_dqn/evaluar.py   evaluación con política congelada y semillas fijas
src/pong_dqn/cli.py       pong-dqn inspeccionar | entrenar | evaluar | benchmark
scripts/graficar.py       curva de entrenamiento y evaluaciones
scripts/grabar_gif.py     GIF de una partida junto a lo que ve la red
tests/                    buffer, red y agente
esquemas/                 diagrama del ciclo DQN dibujado a mano
```

## 11. Declaración de uso de IA

Como equipo, utilizamos asistentes de IA como apoyo para la estructuración y programación de la solución, la ejecución y automatización de pruebas, la elaboración de scripts de medición y gráficos, la organización del repositorio y la redacción de la documentación técnica. Revisamos el código, verificamos las métricas contra los registros en `resultados/` y asumimos la responsabilidad colectiva sobre el contenido y las conclusiones presentadas. `PENDIENTE`: frase sobre el esquema de la sección 3, solo si efectivamente se dibuja a mano.

## 12. Licencia

Apache 2.0 (ver `LICENSE`), igual que el Taller 1.

## Referencias

- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., y
  Bowling, M. (2018). Revisiting the Arcade Learning Environment: Evaluation
  protocols and open problems for general agents. *Journal of Artificial Intelligence
  Research, 61*, 523-562. https://doi.org/10.1613/jair.5699
- Lapan, M. (2020). *Deep reinforcement learning hands-on* (2.ª ed.). Packt.
- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., y
  Riedmiller, M. (2013). *Playing Atari with deep reinforcement learning* [Preprint].
  arXiv. https://doi.org/10.48550/arXiv.1312.5602
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G.,
  Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie,
  C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., y
  Hassabis, D. (2015). Human-level control through deep reinforcement learning.
  *Nature, 518*(7540), 529-533. https://doi.org/10.1038/nature14236
- Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T.,
  Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A.,
  Schulhoff, S., Tai, J. J., Tan, H., y Younis, O. G. (2024). *Gymnasium: A standard
  interface for reinforcement learning environments* [Preprint]. arXiv.
  https://doi.org/10.48550/arXiv.2407.17032
