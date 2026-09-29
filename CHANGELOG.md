# Registro de cambios

## Sin publicar

Taller 2, Unidad 3.

- Pong (`ALE/Pong-v5`) con el preprocesamiento de Mnih et al. (2015): salto de 4 cuadros con
  máximo de los dos últimos, escala de grises, 84 × 84, hasta 30 no-ops, apilado de 4 cuadros.
- Replay buffer que guarda cada cuadro una sola vez (700 MB en lugar de 5,6 GB para 100.000
  transiciones), con prueba de reconstrucción exacta.
- Red convolucional de Nature DQN y Double DQN opcional.
- Entrenamiento con evaluación periódica sobre semillas fijas y guardado del mejor modelo.
- Línea de comandos `pong-dqn`: inspeccionar, entrenar, evaluar, benchmark.
