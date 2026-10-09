# Registro de cambios

## Aporte de Leonar Socarrás Molina (rama `aporte-leonar-robustez`)

- Curva de robustez de los dos mejores modelos frente a acciones pegajosas (0 a 0,5; 30
  partidos por nivel), con intervalos bootstrap y comparación entre semillas
  (`scripts/robustez.py`, `resultados/robustez_sticky/`, sección 5.5 del README).
- Opción `--sticky` en `pong-dqn entrenar` para entrenar con el protocolo de Machado et al.
  (2018); la evaluación periódica usa el mismo valor. Prueba nueva `tests/test_entorno.py`.
- Referencias: se citan en el texto Mnih et al. (2013) y Towers et al. (2024), que estaban
  en la lista sin cita, y se corrige el orden alfabético (Machado).

## Aporte de Bryan Johann Aranzazu Medina (rama `aporte-bryan-analisis-q`)

- `scripts/analisis_q.py`: reproduce con código las mediciones de la sección 6 que no
  tenían script (ritmo de anotación de 77-78 pasos, diferencia de Q entre acciones
  redundantes y sobreestimación de Q frente al retorno descontado real).
- Resultados en `resultados/analisis_q/` para los dos mejores modelos, con y sin acciones
  pegajosas (0,25).
- Hallazgo nuevo en la sección 6: con acciones pegajosas el Q esperado casi no cambia
  pero el retorno real cae a 0,40 y 0,18; la red conserva los valores del emulador
  determinista.
- Documento del aporte en `docs/Aporte_pong_dqn_AnalisisQ.pdf` (método, resultados,
  discusión y referencias en APA)

## Sin publicar

Taller 2, Unidad 3.

- Pong (`ALE/Pong-v5`) con el preprocesamiento de Mnih et al. (2015): salto de 4 cuadros con
  máximo de los dos últimos, escala de grises, 84 × 84, hasta 30 no-ops, apilado de 4 cuadros.
- Replay buffer que guarda cada cuadro una sola vez (700 MB en lugar de 5,6 GB para 100.000
  transiciones), con prueba de reconstrucción exacta.
- Red convolucional de Nature DQN y Double DQN opcional.
- Entrenamiento con evaluación periódica sobre semillas fijas y guardado del mejor modelo.
- Línea de comandos `pong-dqn`: inspeccionar, entrenar, evaluar, benchmark.
