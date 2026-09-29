.PHONY: sync prueba lint inspeccionar benchmark entrenar evaluar grafica

sync:
	uv sync

prueba:
	uv run pytest -q

lint:
	uv run ruff check .

inspeccionar:
	uv run pong-dqn inspeccionar

benchmark:
	uv run pong-dqn benchmark

# caffeinate evita que el Mac se duerma durante el entrenamiento
entrenar:
	caffeinate -i uv run pong-dqn entrenar --pasos 1000000 --etiqueta dqn

evaluar:
	uv run pong-dqn evaluar --modelo saves/dqn/mejor.pt --n 30 --salida resultados/dqn/final_sticky0.json
	uv run pong-dqn evaluar --modelo saves/dqn/mejor.pt --n 30 --sticky 0.25 --salida resultados/dqn/final_sticky025.json

grafica:
	uv run python scripts/graficar.py dqn
