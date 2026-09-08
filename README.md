# NHH-Schedule-Free-Autonomous-Boats-in-Bergen

Tesis de máster (NHH, Bergen): simulación + optimización de un servicio de barcos pequeños **a demanda** para Bergen, Noruega. El servicio no existe todavía — este repo construye la lógica de cómo operaría.

## Estructura

```
.
├── docs/                 # Contexto y decisiones de la tesis — LEER PRIMERO (docs/CLAUDE.md); informe final en docs/informe/
├── papers/               # Papers de referencia (Gu & Wallace 2021, Braathen/Goez/Guajardo 2024, etc.)
├── bergen-boats/         # Instancia base: nodos, matriz de tiempos, ruteo navegable (pasos 01-02)
├── demand/               # Generación de demanda sintética anclada en datos abiertos SSB
├── simulador/            # El motor: entorno Gymnasium, estado, recompensa, métricas, visualización
├── politica_base/        # Política de referencia ("nearest-available") + verificación del simulador
├── modelo_rl/            # Agente PPO: entrenamiento, experimentos de recompensa, corrida final
└── comparacion/          # Agente PPO vs. política base, mismas semillas de evaluación
```

Cada subcarpeta es autocontenida: notebook(s) + `README.md` propio + `output/` con lo que produce. `simulador/`, `politica_base/`, `modelo_rl/` y `comparacion/` son 4 bloques con responsabilidad única sobre un mismo motor de simulación — ver `simulador/README.md` sección 1 para cómo se conectan entre sí y `simulador/config/instance.yaml` (única fuente de verdad de los parámetros que comparten los 4). Ver el `README.md` de cada carpeta para el detalle.

## Por dónde empezar

1. `docs/CLAUDE.md` — contexto completo del proyecto, principios, decisiones con Julio Goez y Stein W. Wallace.
2. `bergen-boats/README.md` — instancia base (nodos, tiempos de viaje navegables).
3. `demand/README.md` — generación de demanda sintética.
4. `simulador/README.md` — el motor de simulación (MDP: estado, acciones, recompensa).
5. `politica_base/README.md` — política de referencia y verificación del simulador (escalones 1-3).
6. `modelo_rl/README.md` — agente PPO (entrenamiento, experimentos de recompensa).
7. `comparacion/README.md` — agente vs. política base.
8. `docs/informe/` — informe final (LaTeX/PDF) con el resultado completo del proyecto.
