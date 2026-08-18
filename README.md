# NHH-Schedule-Free-Autonomous-Boats-in-Bergen

Tesis de máster (NHH, Bergen): simulación + optimización de un servicio de barcos pequeños **a demanda** para Bergen, Noruega. El servicio no existe todavía — este repo construye la lógica de cómo operaría.

## Estructura

```
.
├── docs/                 # Contexto y decisiones de la tesis — LEER PRIMERO (docs/CLAUDE.md)
├── papers/               # Papers de referencia (Gu & Wallace 2021, Braathen/Goez/Guajardo 2024, etc.)
├── bergen-boats/         # Instancia base: nodos, matriz de tiempos, ruteo navegable (pasos 01-02)
└── demand/               # Generación de demanda sintética anclada en datos abiertos SSB (en curso)
```

Cada subcarpeta numerada dentro de `bergen-boats/` y `demand/` es autocontenida: notebook + `README.md` propio + `output/` con lo que produce. Ver el `README.md` de cada una para el detalle.

## Por dónde empezar

1. `docs/CLAUDE.md` — contexto completo del proyecto, principios, decisiones con Julio Goez y Stein W. Wallace.
2. `bergen-boats/README.md` — instancia base (nodos, tiempos de viaje navegables).
3. `demand/README.md` — generación de demanda sintética (en curso).
