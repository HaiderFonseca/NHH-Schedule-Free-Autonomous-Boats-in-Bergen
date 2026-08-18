# bergen-boats

Implementación de la tesis: simulación + optimización de un servicio de barcos pequeños **a demanda** para Bergen, Noruega. El servicio no existe todavía — este repo construye la lógica de cómo operaría.

El contexto completo del proyecto (motivación, decisiones de diseño, respuestas de Julio/Stein) vive en [`../docs/`](../docs/) — léelo antes de tocar código, empezando por `../docs/CLAUDE.md`.

## Cómo está organizado

Cada paso del proyecto vive en su propia carpeta numerada, autocontenida: un notebook que se puede correr de principio a fin, un `README.md` que explica qué hace y por qué, y una carpeta `output/` con lo que produce (CSV, gráficas). Así se puede entender y reproducir cada pieza sin tener que cargar el resto del proyecto en la cabeza.

```
bergen-boats/
├── requirements.txt
├── config/
│   └── instance.yaml              # única fuente de verdad: nodos, velocidad, flota, demanda, garantía
├── src/
│   ├── geo.py                     # funciones compartidas: Haversine, matrices, calibración
│   └── water_routing.py           # ruteo sobre agua: malla navegable, Dijkstra, evita cruzar tierra
├── 01_tiempos_distancias/         # PASO 1 — matriz de distancias/tiempos en línea recta (Haversine)
│   ├── README.md
│   ├── notebook.ipynb
│   └── output/
├── 02_ruteo_navegable/            # PASO 2 — corrige el paso 1: rutas reales que no cruzan tierra
│   ├── README.md
│   ├── notebook.ipynb
│   └── output/
├── 03_demanda/                    # PASO 3 (próximo) — generación de solicitudes Poisson por franja
├── 04_simulacion_despacho/        # PASO 4 (futuro) — rolling-horizon + política de despacho
└── ...
```

Los parámetros que pueden cambiar (coordenadas, velocidad, tamaño de flota, garantía de espera) están todos en `config/instance.yaml`, nunca hardcodeados dentro de un notebook.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`contextily` (los mapas con fondo real de Bergen) necesita conexión a internet para bajar los tiles de OpenStreetMap/CartoDB la primera vez.

## Cómo correr un paso

Cada carpeta numerada es independiente: abre su `notebook.ipynb` en Jupyter/VS Code y corre todas las celdas en orden. También se puede ejecutar sin abrir nada:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_tiempos_distancias/notebook.ipynb
```

## Los 4 nodos de demanda

Kleppestø, Laksevåg (Gravdal), Bryggen y Sandviken (BSI Padling) son las paradas reales (donde la gente sube/baja). **Hegreneset no es una parada** — es un punto de referencia sin demanda propia que se guarda solo como *waypoint* para el ruteo. Ver `config/instance.yaml` (sección `waypoints`) y `../docs/parametros_instancia_base_bergen.md`.

## Estado

- [x] **01 — Tiempos y distancias**: matriz Haversine (línea recta). Al revisarla visualmente, las líneas que tocan Bryggen resultaron cruzar tierra (península de Nordnes) — ver paso 2.
- [x] **02 — Ruteo navegable**: corrige el paso 1 con un módulo de ruteo sobre una malla de agua real (~4.7 m/píxel en Bergen, Dijkstra, sin cruzar tierra). Velocidad de diseño fija: **30 km/h** (decisión, no calibrada). **`02_ruteo_navegable/output/matriz_tiempos_min.csv` es la matriz a usar de aquí en adelante**, no la del paso 1.
- [ ] **03 — Demanda**: generador de solicitudes Poisson por franja horaria.
- [ ] **04 — Simulación + despacho**: rolling-horizon cada 3 min, política de asignación de barcos.
