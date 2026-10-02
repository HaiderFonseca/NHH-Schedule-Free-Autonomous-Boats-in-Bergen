# bergen-boats

Implementation of the thesis: simulation + optimization of a small **on-demand** boat service for Bergen, Norway. The service does not exist yet - this repo builds the logic of how it would operate.

The full context of the project (motivation, design decisions) lives in [`../docs/`](../docs/), and the final report in [`../docs/informe/`](../docs/informe/).

## How it is organized

Each step of the project lives in its own self-contained, numbered folder: a notebook that can be run from start to finish, a `README.md` explaining what it does and why, and an `output/` folder with what it produces (CSV, plots). This way each piece can be understood and reproduced without having to hold the rest of the project in your head.

```
bergen-boats/
├── requirements.txt
├── config/
│   └── instance.yaml              # single source of truth: nodes, speed, fleet, demand, guarantee
├── src/
│   ├── geo.py                     # shared functions: Haversine, matrices, calibration
│   └── water_routing.py           # routing over water: navigable mesh, Dijkstra, avoids crossing land
├── 01_tiempos_distancias/         # STEP 1 - straight-line distance/time matrix (Haversine)
│   ├── README.md
│   ├── notebook.ipynb
│   └── output/
├── 02_ruteo_navegable/            # STEP 2 - corrects step 1: real routes that do not cross land
│   ├── README.md
│   ├── notebook.ipynb
│   └── output/
├── 03_demanda/                    # STEP 3 (next) - generation of Poisson requests per time slot
├── 04_simulacion_despacho/        # STEP 4 (future) - rolling-horizon + dispatch policy
└── ...
```

The parameters that can change (coordinates, speed, fleet size, waiting guarantee) are all in `config/instance.yaml`, never hardcoded inside a notebook.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`contextily` (the maps with a real Bergen background) needs an internet connection to download the OpenStreetMap/CartoDB tiles the first time.

## How to run a step

Each numbered folder is independent: open its `notebook.ipynb` in Jupyter/VS Code and run all cells in order. It can also be run without opening anything:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_tiempos_distancias/notebook.ipynb
```

## The 4 demand nodes

Kleppestø, Laksevåg (Gravdal), Bryggen and Sandviken (BSI Padling) are the real stops (where people board/disembark). **Hegreneset is not a stop** - it is a reference point with no demand of its own, kept only as a *waypoint* for routing. See `config/instance.yaml` (`waypoints` section) and `parametros_instancia_base_bergen.md`.

## Status

- [x] **01 - Times and distances**: Haversine matrix (straight line). When reviewed visually, the lines touching Bryggen turned out to cross land (Nordnes peninsula) - see step 2.
- [x] **02 - Navigable routing**: corrects step 1 with a routing module over a real water mesh (~4.7 m/pixel in Bergen, Dijkstra, without crossing land). Fixed design speed: **30 km/h** (a decision, not calibrated). **`02_ruteo_navegable/output/matriz_tiempos_min.csv` is the matrix to use from here on**, not the one from step 1.
- [ ] **03 - Demand**: Poisson request generator per time slot.
- [ ] **04 - Simulation + dispatch**: rolling-horizon every 3 min, boat assignment policy.
