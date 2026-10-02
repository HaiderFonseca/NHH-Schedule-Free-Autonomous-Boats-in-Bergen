# Step 1 - Time and distance matrix

First block of the base instance: how long a boat takes between each pair of nodes.

## What it does

1. Loads the 4 demand nodes (Kleppestø, Laksevåg, Bryggen, Sandviken) from [`../config/instance.yaml`](../config/instance.yaml).
2. Computes the straight-line distance over water between each pair, with the **Haversine** formula.
3. Calibrates an **effective speed** with the only real data available: route 490 (Askøybåten) between Kleppestø and Bryggen measures 5.36 km and the real ferry takes 14 min door to door → **≈ 23 km/h (12.4 knots)**. That speed already includes maneuvering and docking, so it is used as-is for all pairs.
4. Converts the distance matrix into a **travel time** matrix (min) using that speed.
5. Saves the matrices as CSV and generates the plots.
6. **Visual verification**: draws the 6 straight lines (one per pair of nodes) on the real map, zoomed in tightly on Bryggen, to check whether any of them cross land.

## Why Hegreneset does not appear in the matrix

Hegreneset is an intermediate point between Sandviken and Bryggen, not a stop: nobody boards or disembarks there. The notebook loads it separately (`waypoints` section of the config) only to show it on the map as a geometric reference, and produces a separate reference matrix (`output/*_con_waypoints_REFERENCIA.csv`) that is **not** used in the rest of the project - the real demand matrix is the 4x4 one.

## How to run

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

or open `notebook.ipynb` in Jupyter/VS Code and run all cells.

## Outputs (`output/`)

| File | What it is |
|---|---|
| `matriz_distancias_km.csv` | Haversine distances, 4x4, demand nodes |
| `matriz_tiempos_min.csv` | Travel times, 4x4, demand nodes - **this is the one used by the following steps** |
| `velocidad_calibrada_kmh.txt` | Calibrated effective speed (≈ 22.97 km/h) |
| `mapa_nodos_bergen.png` | Map of Bergen with the 4 demand nodes + Hegreneset marked as a waypoint |
| `heatmap_tiempos.png` | Heatmap of the time matrix |
| `matriz_*_con_waypoints_REFERENCIA.csv` | 5x5 reference matrix including Hegreneset - for future reference only, not used in demand |
| `mapa_lineas_rectas.png` | The 6 straight lines between nodes, with distance in km, over the map |
| `mapa_lineas_rectas_detalle.png` | The same with more street/coastline context (medium zoom) |
| `mapa_zoom_bryggen.png` | Tight zoom on Bryggen, to see the crossing with the Nordnes peninsula |

## Warning: finding - the straight lines touching Bryggen cross land

Visual inspection (section 9 of the notebook) shows that Haversine underestimates the real distance on Bryggen's connections:

- **Bryggen-Sandviken**: the straight line cuts through the center of Bergen instead of exiting through the mouth of Vågen. It clearly crosses land.
- **Bryggen-Kleppestø** (the very segment used for calibration) **and Bryggen-Laksevåg**: pass very close to (probably over) the tip of the Nordnes peninsula.
- **Laksevåg-Sandviken**: same problem with Nordnes.
- **Kleppestø-Laksevåg and Kleppestø-Sandviken**: over open fjord, no problem.

Since Bryggen concentrates the model's 3 "strong connections" (hard 15-minute guarantee), this is relevant: the times to/from Bryggen could be underestimated. This is documented in `../parametros_instancia_base_bergen.md` as pending a decision - it was not corrected automatically because there are several reasonable ways to do so (manual deviation factor, route via waypoints, waiting for real AIS data) and it is a modeling decision.

## Result

Time matrix (min):

| from \ to | Kleppestø | Laksevåg | Bryggen | Sandviken |
|---|---|---|---|---|
| **Kleppestø** | - | 9.3 | 14.0 | 13.5 |
| **Laksevåg** | 9.3 | - | 5.0 | 6.2 |
| **Bryggen** | 14.0 | 5.0 | - | 3.6 |
| **Sandviken** | 13.5 | 6.2 | 3.6 | - |

Matches what is documented in `../parametros_instancia_base_bergen.md` (sanity test included in the notebook: Kleppestø-Bryggen must give exactly 14.0 min).

## Next step

`../02_ruteo_navegable/` - the visual inspection above showed that the straight lines touching Bryggen cross land; that step corrects this with real routes over a water mesh. **The steps after that one (`03_demanda/` onward) use the corrected matrix, not the one here.**
