# Step 2 - Navigable routing (corrects the straight lines that cross land)

Step 1 (`../01_tiempos_distancias/`) used Haversine: straight-line distance. When reviewed visually we found that the lines touching **Bryggen** cross land - Bryggen sits inside Vågen bay, next to the **Nordnes peninsula**. This step corrects that with a **routing module over water**: it builds a real navigable mesh and computes the shortest path that does not pass over land.

## How it works, step by step

1. **Download a real map** of the area (CartoDB Positron with no labels - no labels so that name text is not mistakenly classified as land).
2. **Classify water/land by pixel color.**
3. **Build a graph**: each water pixel is a node connected to its 8 neighbors.
4. **Snap** each port to the nearest water pixel.
5. **Dijkstra** from each port to the others → real navigable distance and the exact path.
6. **Apply the fixed design speed** (30 km/h, see below) to convert distance to time.

All the reusable logic is in [`../src/water_routing.py`](../src/water_routing.py), documented for use in future steps (e.g. if a real boat needs to be routed during the simulation, not just a matrix computed).

---

## How the water/land mask is built, in detail

**The map is a mosaic of XYZ tiles** (the same format used by Google Maps and OpenStreetMap): for each zoom level, the world is divided into a grid of 256x256 pixel tiles, and each level doubles the resolution of the previous one. With `ZOOM=14` that gives a resolution that is **fixed across the entire planet** of:

```
mercator_meters_per_pixel = 156543.03392 / 2^zoom = 156543.03392 / 16384 ≈ 9.55 m/pixel
```

That value is in **projected** meters (Web Mercator, EPSG:3857), which are not real meters: Mercator stretches distances by a factor of `1/cos(latitude)` in order to represent the spherical Earth on a plane (which is why Greenland looks gigantic on a world map). Bergen is at ≈60.4°N, where `cos(60.4°) ≈ 0.494` - so one projected meter there equals only ~0.494 real meters. The real resolution on the ground is:

```
real_meters_per_pixel = 9.55 × cos(60.4°) ≈ 4.7 m/pixel
```

In other words: **each "cell" of the mesh is a square of ~4.7 x 4.7 m of real water or land.** The notebook computes this number explicitly (section 2) rather than leaving it fixed, because it depends on the exact latitude of the instance.

**The graph is built directly on these pixels - there is no separately chosen grid spacing.** A node of the graph *is* a water pixel, so the spacing between two neighboring nodes is exactly that resolution: **~4.72 m** for an orthogonal neighbor (up/down/left/right) and **~6.67 m** (`4.72 × √2`) for a diagonal neighbor, with `ZOOM=14`. The notebook prints this explicitly in section 3, when building the graph. Raising the `ZOOM` gives a finer mesh (more nodes, more precision, slower); lowering it gives a coarser mesh.

**Classification by color:** a test tile of Bergen was downloaded and the most frequent pixel colors were counted. Water in CartoDB Positron turned out to have a very consistent color - `COLOR_AGUA_POSITRON = (212, 218, 220)` (light blue-gray) - which appeared in ~40% of the pixels of the test area, consistent with being the color of the fjord. Each pixel is classified as water if its Euclidean distance in RGB space to that reference color is less than a threshold (`umbral_color=15`); otherwise, it is land.

**Why the mask hugs the real coastline so closely** (see `output/mapa_mascara_agua.png`, where it is overlaid in semi-transparent blue on the real map): because it is *literally* the same raster that draws the map. The pixel↔lon/lat conversion (`wr.pixel_a_lonlat` / `wr.lonlat_a_pixel`) uses the same spherical Web Mercator formula, with the same radius (6378137 m), that the tile server used to render the image. When the classified mask is overlaid back on the original map, it matches pixel for pixel by construction - the accuracy of the coastline depends on how good OpenStreetMap's geometry is (CartoDB's data source), not on any approximation of ours.

*(A technical note that cost us a bug: Web Mercator uses the WGS84 equatorial radius, 6378137 m, not the mean Earth radius of 6371009 m used in Haversine for real distances. Using the wrong radius in the projection formula misaligns the entire grid - this happened to us in the first version and it ended up ~150 km off from where it should have been.)*

## How Dijkstra works here, in detail

1. **The graph**: each water pixel (~1 million in the bounding box of this instance) is a node, connected to its 8 neighbors (up, down, left, right and the 4 diagonals) *only if that neighbor is also water*. If a neighbor is land, that edge does not exist - so the graph **has no way to offer a jump that crosses land**, not even by accident. The weight of each edge is the real distance in km between the centers of the two pixels (with the `cos(lat)` correction applied row by row).
2. **Snap**: the real coordinates of a port almost never fall exactly on a water pixel (they can fall on the edge of a dock). `wr.snap_a_grafo()` searches for the nearest water pixel in an expanding spiral, **restricted to the graph's main connected component** (computed with `scipy.sparse.csgraph.connected_components`). This prevents snapping a port to an isolated puddle or pond that was classified as water by color but is not connected to the open sea - this happened to us with Hegreneset in an early version, before adding this restriction.
3. **Multi-source Dijkstra**: `scipy.sparse.csgraph.dijkstra(grafo, indices=[...], return_predecessors=True)` runs Dijkstra's algorithm once for each source node/port, all in a single call. Since all weights are real distances (always positive), Dijkstra guarantees finding the path of **minimum total distance** from each source to all other nodes in the graph. No A* or any heuristic is needed: with ~1 million nodes and ~8 million directed edges, plain Dijkstra (implemented in Cython inside scipy) runs in ~1 second for all 5 ports at once.
4. **Path reconstruction**: Dijkstra also returns `predecessors`, the previous node in the shortest path to each destination. `wr.reconstruir_ruta_latlon()` walks that chain backward (destination → origin), gathers the visited pixels and converts them back to lon/lat - this is how the real route is drawn on the maps, not just a reported number.

**Known limitation:** by allowing only 8 directions (not 360°), the shortest path can make a slight "zigzag" instead of a perfect diagonal when the real direction does not match any of the 8 allowed ones (visible as a small kink in some routes on the maps). The extra cost of this is small (a few percent at worst) and does not affect the underlying conclusion: it avoids land with certainty.

---

## Nodes in this version

Laksevåg and Sandviken were corrected to confirmed locations (the previous ones were approximations):

| Node | Before (approx.) | Now (confirmed) |
|---|---|---|
| Laksevåg | 60.3945, 5.2875 | **60.390886, 5.259586** (Gravdal) |
| Sandviken | 60.4075, 5.3214 | **60.421149, 5.300502** (BSI Padling) |

Kleppestø and Bryggen did not change.

## Speed: fixed design decision (30 km/h)

Instead of calibrating with the real Kleppestø-Bryggen segment (as in an earlier version of this step, which gave ≈24.65 km/h), the team decided to use a **fixed design speed for the entire fleet: 30 km/h (~16.2 knots)** - `config/instance.yaml` → `calibracion.velocidad_forzada_kmh`. This is not the speed of an existing ferry; it is a design assumption for the small on-demand boats, which can be swept again later as a sensitivity parameter.

## Result: the 6 routes (all pairwise combinations among the 4 nodes)

`output/comparacion_recta_vs_navegable.csv` - straight line (step 1) vs. real navigable route (this step), both at 30 km/h:

| Connection | km straight | km navigable | difference | min straight | min navigable |
|---|---|---|---|---|---|
| Laksevåg ↔ Bryggen | 3.48 | 4.31 | **+23.8%** | 7.0 | 8.6 |
| Kleppestø ↔ Sandviken | 4.33 | 4.78 | +10.5% | 8.7 | 9.6 |
| Bryggen ↔ Sandviken | 3.13 | 3.41 | +8.9% | 6.3 | 6.8 |
| Kleppestø ↔ Laksevåg | 2.47 | 2.67 | +7.8% | 4.9 | 5.3 |
| Kleppestø ↔ Bryggen | 5.36 | 5.75 | +7.3% | 10.7 | 11.5 |
| Laksevåg ↔ Sandviken | 4.05 | 4.31 | +6.4% | 8.1 | 8.6 |

6 pairs = C(4,2), all possible combinations among the 4 demand nodes - all of them are included. With the new coordinates, Laksevåg (now at Gravdal, further inside Puddefjorden) is the one that lengthens the most when routed over real water, because the detour around Nordnes weighs more relative to its total distance.

## Warning: from here on, use this matrix

`output/matriz_tiempos_min.csv` from this step **replaces** the one from `01_tiempos_distancias/` for everything that follows (`03_demanda/` onward).

## Outputs (`output/`)

| File | What it is |
|---|---|
| `matriz_distancias_km.csv` / `matriz_tiempos_min.csv` | Navigable 4x4 matrices at 30 km/h - **the canonical ones from here on** |
| `matriz_*_con_waypoints_REFERENCIA.csv` | Same but 5x5 including Hegreneset |
| `comparacion_recta_vs_navegable.csv` | Table of differences per pair (the 6 combinations) |
| `velocidad_usada_kmh.txt` | 30.0 (fixed, design decision) |
| `mapa_mascara_agua.png` | Validation: water/land mask overlaid on the real map |
| `mapa_rutas_navegables.png` | The 6 real routes between the 4 demand nodes |
| `mapa_zoom_bryggen_comparacion.png` | Bryggen: straight line (red) vs. real route (green) - the key comparison |
| `heatmap_tiempos_navegables.png` | Heatmap of the time matrix at 30 km/h |

## How to run

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Needs an internet connection (downloads map tiles the first time). Takes ~15-20 s in total.

## Next step

`../03_demanda/` - generate the Poisson requests per time slot, using `output/matriz_tiempos_min.csv` from this step.
