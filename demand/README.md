# Synthetic demand anchored in open data (Bergen)

Generates minute-by-minute arrivals of passenger groups for the 4 demand nodes (Kleppestø, Laksevåg, Bryggen, Sandviken), anchored in real SSB population and employment data instead of eyeballed assumptions. It later feeds the simulation and an RL agent.

**Guiding principle: full traceability.** Every number in this README can be traced back to its origin: a file in `output/`, a notebook cell, or an explicit citation. Where something was assumed, it is declared as an assumption (final section), not hidden in the code.

## Status: step closed

| Task | Status | What changed in the last iteration |
|---|---|---|
| 0 - Inspection and clipping | ✅ | - |
| 1 - Masses per node | ✅ | Grunnkretser touching each hand-drawn area, "touches→full inclusion" criterion |
| 2 - Employment (destination) | ✅ | `emp_tot` from the SSB grid, without needing StatBank; `factor_universitario` set by the team |
| 3 - O-D pattern by time band | ✅ | Recalculated with current masses + university factor (morning direction: 85.6/14.4) |
| 4 - Minute-by-minute arrivals | ✅ | Scale as % of real population (10%, not a made-up number); Binomial group size (not Normal) |

The 4 nodes and their final population/employment (Task 1 v2 + Task 2, see detail below):

| Node | Population | Employment |
|---|---|---|
| Kleppestø (Askøy) | 25 153 | 7 550 |
| Laksevåg (Gravdal) | 15 986 | 7 760 |
| Bryggen / Sentrum | 27 695 | 52 213 |
| Sandviken (BSI Padling) | 11 109 | 4 106 |
| **Total** | **79 943** | **71 629** |

## How it is organized

```
demand/
├── README.md                     # this file - single document for Tasks 0-4
├── config/instance.yaml          # single source of truth for the parameters
├── data/
│   ├── *.geojson                 # raw SSB grids (gitignored, see "Data provenance")
│   ├── Areas_estudio.gpkg        # 4 hand-drawn areas (QGIS), one per node
│   └── processed/                 # cached clips and grunnkretser (GeoParquet)
├── src/
│   ├── masas.py                  # Task 0: geographic clipping
│   ├── grunnkretser.py           # Task 1: Kartverket grunnkretser, selection by area
│   ├── patron_od.py              # Task 3: O-D intensity by time band
│   ├── llegadas.py               # Task 4: Poisson process + groups
│   └── plotting.py               # verification maps
├── notebooks/
│   ├── 00_recorte_y_masas.ipynb  # Task 0: geographic clipping (initial exploration, not used for masses)
│   ├── 00b_grunnkretser.ipynb    # exploration: downloads and maps the candidate grunnkretser
│   ├── 00c_masas_por_grunnkretser.ipynb  # Task 1 (canonical): masses with real areas
│   ├── 01_empleo.ipynb           # Task 2
│   ├── 02_patron_od.ipynb        # Task 3
│   └── 03_llegadas.ipynb         # Task 4
└── output/                       # CSV, maps and charts - the evidence for this README
```

## Data provenance

| File | What it is | Provenance |
|---|---|---|
| `2026-08-18-befolkning_250m_2026.geojson` | Population in a 250 m grid, all of Norway | SSB - **[PENDING: confirm exact dataset URL/name with the user]** |
| `2026-08-18-bedrifter_250m_2026.geojson` | Establishments and employment in a 250 m grid, all of Norway | SSB - **[PENDING: confirm exact dataset URL/name with the user]** |
| Grunnkretser (`data/processed/grunnkretser_norge.parquet`, cached) | Boundaries of Norway's small statistical units | Kartverket, WFS "Statistiske enheter grunnkretser" - [kartkatalog.geonorge.no, uuid `cc7ded0b-7d34-4db6-8fdb-c5a7682b6836`](https://kartkatalog.geonorge.no/metadata/statistiske-enheter-grunnkretser/cc7ded0b-7d34-4db6-8fdb-c5a7682b6836), `Grunnkrets` layer, CC BY 4.0 license |
| `Areas_estudio.gpkg` | 4 hand-drawn polygons (QGIS) delimiting each node's catchment area | Produced in-house by the team, with no attributes - the node for each area is inferred by proximity (see Task 1) |

The exact citation for the 2 SSB GeoJSON files could not be verified beyond what the file itself allows to inspect (see Task 0). The plausibility check (national population sum ≈ Norway's real population, see Task 0) gives confidence that these are genuine SSB data, but the formal citation remains pending. The grunnkretser source, on the other hand, was fully identified (official Kartverket WFS, exact URL above).

---

## Task 0 - Geographic inspection and clipping

**Notebook:** `notebooks/00_recorte_y_masas.ipynb`

### What was found in the raw files

| | Population (`befolkning_250m`) | Bedrifter (`bedrifter_250m`) |
|---|---|---|
| CRS | **EPSG:32633** (UTM 33N) - metric, correct for radii in meters | **EPSG:32633** |
| Columns | `ssbid250m` (id), **`pop_tot`** (cell population) | `SSBID250M` (id), `est_tot` (number of establishments), **`emp_tot`** (actual employment) |
| Cell resolution | 250 × 250 m (verified in code, not only in the file name) | 250 × 250 m |
| No. of features (all of Norway) | 225 238 | 133 688 |
| Coverage | All of Norway, includes Askøy (explicitly verified) | All of Norway |
| Plausibility check | National sum of `pop_tot` = **5 617 894** ≈ Norway's real population | Max. for a single cell = 10 628 jobs (plausible: one large, concentrated employer) |

**Important finding that changed the plan (Task 2):** `bedrifter_250m` includes `emp_tot` (actual employment), not just an establishment count as assumed during planning. See Task 2.

### Clipping bounding box

Calculated as the rectangle covering the 4 nodes (read from `bergen-boats/config/instance.yaml`) plus an 8000 m margin (`recorte.margen_bbox_m`):

```
EPSG:32633: xmin=-45041.1, ymin=6726620.1, xmax=-24044.2, ymax=6745676.7
Width: 21.00 km, Height: 19.06 km
```

### Cells after clipping

| | All of Norway | After clipping to Bergen+Askøy |
|---|---|---|
| Population | 225 238 cells | **2 557 cells** (260 168 inhabitants) |
| Bedrifter | 133 688 cells | **2 127 cells** (160 895 jobs) |

Clip cached in `data/processed/poblacion_bergen_askoy.parquet` and `bedrifter_bergen_askoy.parquet` - not repeated in future runs.

---

## Task 1 - Masses per node

**Notebooks:** `notebooks/00b_grunnkretser.ipynb` (exploration) → `notebooks/00c_masas_por_grunnkretser.ipynb` (final calculation) · **Output:** `output/masas_por_nodo.csv`, `output/grunnkretser_seleccionadas_por_nodo.csv`, `output/mapas_zonas_grunnkretser_por_nodo.png`, `output/mapa_general_zonas_grunnkretser.png`

**Source of the administrative boundaries:** grunnkretser (Norway's actual small statistical unit, ~403 in Bergen+Askøy), downloaded from the official Kartverket WFS (`wfs.grunnkretser`, `Grunnkrets` layer, see "Data provenance"). Confirmed by code: Bergen = kommunenummer 4601, Askøy = 4627 (verified by searching for "Kleppestø" in the grunnkrets names).

**Process:**
1. The team hand-drew 4 study areas in QGIS (`data/Areas_estudio.gpkg`, with no overlaps among them - verified, intersection = 0 km² across the 6 pairs).
2. Each area was assigned to the node whose center is closest - unambiguously: the distance to the correct node is 3.9x to 10.7x smaller than to the second-closest one (explicitly verified in the notebook, not merely assumed).
3. The grunnkretser that **touch** each area were selected (an "intersects" criterion, not "center inside": if the area grabs even a small piece of a grunnkrets, that grunnkrets is taken in **full**). Explicit decision by the user: the areas were drawn with this in mind, and under the previous criterion ("center inside") coastal zones of Sandviken that the area did touch were being left out.
4. **Duplicate resolution:** with "intersects", a grunnkrets right on the border between two areas can touch both. **1 case** was found ("Sandviksfjellet", 46010637, touched both Bryggen and Sandviken) and it was resolved by leaving it only in the node with the greater actual overlap (Sandviken, 18.4% of its area vs. 0.05% in Bryggen) - explicitly verified and reported in the notebook, not just silently fixed.
5. Population and employment were recalculated over the union of those grunnkretser, using the same 250 m SSB grid as always - the shape of the zone changes, not the source of the population/employment numbers.

**Selected grunnkretser and resulting masses** (`output/grunnkretser_seleccionadas_por_nodo.csv` has the full detail, number and name of each one):

| Node | Grunnkretser | Area (km²) | Total population | Total employment |
|---|---|---|---|---|
| Kleppestø (Askøy) | 28 | 80.59 | **25 153** | **7 550** |
| Laksevåg (Gravdal) | 18 | 14.85 | **15 986** | **7 760** |
| Bryggen / Sentrum | 67 | 20.21 | **27 695** | **52 213** |
| Sandviken (BSI Padling) | 21 | 14.61 | **11 109** | **4 106** |

Bryggen's employment is clearly the highest (~73% of the total across the 4 nodes), consistent with it being the city's real employment hub. Kleppestø is the node with the largest population (25 153 inhabitants), as it covers a large part of the municipality of Askøy.

**Verification maps:** `output/mapa_general_zonas_grunnkretser.png` (the 4 zones together, with the drawn area as a dotted line and the selected grunnkretser filled in) and `output/mapas_zonas_grunnkretser_por_nodo.png` (zoom per node) - confirm that no coastal gaps remain inside the drawn areas.

---

## Task 2 - Employment as destination mass

**Notebook:** `notebooks/01_empleo.ipynb` · **Resolved without needing the StatBank API.**

The original plan assumed employment would have to be pulled via API because `bedrifter_250m` would be "just an establishment count." When inspecting the file (Task 0), the `emp_tot` column appeared with actual employment. Before using it, the planned alternative was checked:

- `https://data.ssb.no/api/v0/no/table/al/al06/regsys/SBMENU4262` was explored → table **12850** ("Sysselsatte, etter bosted, arbeidssted, næring..."). Its `Region` variable only goes down to **fylke/kommune** (Vestland, Bergen, etc.), not to grunnkrets/delområde - SSB does not publish fine-grained employment by workplace via StatBank due to disaggregation privacy rules; instead it publishes it as a grid, which is exactly what was already available.
- **Conclusion:** `emp_tot` from the 250 m grid is the correct source, and in fact finer-grained than grunnkrets.

**Evidence that `emp_tot` is actual employment and not noise from `est_tot`:** in the study area (2127 cells), the `est_tot` vs. `emp_tot` correlation = **0.581** (positive but far from 1 - consistent with actual employment: one large office concentrates a lot of employment in few establishments). See `output/establecimientos_vs_empleo.png`.

Numbers updated with the grunnkretser zones from Task 1 (see above):

| Node | Establishments | Employment | Employees/establishment |
|---|---|---|---|
| Kleppestø | 1 884 | 7 550 | 4.0 |
| Laksevåg (Gravdal) | 1 447 | 7 760 | 5.4 |
| Bryggen / Sentrum | 7 101 | 52 213 | 7.4 |
| Sandviken (BSI Padling) | 879 | 4 106 | 4.7 |

`est_tot` is kept only as a comparative reference, it is not used as the main mass.

**On the university population (updated):** Bergen is a university city (NHH, UiB, HVL) and `emp_tot` does not capture students, who also generate trips. No open source of fine-grained geographic enrollment/student population data was found, so a hook was left in the config: `masas.factor_universitario`, which weights each node's **destination** mass (employment) in Task 3. The team set the values by hand: `kleppesto: 0.7`, `laksevag: 0.85`, `bryggen: 1.0`, `sandviken: 1.0` (Sandviken sits next to NHH/Kronstad/HVL, and in fact Gu & Wallace (2021) itself - the paper that originated these same 4 nodes - explicitly describes "Sandviken & Hegreneset" as a zone with "office buildings, apartments, houses **and a university**"). This is a design decision by the team, not a data source - explicitly declared in "Assumptions and limitations".

---

## Task 3 - Origin-destination pattern by time band

**Notebook:** `notebooks/02_patron_od.ipynb` · **Output:** `output/matriz_intensidad_od.csv`, `output/heatmaps_od_por_franja.png`, `output/inversion_direccion_bryggen.png`, `output/factor_distancia.png`

### Method: Braathen, Goez & Guajardo (2024) §4.1, adapted

The paper (`papers/Autonomous ferries in light of labor regulations-A passenger.pdf`) defines:

```
pop_pg := ln(1 + Π(i=1..4) p_i,pg)          p1..p4 = origin, destination, day, hour (station scores "at discretion")
tamaño_pg := ceil((pop_pg − min_j pop_j) / β)
```

with **mean 8.6 passengers/group, standard deviation 2.6, mode 9** (its Fig. 1-2, p. 8).

**What is borrowed and what is an original contribution - explicit:**

| Element | Origin |
|---|---|
| Multiplicative formula | Braathen et al. §4.1 |
| The 4 factors (origin, destination, day, hour) | Braathen et al. §4.1 |
| Station score = **the same** for origin and destination | Braathen et al. §4.1 - **we did not copy this as-is** |
| Station scores set "at discretion," with no published table | Braathen et al. - not reproducible by the paper's design |
| Replacing station scores with real SSB masses (Tasks 0-2) | **Original contribution** |
| Splitting the score by role - population if origin, employment if destination, with AM/PM inversion | **Original contribution**, not in the paper |
| Distance factor decreasing with travel time | **Original contribution** - the paper does not weight this way in §4.1 (it uses distance in a different problem, filtering "realistic routes") |
| Group size as a configurable distribution (vs. the paper's deterministic formula) | **Original contribution**, calibrated to match the *reported* mean/standard deviation |
| Time-band and day-of-week factors | Set with our own judgment, same as the paper (also not reproducible from the paper) |

### Mass normalization

Population and employment are normalized by dividing by the maximum across the 4 nodes (leaving them in `(0, 1]`) before being multiplied - otherwise, the raw scale (hundreds vs. thousands) would arbitrarily dominate the product. See the full table in the notebook.

### Configured time bands

| Time band | Hours | Volume factor | Origin role | Destination role |
|---|---|---|---|---|
| Morning | 6-9 | 1.0 | population | employment |
| Off-peak | 9-15 | 0.4 | mixed | mixed |
| Afternoon | 15-18 | 1.0 | employment | population |
| Night | 18-24 | 0.15 | mixed | mixed |

`factor_distancia = exp(-tiempo_viaje_min / tau_min)`, with `tau_min = 15` - shape and parameter chosen for reasonableness (see "Assumptions").

### Do the roles change between weekdays and weekends? No - only the volume

A question worth answering explicitly: **the role table above (morning=population→employment, afternoon=employment→population, off-peak/night=mixed) is the same all 7 days of the week.** `factor_dia_semana` (Task 4: `entre_semana=1.0`, `fin_de_semana=0.4`) is a **global** multiplier on the day's total passengers - it does not touch `matriz_intensidad_od.csv`, which is calculated once and does not distinguish day of the week. In other words: Saturday and Sunday have exactly the same directional pattern as a weekday, only at ~40% of the volume (plus Poisson's own noise).

Verified with the generated data (`grupos_semana.csv`, % of groups by origin node within each time band):

| Time band | Dominant origin on weekdays | Dominant origin on weekends |
|---|---|---|
| Morning | Kleppestø 39% | Kleppestø 38% |
| Afternoon | Bryggen 73% | Bryggen 73% |
| Night | Bryggen 36% | Bryggen 44% |

Practically identical between weekdays and weekends (the "night" band varies a bit more because it has fewer total groups, making it more sensitive to Poisson noise). This is a declared simplification: in reality, a Saturday's pattern (leisure, tourism) is probably not just "the same commute as always but lighter" - but the current model does not distinguish this. See "Assumptions and limitations".

### Key verification: does the dominant direction emerge on its own?

`../bergen-boats/parametros_instancia_base_bergen.md` had assumed **90/10** toward Bryggen in the morning, by eye, without data. Here **that number was not fixed** - it was left to emerge from the model (population in neighborhoods + employment in Bryggen). Actual result:

| Time band | % intensity toward Bryggen | % intensity from Bryggen |
|---|---|---|
| Morning | **85.6%** | 14.4% |
| Off-peak | 50.0% | 50.0% (by design: symmetric "mixed" role) |
| Afternoon | 14.4% | **85.6%** (perfect inversion relative to the morning) |
| Night | 50.0% | 50.0% |

With the grunnkretser-based masses (Task 1) and the `factor_universitario` set by the team (which discounts Kleppestø/Laksevåg as a destination), the result was **85.6/14.4** - close to the 90/10 originally assumed without data. It remains consistent (exact morning↔afternoon inversion, by construction). See `output/inversion_direccion_bryggen.png`.

---

## Task 4 - Minute-by-minute arrivals (day and week)

**Notebook:** `notebooks/03_llegadas.ipynb` · **Output:** `output/grupos_dia.csv`, `output/grupos_semana.csv`

For each `(origen, destino, hora)`, the Task 3 intensity is converted into an expected rate of **groups** per hour (not individual people - grouped from this first iteration onward), using the passengers/day scale (see below) and the mean group size. Within each hour: `N ~ Poisson(λ_hora)` groups, each with a uniform minute within the hour.

### Demand scale: percentage of real population, not a made-up number

The first version used `demanda.escala_global_viajes_dia = 2000` (passengers/day), a number set by hand with no support. It was replaced by `demanda.porcentaje_poblacion_dia`, applied to the real population of the 4 zones (`poblacion_total` from `masas_por_nodo.csv`, Task 1 v2 = **79 943 inhabitants**). Neither of the two reference papers reports a modal share percentage directly, so it was derived by cross-referencing their demand scales against population:

| Reference | Reported scale | Translated to % of population |
|---|---|---|
| **Gu & Wallace (2021)** - the original water-taxi paper for Bergen, **uses these same 4 nodes** (Kleppestø, Laksevåg, Bryggen, Sandviken & Hegreneset) | 300 "demands"/day (its Table 2: 76+74+74+76) × 8.6 pax/group (Braathen) = 2 580 pax/day | **3.2%** of the population of the 4 zones (79 943 inhabitants) |
| **Braathen, Goez & Guajardo (2024)** - large instance, hypothetical citywide network of 6 stops | 85 920 pax/week = 12 274 pax/day | **4.3%** of Bergen's total population (~285 900 inhabitants) |

Both references converge around 3-4%. The team, however, set the parameter at **`porcentaje_poblacion_dia: 0.10`** (10%) - deliberately above that range, as a higher-adoption/higher-demand scenario, not as a direct reading of the two references. This is explicitly declared as a team decision (see "Assumptions and limitations"), not a figure drawn from the literature. With this, the weekday scale comes out to **7 994 passengers/day** (`79 943 × 0.10`), calculated dynamically from the population - if the Task 1 area outlines are redrawn, the scale recalculates itself in the same proportion, with no loose, out-of-sync number left behind.

### Group size: Binomial, not Normal

Braathen et al. (2024) report a mean of 8.6 pax/group and a standard deviation of 2.6 (its Fig. 1-2) but **do not state which distribution to use** - in the paper, group size is deterministic (a function of the popularity score), not drawn from sampling a distribution. The first version of this step used a rounded `Normal(8.6, 2.6)`, which is not appropriate: it is continuous and symmetric, intended for continuous quantities, not for counts of people (discrete, non-negative).

It was replaced with **Binomial(n=40, p=0.215)**, calibrated by the method of moments to match Braathen's mean and standard deviation:

```
mean = n·p = 8.60          (target: 8.6)
standard deviation = √(n·p·(1-p)) = 2.598   (target: 2.6)
P(group of 0 people) = 0.006%    (truncated Normal: ~0.09% - 15x more frequent)
```

Besides being discrete by construction (there is never a need to round or truncate negative values), it is more realistic: it can be interpreted as there being a "candidate" group of up to 40 people nearby in time/place, each independently deciding with a 21.5% probability to join that trip - a more plausible group-formation process than a continuous bell curve.

- **Maximum wait:** 15 min on strong connections (reuses `bergen-boats/config/instance.yaml → garantia.conexiones_fuertes`: Bryggen↔Kleppestø/Laksevåg/Sandviken), 30 min elsewhere, with ±5 min jitter.
- **Week:** `numpy.random.SeedSequence(42).spawn(7)` → 7 child seeds, one per day - same underlying pattern, distinct and reproducible arrivals.

### Result, generated week (seed 42)

| Day | Groups | Passengers | Weekend |
|---|---|---|---|
| 0 (Mon) | 940 | 8 124 | No |
| 1 (Tue) | 953 | 8 385 | No |
| 2 (Wed) | 963 | 8 097 | No |
| 3 (Thu) | 932 | 8 120 | No |
| 4 (Fri) | 929 | 8 137 | No |
| 5 (Sat) | 372 | 3 206 | Yes |
| 6 (Sun) | 369 | 3 108 | Yes |

Weekdays hover around ~8 100 derived passengers/day (10% of 79 943 inhabitants); weekends drop to ~40% (`factor_dia_semana.fin_de_semana = 0.4`, plus Poisson randomness) - same directional pattern, only the volume changes (see Task 3). See `output/pasajeros_por_dia_semana.png`.

**Observed group size (full week):** mean 8.64, standard deviation 2.64 - against the Binomial(40,0.215) target of 8.60/2.598, and well below the boat's capacity (20 passengers, `bergen-boats/config/instance.yaml`). See `output/histograma_tamano_grupo.png`.

**Reproducibility explicitly verified:** same seed → identical table (`.equals()` in the notebook, `True`). Different seed → different result. See notebook, verification section.

**Hourly demand profile** (`output/perfil_demanda_hora.png`): the morning peak (6-9h) and afternoon peak (15-17h) are clearly visible, with off-peak and night lower - it reproduces the expected shape without having forced it directly in the arrival generation (it comes from the Task 3 time bands).

**O-D heatmap of the generated arrivals** (`output/heatmap_od_llegadas_generadas.png`): Kleppestø↔Bryggen is the dominant pair (7 249 + 7 379 = 14 628 passengers/week) - consistent with Kleppestø having the largest population among the 4 nodes (25 153 inhabitants) and Bryggen the largest employment (52 213), reinforced by the `factor_universitario` that concentrates destination attraction on Bryggen/Sandviken. Bryggen↔Laksevåg (6 079+6 264 = 12 343) and Bryggen↔Sandviken (4 992+4 948 = 9 940) follow.

### Columns of the groups table

`grupo_id, dia, franja, hora, minuto_dia, origen, destino, tamano_grupo, espera_maxima_min, conexion_fuerte, fin_de_semana`

---

## Assumptions and limitations

Honest list of what was assumed throughout this step (Tasks 0-4), so the committee can judge how solid each number is. It is ordered from highest to lowest impact on the final results.

- **Drawing of the 4 study areas (`Areas_estudio.gpkg`):** this is the decision with the most leverage in the whole step - it defines population, employment, and, in cascade, the demand scale (see below). It is a team judgment call made by looking at the map, not a measured datum; a different outline would have given a different population/employment per node and, therefore, a different total demand scale (since the latter is a percentage of that population). Recommendation for a future iteration: run a sensitivity analysis of the results against 2-3 reasonable alternative outlines.
- **Grunnkretser selection criterion ("touches → taken in full", Task 1 v2):** more generous than "center inside" - it can incorporate large, sparsely populated grunnkretser (mountain, forest) that only brush the drawn edge, as happened with Bryggen (its area rose from 4.0 to 20.2 km² without population/employment rising proportionally). It inflates the area, without distorting the population/employment much, but it is a property of the method that must be kept in mind.
- **`factor_universitario` (kleppesto=0.7, laksevag=0.85, bryggen=sandviken=1.0):** a design decision by the team, not a source of real enrollment data (which still has not been obtained). It has a direct impact on the morning/afternoon direction (see Task 3: 85.6/14.4). Gu & Wallace (2021) does mention a university next to Sandviken, which gives some qualitative support for why Sandviken is not penalized, but the exact numeric values are the team's, not from a source.
- **Demand scale (`porcentaje_poblacion_dia = 10%`):** this is no longer a made-up, unanchored number - it is a percentage of real population, and the two cross-referenced sources (Gu & Wallace ≈3.2%, Braathen et al. ≈4.3%, see Task 4) give a range of 3-4%. The team chose **10%**, deliberately above that range (a higher-adoption scenario) - this is an explicit team decision, not a reading from the literature, and it **inherits any error from the area outlines** (if the zones' population changes, the scale changes in the same proportion).
- **Group size (Binomial(40, 0.215)):** discrete and calibrated by the method of moments to match the mean/standard deviation *reported* by Braathen et al. (8.6/2.6) - not their actual deterministic formula (which depends on a `β` scale specific to their station data, not directly transferable to SSB masses). It is an improvement over the truncated `Normal` from the first version, but it is still a distribution choice made by the team, not measured from real Bergen data (which do not exist because the service does not exist).
- **Mass normalization (dividing by the maximum across the 4 nodes):** our own scaling decision, it does not come from the paper or from SSB. A different normalization (by the total sum, or log) would give a different relative intensity.
- **"Mixed" roles in off-peak/night bands:** a simple average of normalized population and employment - a simplification; off-peak trip purposes (errands, leisure, health) do not necessarily follow that mix.
- **Weekend = same directional pattern, less volume:** `factor_dia_semana` only scales the day's total passengers (Task 4); it does not change the per-band roles from Task 3. A Saturday has the same morning→Bryggen/afternoon→home commute as a Monday, only lighter - not a pattern of its own for weekends (leisure, tourism). See Task 3, explicit verification with the generated data.
- **Distance factor (`exp(-tiempo/tau)`, tau=15 min):** shape and parameter chosen for reasonableness within the instance's range of travel times (5-11 navigable minutes), not calibrated against real demand-elasticity-vs-travel-time data.
- **Time-band and day-of-week factors** (relative volume, weekday vs. weekend): set with our own judgment, same as Braathen et al. - they do not come from a real passenger count in Bergen.
- **Maximum wait per group:** design values (15/30 min + jitter), not measured.
- **Exact provenance of the 2 SSB GeoJSON files:** pending confirmation of the dataset's exact URL/name (see "Data provenance" above) - the plausibility of the numbers (national population sum ≈ Norway's real population) gives confidence, but the formal citation is missing. The grunnkretser source, on the other hand, was fully identified.

## References

- **Gu, Y. & Wallace, S.W. (2021).** "Operational benefits of autonomous vessels in logistics - A case of autonomous water-taxis in Bergen." *Transportation Research Part E* 154, 102456. `papers/Operational benefits of autonomous vessels in logistics-A case of.pdf`. **Uses the same 4 nodes as this project** (Kleppestø, Laksevåg, Bryggen, Sandviken & Hegreneset) - it is the most comparable reference for the demand scale (Task 4) and confirms the university next to Sandviken (Task 2/3).
- **Braathen, C., Goez, J.C. & Guajardo, M. (2024).** "Autonomous ferries in light of labor regulations - A passenger perspective." *Maritime Transport Research* 7, 100115. `papers/Autonomous ferries in light of labor regulations-A passenger.pdf`. Source of the multiplicative group-generation method (§4.1, Task 3) and of the group size statistics (Task 4).

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00b_grunnkretser.ipynb             # zone exploration
jupyter nbconvert --to notebook --execute --inplace notebooks/00c_masas_por_grunnkretser.ipynb   # Task 1
jupyter nbconvert --to notebook --execute --inplace notebooks/01_empleo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_patron_od.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_llegadas.ipynb
```

In order - `02` and `03` depend on the `output/masas_por_nodo.csv` that `00c` leaves behind. On Windows, the notebooks that use the grunnkretser WFS (`00b`, `00c`) need `PYTHONUTF8=1` in the environment (names with å/æ/ø break GDAL otherwise). All notebooks that download maps need an internet connection.
