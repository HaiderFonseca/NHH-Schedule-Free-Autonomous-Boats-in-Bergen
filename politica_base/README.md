# Base policy ("nearest-available") - reference + simulator verification

**What this is, in one sentence:** the reference policy (a fixed rule, not learned) and the notebooks that, by running it over demand of increasing size, thoroughly verify that the engine (`simulador/`) does what it should - before connecting any RL agent (`modelo_rl/`).

---

## 1. How it is organized

```
politica_base/
├── README.md                      # this file
├── src/
│   └── politica_base.py           # the "nearest-available" rule + fleet coordination
├── notebooks/
│   ├── 00_preparar_demanda_escalones.ipynb   # generates the test demand (stages 1-3)
│   ├── 01_escalon_1_verificacion.ipynb       # morning window, visual verification + animation
│   ├── 02_escalon_2_metricas.ipynb           # full day, aggregated metrics (no animation)
│   └── 03_escalon_3_semana.ipynb             # full week (7 days), aggregated metrics
└── output/
    ├── escalon1/                  # everything produced by notebook 01
    ├── escalon2/                  # everything produced by notebook 02, same file structure
    └── escalon3/                  # everything produced by notebook 03 (week)
```

The notebooks import `simulador/src` (engine: `env.py`, `estado.py`, `recompensa.py`, `metricas.py`, `visualizacion.py`) in addition to its own `src/politica_base.py`, and read `simulador/config/instance.yaml` (single source of truth for shared parameters). Each stage has its own folder in `output/` with the same filenames inside (`grupos.csv`, `metricas_por_par.csv`, `wait_profile.html`, etc.) - easy to compare one stage against another by opening the corresponding folder. The only difference in content: `escalon1/` has `animacion.gif` (the run is small, 90 steps, so it can be animated); `escalon2/` and `escalon3/` do not (long runs - a GIF of that duration is heavy and takes much longer, and the step-by-step visual verification was already done thoroughly in stage 1). `escalon3/` uses `grupos_semana.csv` instead of `grupos.csv` (7 days concatenated, column `dia`).

---

## 2. The reference policy ("nearest-available")

`src/politica_base.py` has two functions:

- **`politica_base(barco, estado, matriz_tiempos, cfg)`** - decides for ONE boat: it prioritizes demand leaving from its current node (immediately servable); if there is nothing there, it considers repositioning empty toward the node with the most urgent demand in the rest of the system.
- **`asignar_flota(barcos_libres, estado, matriz_tiempos, capacidad_barco, cfg)`** - applies the above to **several free boats at once, one by one**, locally deducting what each boat would already "take" before deciding the next one. This is the correct way to use the policy with more than one boat (see "fleet coordination" below) - the one actually used in the notebooks.

It is a **competent, non-trivial** heuristic: it is not just "go to the nearest neighbor" - it combines strict priority for local demand, repositioning to the most urgent remote demand when there is nothing local, and coordination among several free boats in the same step so as not to duplicate effort. It serves as a serious baseline against which to compare the RL agent (`comparacion/`) - it is written as independent functions, separate from the engine, precisely so it can be compared against another policy without touching the simulator.

### 2.1 Who boards each boat, and where it goes

**The policy does NOT choose who it picks up, it chooses the destination NODE.** Who boards is a fixed rule of the simulator (`simulador/`): the queues are separated by origin-destination pair (12 queues, one for each combination). When a free boat at A receives the order "go to B", it can only board from the exact A→B queue, in order of arrival, until the boat is full - it never carries people with mixed destinations, and upon arriving at B it drops everyone off. It is a direct point-to-point trip, not a route with intermediate stops.

**What if the people who most need a boat are at ANOTHER node?** An important rule, not the only reasonable way to do it:

> **The policy ALWAYS prioritizes demand at the node where the boat already is, over any demand at another node - no matter how long the people at other nodes have been waiting.**

Concrete example: a boat becomes free at Bryggen. Right there are 5 people who just arrived (waiting 1 minute) wanting to go to Laksevåg. At Kleppestø there is 1 person who has been waiting 20 minutes to go to Sandviken. **The policy sends the boat to pick up the 5 people at Laksevåg, not the person at Kleppestø** - even though that person has been waiting much longer and is much closer to being lost. The rule does not compare "how urgent each one is" globally; it first exhausts ALL local demand (no matter how small or recent) and only looks at other nodes when absolutely no one is left waiting at its own node.

Why was it designed this way? So as not to leave "abandoned" people that the boat could already serve immediately, in exchange for chasing someone who still needs to travel. It is a reasonable design decision but **not the only possible one** - an alternative would be to compare the urgency of ALL candidates (local and remote) in a single list, and have the boat sometimes go fetch someone far away if they are much more urgent than local demand. That alternative is not implemented; the current one is simpler to explain and verify, but it can lead to someone very urgent at another node waiting much longer than reasonable. Since the simulator does not purge anyone for patience (`simulador/README.md`, section 4.5), that wait no longer has an implicit limit - it can grow indefinitely if the demand pattern never gives priority to that pair. This is stated as a known limitation, not hidden (section 5).

Mechanically, in `politica_base`: it first builds the `candidatos_directos` list (only pairs that LEAVE the boat's current node); if that list is not empty, it picks the most urgent one from it and that is it - it never looks at `candidatos_reposicion` (demand from other nodes) unless `candidatos_directos` is completely empty. When it does reposition, the boat travels **empty** to the node with the most urgent remote demand, and there, at its next free moment, it decides again with fresh information - it is a myopic decision (it does not plan both stages at once).

### 2.2 Fleet coordination - why `asignar_flota` exists

With the policy applied INDEPENDENTLY per boat (each one looking at the same snapshot of the world, before either of them has boarded anyone), two free boats at the same node and moment can both decide "go to Bryggen" thinking there are, say, 11 people waiting there - whichever gets processed first takes all of them, the second travels **empty** to Bryggen, paying the movement penalty without serving anyone. This was detected during the original verification by inspecting a specific minute of an example run (`visualizacion.inspeccionar`) and was confirmed in the text log.

`asignar_flota` fixes this by assigning the free boats one at a time, with a LOCAL copy of the queues that gets deducted as each boat decides - so the next boat in the list sees the queue already reduced by what the previous one would take. Effect measured at the time (under the patience system, before it was removed entirely - `simulador/README.md` section 4.5): the compliance percentage rose from 53% to 90% - this was not a minor detail. It is exactly the kind of coordination that turns "a simple rule applied several times" into "a competent heuristic" - without it, more boats in the fleet do not necessarily serve more people.

---

## 3. The stages (test instances, increasing demand)

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| When | Morning window (6-9h) | Full day (6-24h) | Full week (7 days, 6-24h each) |
| Boats | 2 | 3 | 3 |
| Groups / people | 23 / 202 | 133 / 1106 | 620 / 5429 |
| 2 min steps | 90 | 540 | 7 × up to 540 (7 independent episodes) |
| Purpose | Visual verification, text log, animation | Aggregated metrics, no animation | Aggregated metrics over weekdays + weekend |
| Notebook | `01_escalon_1_verificacion.ipynb` | `02_escalon_2_metricas.ipynb` | `03_escalon_3_semana.ipynb` |

The demand for each stage was generated with `demand/src/llegadas.py` **without modifying it** - it was simply passed a `porcentaje_poblacion_dia` lower than the official one (10%, in `demand/config/instance.yaml`, untouched), calibrated by trial and error until approaching "~20-30 groups" in stage 1. Stage 3 reuses the density and fleet from stage 2 over `generar_llegadas_semana` (`demand/src/llegadas.py`) instead of `generar_llegadas_dia` - 7 days (Monday=0..Sunday=6), with the weekday/weekend factor already resolved internally.

---

## 4. Verification results

**Conservation** (`metricas.verificar_conservacion`): every person generated must end up accounted for.

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Generated | 202 | 1106 | 5429 |
| = Served + | 202 | 1106 | 5406 |
| Waiting at the end + | 0 | 0 | 0 |
| On board at the end | 0 | 0 | 23 |

Stages 1 and 2: 100% served (the time window is enough to drain the backlog). Stage 3: **99.58%** - with 7 closing windows instead of one, there are 7 times more opportunities for someone to board a boat right before the operating hour ends (24:00) and the boat does not manage to arrive before the cutoff. This is not a bug: it is exactly the case that the "waiting/on board at the end" categories exist to correctly capture (the simulator never "loses" anyone, `simulador/README.md` section 4.5).

**Global** (`metricas.metricas_globales`):

| Metric | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| % served | 100.0% | 100.0% | 99.58% |
| Mean wait | 17.4 min | 10.9 min | 10.0 min |
| Mean / maximum time in system | 27.2 / 54.0 min | 20.5 / 53.6 min | 19.8 / 57.1 min |
| Wait p50 / p90 / p95 | 15.6 / 33.4 / 33.5 min | 9.4 / 21.6 / 26.3 min | 9.1 / 21.4 / 25.1 min |

Stage 2, despite having 5.5× more demand and 9× more steps than stage 1 (with only 1 more boat), ends up with **lower** mean wait and p90/p95 - this makes sense: stage 1 concentrates all its demand in a 3h peak window, with no margin for the backlog to drain between peaks, while stage 2 covers 18h with alternating high- and low-intensity windows, giving the fleet more opportunities to catch up during the lulls. Stage 3 demand per day: Monday-Friday between 756 and 1121 people; Saturday and Sunday, 324-326 (factor `fin_de_semana=0.4` from `demand/config/instance.yaml`). The backlog of the 23 people on board at closing is concentrated in 3 pairs (`bryggen->laksevag`: 10, `laksevag->bryggen`: 7, `kleppesto->bryggen`: 6) - the busiest routes of the week.

Reproducibility verified across the three stages: same demand seed → same exact run; different seed → different result.

**By origin-destination pair, by boat, by user (percentiles), and backlog at the end** - complete tables in each notebook (`metricas.metricas_por_par`, `metricas_por_barco`, `metricas_por_usuario`, `metricas.sin_atender_al_final_por_par`), also saved in `output/escalonN/metricas_por_par.csv` / `metricas_por_barco.csv`. In stages 1 and 2, `sin_atender_al_final_por_par` returns an empty table (nobody was left unserved), consistent with the 100%.

**Charts** (`simulador/src/visualizacion.py`, interactive - Plotly; `output/escalonN/*.html`): time profile of people waiting (`wait_profile.html`), fleet occupancy over time (`fleet_occupancy.html`), heatmap of % served by pair (`pct_served_heatmap.html`), reward breakdown over time (`reward_breakdown.html`), backlog bars by pair (`backlog_by_pair.html`) - same function, for all three stages. Native zoom/pan, exact values on hover; the `fleet_occupancy.html` legend allows isolating one boat (click) or showing them all again (double click).

`escalon1/` also has `animacion.gif` (90 frames, one per step), `visualizacion.inspeccionar` (a specific minute - always works, does not require a live kernel) and `visualizacion.reproductor_interactivo` (step-by-step buttons, **only works with a live Jupyter kernel**: the notebook must be opened in VS Code/Jupyter Lab and the cells run there, it is not enough to view it already executed). The background map uses Esri.WorldGrayCanvas (no API key needed) - CartoDB.Positron (the one used by `bergen-boats/`) started requiring an API key, and OpenStreetMap blocked the request due to usage policy; it is downloaded once per run, not once per frame.

---

## 5. Assumptions and limitations (of the reference policy)

- **`asignar_flota` always prioritizes local demand over remote demand** (section 2.1), without comparing global urgency. Since the simulator has no patience, this can leave someone very urgent at another node waiting **indefinitely** while the boat serves less urgent local demand, if the demand pattern never gives priority to that pair. In the stage runs this did not end up happening (99.6-100% served), but with a smaller fleet or more unbalanced demand it could - explicitly stated, this is not the only reasonable way to design the policy.
- **`asignar_flota` is myopic, not optimal.** It coordinates the free boats within a SINGLE step so they do not duplicate each other (section 2.2), but it still decides one at a time, in the order they appear in the boat list - it does not evaluate all possible combinations to find the optimal joint assignment. It is a real improvement over the uncoordinated version, but it remains a simple rule, appropriate as a baseline.
- **Empty repositioning:** a design decision of its own, not explicitly detailed in the original specification - it was deduced as the simplest and most consistent way to handle demand outside the boat's current node under the direct point-to-point trip model.
- **Demand scale of the stages (0.8%/1.2% of the population):** values specific to this verification, chosen only so that the run size would be manageable to verify visually - they have no relation to the "official" 10% from `demand/`.

---

## 6. How to run

```bash
cd politica_base/notebooks
jupyter nbconvert --to notebook --execute --inplace 00_preparar_demanda_escalones.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_escalon_1_verificacion.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_escalon_2_metricas.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_escalon_3_semana.ipynb
```

It needs `demand/output/` and `bergen-boats/02_ruteo_navegable/output/` to already exist (previous, completed steps). Notebook 00 must run before 01, 02 and 03 (it generates `output/escalon1/grupos.csv`, `output/escalon2/grupos.csv` and `output/escalon3/grupos_semana.csv`, which the others read).
