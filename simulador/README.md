# Boat dispatch simulator (engine: Gymnasium environment, state, reward, metrics)

**What this is, in one sentence:** the engine that models, minute by minute (2 min steps), what the boats and passengers do in Bergen - no policies or experiments, those live in `politica_base/`, `modelo_rl/` and `comparacion/` (see the project root for the view of the 4 blocks).

**Source document (takes precedence over any doubt):** `docs/especificacion_simulador_rl.md`. Everything below is mapped to its sections (§1-§12).

---

## 1. How it is organized

```
simulador/
├── README.md                 # this file
├── config/instance.yaml      # SINGLE source of truth for the parameters shared by the 4 blocks
└── src/
    ├── unidades.py            # what a "Unidad" (person or group) is and how they are generated
    ├── estado.py               # the snapshot of the world at each instant + the flattened vector (§4)
    ├── recompensa.py           # the scoring formula (§7)
    ├── env.py                  # the simulator itself - Gymnasium (ties together everything above, §3/§5/§6)
    ├── metricas.py              # conservation check + detailed metrics + combinar_corridas
    └── visualizacion.py         # animated map, interactive charts, inspector for a specific minute
```

`config/instance.yaml` is read (never copied) by the other 3 blocks via a relative path (`../simulador/config/instance.yaml`) - the same rule already used by `demand/`: each folder owns its own parameters, the others read them.

**The other 3 blocks of the project:**

- **`politica_base/`** - the reference policy (`asignar_flota`, fixed "nearest-available" rule + fleet coordination) and its step-by-step verification (notebooks 00-03, the "stages" of increasing demand).
- **`modelo_rl/`** - the PPO agent: the random-demand wrapper used for training, the sequence of experiments on the shape of the reward, and the final long training run.
- **`comparacion/`** - PPO agent vs. base policy, on the same evaluation seeds.

**How to read this if you have never programmed in Python:** each file in `src/` is a "piece" with one responsibility. Nothing runs just by creating the files - a notebook is needed (in `politica_base/`, `modelo_rl/` or `comparacion/`) that imports and runs them.

**What that first line `from __future__ import annotations` is:** a Python language instruction, not something specific to this project - it tells Python to treat type annotations (`nodos: list[str]`, `politica: str | None`) as text instead of evaluating them immediately, avoiding errors in cases where a class refers to itself before it finishes being defined. Standard practice in modern Python.

---

## 2. What Gymnasium is (and what it is NOT)

**Gymnasium does not come with prebuilt simulators to choose from.** It is a library that defines a **standard mold**: any "environment" must be a Python class with exactly two methods:

- `reset()` → starts the world from scratch, returns the initial state.
- `step(action)` → receives an action, advances the world one step, returns `(new_state, reward, terminated, truncated, extra_info)`.

That mold is ALL that Gymnasium imposes. We write 100% of the real logic (what a boat does, how passengers board and disembark) in `src/env.py`, class `SimuladorBarcosBergen`. Gymnasium knows nothing about boats or Bergen - it only guarantees that our class "speaks the same language" that any RL library expects afterward (Stable-Baselines3, already connected - see `modelo_rl/README.md`). It is like a plug: Gymnasium standardizes the shape of the plug, we build the appliance.

**The "type" of simulation** (a different question, about the internal mechanics, not about Gymnasium): it is a **discrete-time simulation with fixed 2-minute steps** - each `step()` always advances exactly 2 minutes, never less or more (unlike an "event-driven" simulation, which jumps straight to the next interesting moment). The specification (§3) explicitly calls for fixed steps, for simplicity and because it fits naturally with Gymnasium's `step()`.

**Are we really using Gymnasium, or just the name?** Yes, really: `src/env.py`, `class SimuladorBarcosBergen(gym.Env):` - it literally inherits from the library's base class (`import gymnasium as gym`). That requires (and verifies at runtime) that the class has `action_space` and `observation_space` properly declared (`MultiDiscrete` for actions, `Box` for the flattened vector) and the `reset()`/`step()` methods with the exact signature the library expects. If `SimuladorBarcosBergen` did not fulfill the contract, Stable-Baselines3 could not use it - in fact `check_env` (`modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb`) confirms this at runtime.

**What seed are we using?** `config/instance.yaml` → `semilla: 42`. That seed is used to regenerate the DEMAND (`demand/src/llegadas.py`), not inside `env.py` - the environment itself has no randomness of its own, so `env.reset(seed=...)` does not change anything by itself; what actually reproduces a run is generating the group table with that seed and running the environment over that same table.

---

## 3. Where each parameter comes from (nothing is invented or duplicated)

Project rule (already used in `demand/`): each folder owns its own parameters; the others **read** them from the source, never copy them.

| Parameter | Lives in | Why there |
|---|---|---|
| Boat capacity (20), initial node (Bryggen), strong connections | `bergen-boats/config/instance.yaml` → `flota`, `garantia` | These are fleet/physical properties, already defined in the routing step |
| Travel times between nodes | `bergen-boats/02_ruteo_navegable/output/matriz_tiempos_min.csv` | Already computed (Dijkstra over real water), never recalculated here |
| Route geometry (for animation) | `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson` | Same - already computed, persisted in that same job |
| Pattern of who travels where and when | `demand/src/llegadas.py` + `demand/output/matriz_intensidad_od.csv` | The real demand generator (gravity + SSB + Poisson), not touched |
| **Tolerance (12 min), fixed normalizer (18 min), per-person cap, movement weight (production)** | `simulador/config/instance.yaml` → `recompensa` | Specific to this engine (reward formula, section 4.4) |
| **Fleet size and % of demand per stage** | `simulador/config/instance.yaml` → `escalones` | Specific to verification (they do not exist in `demand/` or `bergen-boats/`) |
| **Time step (2 min), seed, demand unit (people/groups)** | `simulador/config/instance.yaml` | Specific to this engine |
| **PPO hyperparameters, reward experiments, final training, evaluation seeds** | `simulador/config/instance.yaml` → `agente` | Shared by `modelo_rl/` and `comparacion/` - see `modelo_rl/README.md` |

If you are looking for a number and it is not in `simulador/config/instance.yaml`, it is almost certainly because it belongs to another step and is only **read** here - check the table above.

---

## 4. The MDP explained with real examples

### 4.1 State (§4) - two versions, one for humans and one for the network

Each instant of the world is stored in an `EstadoSimulacion` object (`src/estado.py`). It has TWO ways of displaying itself:

- **`.to_dict()`** - human-readable, for us (and for the functions in `metricas.py`/`visualizacion.py`, which read this form). Real example from a run (trimmed):
  ```json
  {
    "tiempo": {"minuto_del_dia": 412, "dia_semana": 0},
    "barcos": [
      {"origen": "L", "destino": "B", "min_para_llegar": 3.0, "ocupacion": 17, "libre": false},
      {"origen": "L", "destino": "K", "min_para_llegar": 1.0, "ocupacion": 0,  "libre": false}
    ],
    "demanda": {"K->B": {"personas": 20, "espera_max": 6.0}, "...": "..."}
  }
  ```
  `libre` still exists HERE (it is useful for logs/human inspection) even though it is no longer in the flattened vector - see below.

- **Flattened vector (`aplanar_estado`)** - a fixed-size list of numbers (46 for 2 boats), the only thing an RL agent sees. `dimension_vector(num_barcos, num_nodos)` computes the exact size: `10` values per boat (one-hot origin + one-hot destination + `min_para_llegar` + occupancy, with 4 nodes) + `24` (12 O-D pairs × 2: people waiting and maximum wait) + `2` (time: minute of day and day of week, each a single normalized value).

**Two design decisions about the vector, explicit (not accidents):**

- **No cyclical encoding (sine/cosine) of time.** Each simulated run is an INDEPENDENT episode - it never crosses midnight (`env.reset()` goes back to `hora_inicio_min`, `env` ends upon reaching `hora_fin_min`) nor does it change day of week midway through an episode. Cyclical encoding (`sin(2π·minuto/1440)`, `cos(...)`) makes sense when the "end" and the "start" of a time variable are close in the real world (23:59 and 00:00 are almost the same instant) and an agent needs to see them as close - but that never happens here: no episode starts at 23:58 and continues until 00:02 of the next day. Using sine/cosine would only add 2 more numbers to the vector (4 instead of 2) without any new information that a simple linear value does not already provide. That is why `aplanar_estado` uses `minuto_del_dia / 1440.0` and `dia_semana / 6.0` directly - two scalars normalized to `[0, 1]`, nothing more.
- **No `libre` flag in the vector.** It is deducible from two other fields that are ALREADY in the vector: a boat is free exactly when its origin node and its destination node are the same (the origin one-hot == the destination one-hot) and `min_para_llegar ≈ 0`. Repeating that information as a third number gives the network nothing it cannot already infer from the other two - it was removed so as not to load the vector with a redundant value. (It still exists in `EstadoSimulacion.to_dict()`, where it is indeed useful for logs and so that `politica_base`/`metricas.py` do not have to re-derive it everywhere they need it.)

**Compatibility note:** this 46-value vector (2 boats) differs in size from an earlier version (50 values, with sine/cosine and `libre`) - any model trained against that earlier version is now obsolete; the models in `modelo_rl/output/` all postdate this change.

### 4.2 Demand unit: people, not groups

`demand/` generates **groups** (e.g. "3 people leaving Kleppestø together at 6:23"). By default, this engine **explodes** each group into individual people (`src/unidades.py`, function `grupos_a_unidades`) - each person inherits the same origin/destination/arrival time as their group, but is treated as an independent request. It is a switch in the config (`unidad_demanda: "personas" | "grupos"`), not two separate simulators.

`demand/` still computes a patience value per group (column `espera_maxima_min`, 15/30 min ± jitter) because it is a datum specific to that module - but this engine no longer reads it anywhere (section 4.5): the `Unidad` in this step does not have that field.

### 4.3 Actions - direct point-to-point trips

Each free boat receives one of 5 orders: go to each of the nodes, or wait. A boat en route ignores any order and continues until it arrives (it is not redirected mid-route). Who boards each boat is a fixed rule of the simulator (queues separated by origin-destination pair, boarding in order of arrival until capacity is filled) - **who decides each boat's destination is the policy**, see `politica_base/README.md` (reference rule, including coordination among several free boats at once) and `modelo_rl/README.md` (the agent).

**"Wait" - one decision, two equivalent ways of making it.** A free boat can explicitly receive the "wait" order (the dedicated action index, `acciones_posibles[-1] = None`), or it can receive the order to go to its OWN current node (`destino_elegido == nodo_origen`) - the two are, mechanically, the same thing: in `env.py`, `step()`, both fall into the same `continue` (no-op), without deducting anything from any queue, without updating `nodo_destino`, and therefore **without any movement penalty** (`recompensa.py` counts boats in motion by comparing `nodo_origen != nodo_destino` of the state ALREADY updated after the step - a boat that stayed put never changes that comparison). The decision log (`env.log_eventos`, type `"decision"`) records both forms equally as `"esperar"` - so no metric that counts waits by the string `"esperar"` (`metricas.decisiones_por_barco`) undercounts the case of "chose its own node".

### 4.4 Reward - the exact formula, with magnitude examples

```
sobrante      = max(0, tiempo_en_el_sistema - 12)             # 12 = tolerancia_incomodidad_min
penalizacion  = min(techo, (sobrante / 18) ** 2)               # 18 = sobrante_normalizador_min, fixed for everyone
                                                                 # techo = penalizacion_maxima_persona

r = -[ Σ tamano × penalizacion (for each active person/group: waiting OR on board) + peso_movimiento × (boats sailing) ]
    + premio_por_persona_entregada × (people delivered this step)
```

PRODUCTION values (`config/instance.yaml` → `recompensa`, the ones used by `politica_base/` and its stages 1-3): `tolerancia_incomodidad_min=12`, `sobrante_normalizador_min=18`, `penalizacion_maxima_persona=1.0`, `peso_movimiento=0.1`, `premio_por_persona_entregada=0.0`. The RL experiments (`modelo_rl/`) use **overrides** of this same formula (`agente.*.recompensa_overrides`) that NEVER touch these production values - see `modelo_rl/README.md`.

**Why does a cap (`penalizacion_maxima_persona`) exist?** The simulator has no patience (section 4.5) - nobody ever withdraws, so someone can wait a long time if the demand pattern never gives priority to their pair. Without a cap, that one person alone could dominate `r` with an ever-growing number, crushing the signal from everyone else. The cap prevents that: past a certain wait time, that person's penalty stops growing, even if they keep waiting - they keep being "charged", but at the maximum, in a stable way. (The `modelo_rl/` experiments explicitly evaluate whether keeping this cap active or not changes the learned behavior - see `modelo_rl/README.md`.)

**Magnitude examples** (for a person with `tamano=1`, with the production cap = 1.0):

| Situation | That person's penalty |
|---|---|
| Waits 12 min or less (within tolerance) | 0 |
| Waits 18 min (6 min of excess over 18) | (6/18)² ≈ 0.11 |
| Waits 24 min (12 of excess) | (12/18)² ≈ 0.44 |
| Waits 30 min (18 of excess = the full normalizer) | (18/18)² = 1.0 (the cap) |
| Waits 60 min (48 of excess) | without a cap it would be (48/18)² ≈ 7.1 - with the cap, it stays at 1.0 |

A boat moving costs `peso_movimiento` (0.1 in production) - deliberately small so as not to discourage dispatching boats when it is actually needed. The penalty grows with the square of the excess up to the cap, so it stays small while someone is only a little past the tolerance and accelerates the longer they wait - the pressure concentrates on whoever has been waiting longest, it is not spread evenly across everyone.

### 4.5 Why there is no patience or losses (a design decision, not a bug)

The original specification had a per-person **patience** value (15 min on strong connections, 30 min on the rest, with ±ε noise): if nobody served them before it ran out, they withdrew from the system - counted as "lost", never making it onto a boat.

**It was removed entirely** (`unidades.py`, `env.py`, `recompensa.py`, `metricas.py` - no engine file still has a patience field or logic). Now **nobody ever leaves**: every person/group generated waits until served, or until the run cuts off (counted as "waiting at the end" or "on board at the end").

**Why?** Patience introduced a **data censoring** problem: someone who had been waiting a long time and was lost never got to reveal how long they would actually have waited - they were "erased" right before that data point existed, biasing any percentile computed only over served units downward. Without patience, the wait time of ALL served units is real, uncensored data, and the percentiles from `metricas.metricas_por_usuario` (p50/p90/p95/max) can be used directly to define a service-time guarantee grounded in data, not in an assumption.

The cost of this change: there is no longer an explicit signal of "this cannot keep being postponed" in the reward - that pressure depends entirely on the capped discomfort term (section 4.4).

---

## 5. Metrics and visualization (generic functions, policy-agnostic)

`src/metricas.py` - single entry point `reporte_completo(env)`, which brings together: `verificar_conservacion` (every person generated must end up accounted for), `metricas_globales`, `metricas_por_par`, `metricas_por_barco` (movements, time sailing vs. waiting - see next paragraph -, occupancy), `metricas_por_usuario` (wait/travel/system percentiles), `sin_atender_al_final_por_par` (backlog, not "losses": the simulator never loses anyone), `desglose_recompensa`. `combinar_corridas(envs)` combines N independent runs (days of a week, evaluation seeds) into a single object with the same interface, so `reporte_completo` can be called on the set without duplicating any logic.

**Waiting vs. sailing - two independent views, which should match approximately.** `metricas_por_barco` reports `tiempo_esperando_min`/`pct_esperando` by counting, for each step of `env.historial_estados`, whether the boat was free at that step (measured in MINUTES). `decisiones_por_barco` counts how many times the decision log (`env.log_eventos`, type `"decision"`) recorded `"esperar"` for that boat (measured in DECISIONS, see section 4.3 for why both forms of "staying put" count equally there). A boat is "waiting" at a step almost always because it decided `"esperar"` the previous step - except for two cases with no "wait" decision of its own behind them: the first step of the run (the boat starts free by construction, before anyone decides anything) and the exact step in which a just-arrived boat becomes free (it is only asked what to do on the NEXT step). That is why `tiempo_esperando_min / paso_tiempo_min` is slightly GREATER than `decisiones_por_barco()["veces_espero"]` - the expected difference is small and explainable (one for the start, one for each arrival of that boat), and it serves as a cross-check: a much larger difference would signal an incomplete decision log.

`src/visualizacion.py` - interactive (Plotly, not matplotlib): `graficar_perfil_espera` (people waiting over time), `graficar_ocupacion_flota` (occupancy of each boat over time, click-to-isolate legend), `graficar_heatmap_cumplimiento` (% served by O-D pair), `graficar_desglose_recompensa` (discomfort/movement/delivery per step), `graficar_sin_atender_al_final` (backlog by pair). All of them accept any object with the interface of an `env` (a single one, or a `CorridaCombinada`) - the same 5 functions, used identically by `politica_base/`, `modelo_rl/` and `comparacion/`. Plus geospatial: `animar_corrida` (GIF), `inspeccionar` (a specific minute, always works), `reproductor_interactivo` (step-by-step buttons, only with a live Jupyter kernel).

**Where the logs live, while the run lasts** (`env` attributes, in memory, filled automatically inside `env.step()`): `env.log_eventos` (board/alight/decision/movement), `env.log_recompensa` (breakdown per step), `env.historial_estados` (a complete snapshot per step), `env.atendidas_historico` (people already resolved). Nothing is saved to disk automatically - each notebook exports what it needs to its own `output/` folder.

---

## 6. Assumptions and limitations (of the engine)

- **The environment itself has no randomness of its own.** All the pipeline's randomness lives in demand generation (`demand/`, with its seed, or `modelo_rl/src/entrenamiento.py` for RL). Once the group table is fixed, the simulator is 100% deterministic - a desirable property for verification, not a flaw.
- **No patience, no losses - an explicit design decision** (section 4.5). It enables real, uncensored wait percentiles, but it also means the reward no longer distinguishes a strong connection from a normal one (the normalizer is fixed for everyone), and that the pressure against leaving someone waiting a long time depends entirely on the capped discomfort term.
- **Reported wait time** = time until the person's destination was resolved (they boarded a boat) - not the travel time on board after boarding. Time in system does add both together. It is only computed over served units.
- **The wait percentiles (`metricas_por_usuario`) are the basis for defining service-time guarantees with real data**, not assumptions - that is precisely the point of section 4.5.
