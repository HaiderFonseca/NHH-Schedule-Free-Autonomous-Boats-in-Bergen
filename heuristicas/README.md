# Dispatch heuristics - H0 → H1 → H2 → H3

**What this is, in one sentence:** the strong-heuristics phase (without RL) of the on-demand boats in Bergen project - four dispatch policies, each built as a strict extension of the previous one, audited and validated with controlled test cases before comparing them in a full experiment.

**Why there is a `heuristicas/` folder separate from `politica_base/`:** the first versions of these ideas (`politica_base/src/politica_h1.py`, `politica_h2/`, `politica_h3/`, `politica_h0c/`) were tested, did not consistently outperform the base policy, and an audit found out why (a cost function with a misplaced term). Those folders no longer exist in the repo - they were cleaned up once it became clear that the current, audited, and validated definition is the one here. `politica_base/` now contains only `politica_base.py` (H0, the reference) and the generated demand (`output/escalon_dia_10pct/grupos_seed*.csv`) that the 4 policies here reuse.

---

## 1. The control loop (the same for all 4 policies - this never changes)

The simulator (`simulador/src/env.py`) advances in fixed 2-minute steps. No policy touches this - all of them receive the same snapshot of the world and return the same kind of decision (which node each free boat should go to, or `None` to wait). The control loop (in each notebook) does, at each step:

```text
1. Look at the current state: which queues have people waiting, where each boat is,
   which ones are free.
2. For each FREE boat, the policy decides: which node does it go to, or does it wait?
   (a boat IN TRANSIT does not receive any new decision - the engine does not allow
   redirecting it mid-route)
3. The engine applies the decisions: the free boat that decides to "go to B" boards
   immediately whatever fits from the queue (A,B), FIFO up to its capacity, and departs.
4. The engine advances the clock 2 minutes. The boat that arrives at its destination drops off ALL
   the passengers it is carrying (direct trip, never mixes destinations) and becomes free.
5. The engine adds to the queues the people who arrived during those 2 minutes
   - only visible for the NEXT decision, never for the one just made.
6. Back to 1.
```

No policy ever sees demand that has not yet arrived (step 5) ( this is what guarantees that none of the 4 uses information from the future. The only difference between H0/H1/H2/H3 is **what step 2 computes** ) the rest of the cycle is identical, and none of them modifies it.

---

## 2. H0 - base policy (`politica_base/src/politica_base.py`)

Rule, for a free boat at node A:


1. **Local demand first.** If there are queues departing from $A$, the boat goes to the destination whose oldest person has waited the longest:

2. **Repositioning.** Only if there is no local demand, the boat goes to find the most urgent remote demand (tie: the node closest in travel time):


Coordination among several free boats in the same step (`asignar_flota`): it decides the boats in the order they appear in the list, subtracting from a local copy of the queues what each one would take, before moving on to the next.



## 3. H1 = H0 + persistent reservation (`h1_reserva/src/politica_h1.py`)

The whole decision rule above is kept exactly the same. The only thing that changes is how several free boats are coordinated, and that there is now a reservation that survives between steps.

### 3.1 The reservations dictionary

```python
reservas: dict[str, tuple[str, str]]   # boat_id -> (origin, destination)
```

Lives outside the policy, maintained by the control loop (just as it already maintains `obs, info` from `env.step()`):

```python
reservas = {}
while True:
    decisiones, reservas = asignar_flota_h1(libres, estado, matriz_tiempos, capacidad, reservas)
    ...
```

### 3.2 When it is created

Only when a boat decides to *reposition* (branch 2 of the rule - remote demand). A *local* decision (branch 1) boards immediately - nothing pending is left to protect, so it does not generate a reservation.

Step-by-step example: t=100 min, a free boat B1 at Bryggen, with no local demand. On Laksevåg→Kleppestø there are 3 people waiting. B1 decides to reposition to Laksevåg → `reservas["B1"] = ("laksevag", "kleppesto")` is created.

### 3.3 How it protects against duplication

At each step, before evaluating candidates, the effective queue of each pair is computed: `len(cola) − Σ(boat_capacity for each reservation pointing to that pair)`. With capacity 30 and a single active reservation on `(laksevag,kleppesto)`, that queue is left with effective availability `max(0, N−30)` for any OTHER free boat - if it has fewer than 30 people, it is left at 0: no other boat sees it as a candidate.

People keep arriving at the same queue while B1 is traveling (2 min later 4 more arrive, total 7) - the reservation keeps covering up to 30, so the 7 remain protected without anyone needing to recalculate anything (verified in `01_metodologia_heuristicas.ipynb`, Case 2, with two separate calls: the reservations that enter step 2 are *identical*, bit for bit, to those that came out of step 1).

### 3.4 When it is released

A single possible event: the boat arrives at its destination and becomes free again. The engine never redirects a boat mid-route (`env.py`) and there is no cancellation mechanism, so "the reservation stops being viable" cannot happen before arrival - release happens automatically, at the start of the next call to the policy, for any boat whose `.libre` is already `True`.

### 3.5 Coordination among several free boats

Instead of H0's arbitrary list order, each free boat proposes its best option (local if it has one, remote otherwise); the proposal with the GREATEST wait among ALL pending proposals is committed first, it is subtracted, and the process repeats.



## 4. H2 = H1 + cost, with the local/remote partition preserved (`h2_costo_local/src/politica_h2.py`)

Cost function (in minutes):

$$C(b,q) = T_{pickup}(b,q) - W_{max}(q)$$

- $T_{pickup}(b,q)$: minutes boat $b$ takes to reach the origin of queue $q$ (0 if it is already there).
- $W_{max}(q)$: minutes the oldest person in queue $q$ has been waiting.

Rule per boat:

```text
1. If there is local demand: compare ONLY the local queues using C, choose the one with the lowest C.
2. If there is no local demand: compare ONLY the remote queues using C, choose the one with the lowest C.
3. If there is nothing: wait.
```

**Why the local branch does not change anything relative to H0:** for any local candidate, $T_{pickup}=0$, so $C=-W_{max}$ - minimizing $C$ is exactly maximizing the wait, the SAME rule as H0. The cost only contributes something new in the remote branch: instead of "greatest wait, tie broken by shortest trip" (H0's rule, which compares by thresholds), it makes a continuous trade-off - a remote queue that is somewhat less urgent but much closer can beat one that is more urgent but very far away.

**A remote queue NEVER competes with a local one**, regardless of magnitude - verified numerically: with 5 people waiting 3 min in a local queue and 5 people waiting **200 min** in a remote one, H2 still chooses the local one (`01_metodologia_heuristicas.ipynb`).



## 5. H3 = H1 + global cost, no partition (`h3_costo_global/src/politica_h3.py`)

The exact same cost function as H2. The only difference:

```text
candidates = ALL queues with people waiting (local AND remote at the same time)
-> choose the one with lowest C, without any level restriction
```

**The exact threshold for when remote beats local** (for a boat with local demand wait $W_{local}$ and a remote queue with pickup $T_{pickup}$):

$$W_{remoto} > T_{pickup} + W_{local}$$

That is: remote only wins when its wait advantage exceeds what it costs to go get it - not because of trivial differences. **Verified numerically:** with pickup Bryggen→Kleppestø = 11.5 min and local wait = 3 min, the theoretical threshold is 14.5 min; H3 actually crosses over to the remote queue exactly between 14 and 15 min of remote wait, not a minute before (`01_metodologia_heuristicas.ipynb`, Case 4).

This is the only one of the 4 that allows remote demand, if urgent enough, to beat local demand - correcting the documented weakness of H0 (it can leave someone very urgent in another node waiting indefinitely while the boat handles anything local, no matter how low its urgency).


## 6. Comparison of the 4 rules

| | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| Decision rule per boat | strict local > remote | same as H0 | strict local > remote, cost within each level | global cost, no partition |
| Coordination among free boats in a step | list order (repositioning bug) | by urgency (bug fixed) | by urgency | by urgency (cost) |
| Persistent reservation between steps | no | yes | yes | yes |
| Cost function | none | none | $C=T_{pickup}-W_{max}$ (remote only) | $C=T_{pickup}-W_{max}$ (local and remote) |
| Can remote demand beat local? | never | never | never | if $W_{remoto} > T_{pickup}+W_{local}$ |
| Forecasting / future demand | no | no | no | no |
| Multi-stop | no | no | no | no |
| Weights to calibrate | none | none | none | none |



## 7. Central result (5 evaluation seeds, capacity 30, full day at 10% of population)

Mean wait (min), full 4-16 boat sweep:

| Boats | H0 | H1 | H2 | **H3** |
|---|---|---|---|---|
| 4 | 73.9 | 71.5 (−3%) | 72.2 (−2%) | 72.9 (−1%) |
| 6 | 26.6 | 25.4 (−4%) | 25.7 (−3%) | **24.3 (−8%)** |
| 8 | 12.0 | 12.0 (0%) | 12.0 (0%) | **10.6 (−12%)** |
| 10 | 7.6 | 7.8 (+3%, worse) | 7.8 (+3%, worse) | **6.3 (−17%)** |
| 12 | 6.2 | 7.3 (+18%, worse) | 7.0 (+13%, worse) | **4.8 (−22%)** |
| 14 | 5.6 | 7.1 (+27%, worse) | 6.8 (+22%, worse) | **4.4 (−22%)** |
| 16 | 5.5 | 7.2 (+31%, worse) | 6.7 (+23%, worse) | **4.1 (−25%)** |

H1 and H2 do not consistently outperform H0 - the gap against them worsens with fleet size (H1 reaches +31% worse at 16 boats): they are more conservative about when to move a boat, with no way to assess whether it is worth it, and the fleet ends up idle more often. H3 is the only one that beats H0 across all 7 fleet sizes tested, and the advantage holds even as it grows (−25% at 16 boats) - with fewer total movements than any of the other three.

H3's curve flattens but does not plateau completely: from 12→14 boats it improves −8.8%, from 14→16 only −6.9% - clear diminishing returns, but H3 keeps extracting value from additional boats beyond where H0 is already essentially flat (H0 from 14→16 only improves −2.3%). See `03_comparacion_flota.ipynb` for the full table and curves (wait, time in system, backlog, movements, occupancy, and the series of boats moving vs. idle per hour of the day, included as evidence for evaluating a variable-size fleet throughout the day).


## 8. Structure

```
heuristicas/
├── README.md                          -- this file
├── comun/src/reservas.py              -- shared persistent reservation (create/release/subtract)
├── h1_reserva/src/politica_h1.py
├── h2_costo_local/src/politica_h2.py
├── h3_costo_global/src/politica_h3.py
├── experimento_fleet_sweep.py         -- standalone fleet sweep (same logic as notebook 03, to run without Jupyter)
├── notebooks/
│   ├── 01_metodologia_heuristicas.ipynb   -- rules for each policy + the 5 controlled cases, run live
│   ├── 02_experimentos_heuristicas.ipynb  -- H0/H1/H2/H3 under identical conditions (1 fleet, 1 seed), interactive charts
│   ├── 03_comparacion_flota.ipynb         -- 4-16 boat sweep x 5 seeds, fully interactive (Plotly)
│   └── 04_visualizacion_heuristicas.ipynb -- 2h animation, real map, explicit reservations, step-by-step decision inspector
└── outputs/
    ├── resultados/  -- CSVs (per run, aggregated, and per-minute movement series)
    └── figuras/     -- GIFs (individual per policy + combined); the service and fleet charts
                        are interactive (Plotly) and live only inside notebooks 02/03, not exported to file
```

## 9. What none of the 4 does (constraints respected by all of them)

- Does not use demand information that has not yet arrived (section 1, step 5).
- Does not perform anticipatory repositioning or forecasting.
- Does not allow multi-stop routes - every trip is direct origin→destination.
- Has no weight or coefficient to calibrate - everything that is not a boolean rule is expressed in minutes.
- Does not implement a second boarding/capacity logic - that remains, in all 4, the exclusive responsibility of `simulador/src/env.py`.

## 10. How to run

```bash
cd heuristicas/notebooks
jupyter nbconvert --to notebook --execute --inplace 01_metodologia_heuristicas.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_experimentos_heuristicas.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_comparacion_flota.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_visualizacion_heuristicas.ipynb
```

Requires `politica_base/output/escalon_dia_10pct/grupos_seed*.csv` already generated (previous phase) and `bergen-boats/02_ruteo_navegable/output/` (navigable routes + time matrix).
