# PPO agent vs. base policy

**What this is, in one sentence:** runs the trained agent (`modelo_rl/`) and the reference policy (`politica_base/`) on the same 5 evaluation seeds, aggregated and with full detail for a specific seed.

---

## 1. How it is organized

```
comparacion/
├── README.md                                  # this file
├── notebooks/
│   └── 05_comparacion_agente_vs_base.ipynb    # aggregated comparison (5 seeds) + seed 1001 detail
└── output/                                     # tables, interactive charts, animations, GIFs
```

The notebook imports `simulador/src` (engine), `politica_base/src` (`asignar_flota`) and `modelo_rl/src` (`EntornoDemandaAleatoria`), and reads `simulador/config/instance.yaml`. It needs `modelo_rl/output/modelo_final/modelo_ppo.zip` to already exist (`modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb` already run).

**How it is evaluated:** both policies run on the same 5 seeds (`agente.evaluacion.semillas`: 1001-1005, outside the range of `semilla_entrenamiento`), via `EntornoDemandaAleatoria.reset(seed=...)` - same seed, same demand for both, so the comparison is fair. The reward is computed with the RL formula (`agente.entrenamiento.recompensa_overrides`: no cap, `peso_movimiento=0.003`) for BOTH policies, not the production one - if each used a different formula, comparing the total reward would mean nothing.

---

## 2. Aggregated comparison (5 seeds)

**Current model: 300 000 timesteps (585 PPO updates)** - raised from the 150 000 (292 updates) of the previous version. See `modelo_rl/README.md` section 4.2 for the actual measured time of this run.

**Why we do not lead with % served.** The simulator loses no one - nobody ever withdraws (`simulador/README.md` section 4.5), so someone who did not board a boat by the end of the run is not "lost", they remain in the queue and would have been served if the run had continued a while longer. This means "% served" mixes two different things: how well the policy dispatches, and how much operating window was left when the last people arrived - the second one is arbitrary, it says nothing about the policy. What the reward actually penalizes (and what we truly care about) is how much time people spend in the system - mean and, above all, maximum - and how many movements it costs the fleet to achieve it. That is why the table below leads with those two things, and leaves % served for last, only as a reference.

| Metric | PPO agent | Base policy | Difference |
|---|---|---|---|
| Mean time in system | 26.1 min | 26.1 min | +0.05 min (tie) |
| Maximum time in system | 66.8 min | 69.7 min | **-3.0 min** |
| Mean wait | 16.3 min | 16.4 min | -0.07 min (tie) |
| Total movements (5 episodes) | 185 | 155 | +30 |
| Total reward (5 episodes, RL formula) | **-3754.1** | -5259.0 | **+1504.9** |
| % served (reference, see above) | 87.8% | 88.0% | -0.2 pp |

With more training, the agent now practically TIES the base on mean time and mean wait (it used to cost 3 min more on average; now the difference is noise), and it still wins in the worst case (66.8 vs. 69.7 min) - the expected effect of training without a cap on the penalty (`modelo_rl/README.md` section 3): it keeps costing the agent more and more to leave someone waiting a long time, so it prioritizes not leaving anyone in the worst case, without sacrificing the average anymore as before. The remaining cost is movements (185 vs. 155, 30 more) - it moves the fleet more actively. See section 2.1 for why this is seen even more clearly in the percentiles than in the average.

### 2.1 Percentiles (the real basis for a service guarantee)

`metricas.metricas_por_usuario` - percentiles over SERVED units, uncensored (the simulator loses no one, so this is real data, not a sample biased toward the fast cases). It is what is needed to be able to say something like "we serve 95% of people within X minutes" with support, not just the average - and it is where the agent's advantage is seen much more clearly than in the table above:

| Metric | Percentile | PPO agent | Base policy | Difference |
|---|---|---|---|---|
| Wait | p50 | 14.1 min | 13.9 min | +0.1 min |
| Wait | p90 | 31.9 min | 37.2 min | **-5.3 min** |
| Wait | p95 | 35.0 min | 46.1 min | **-11.1 min** |
| Wait | max | 54.8 min | 59.7 min | **-5.0 min** |
| System | p50 | 23.5 min | 23.0 min | +0.5 min |
| System | p90 | 40.7 min | 37.8 min | +2.9 min |
| System | p95 | 44.2 min | 58.8 min | **-14.6 min** |
| System | max | 66.8 min | 69.7 min | -3.0 min |

At the median (p50) the two policies are practically identical - the real difference is in the TAIL of the distribution: at p95 of wait, the agent beats the base by almost 11 minutes; at p95 of time in system, almost 15 minutes. This is consistent with the theory (`modelo_rl/README.md` section 3, uncapped reward): the agent is not optimized for the typical case, it is optimized so that NO individual case grows without bound - and that is exactly what a high percentile measures. Full table: `output/percentiles_agregado.csv`.

**By seed, the result is not uniform** (`output/reward_por_semilla.csv`):

| Seed | Agent reward | Base reward | Difference |
|---|---|---|---|
| 1001 | -216.0 | -132.8 | -83.1 (base wins) |
| 1002 | -640.1 | -617.1 | -23.0 (base wins) |
| 1003 | -787.7 | -318.8 | -469.0 (base wins) |
| 1004 | -1699.3 | -4091.1 | **+2391.7** (agent wins, by a lot) |
| 1005 | -411.1 | -99.3 | -311.7 (base wins) |

The agent loses in 4 of the 5 individual seeds, but wins the average because on seed 1004 the base has a particularly bad run (-4091.1, the worst of the 10 runs) while the agent handles it well (-1699.3) - same reading as before: the agent is more consistent in the worst case, the base can be better in the typical case but has more downside variance. With more training the agent now loses on one more seed than before (4 of 5, it used to be 3 of 5) - worth keeping in mind: winning the aggregate because of a single extreme case is a more fragile basis than "the agent wins on average" suggests at first glance.

**Does the agent simply not move the fleet?** No - it is just the opposite. `metricas.decisiones_por_barco` (crosses the decision log with the state at that moment - see `simulador/README.md` section 5) shows that the agent waits in only **4.1%** of its decisions (8 of 193), versus **47.5%** for the base policy (140 of 295) - about 12 times less. Full table by boat in `output/decisiones_por_barco_comparado.csv` and `output/por_barco_comparado.csv` (movements, time sailing/waiting, occupancy per boat).

### 2.2 `veces_espero_con_demanda_local` - why the base "waited with local demand" (and no longer does)

The base policy's rule is strict: **if someone is waiting at the node where the boat is, it ALWAYS picks them up before anything else** (`politica_base/README.md` section 2.1) - there is no legitimate case where it decides to wait while having local demand, except one: when ANOTHER boat is also free at the same node in the same step, and fleet coordination (`asignar_flota`, `politica_base/README.md` section 2.2) has already assigned that demand to the first one - the second one, correctly, has nothing to pick up (even though the overall snapshot of the node still shows people waiting, they are already "claimed").

Before this revision, `metricas.decisiones_por_barco` reported many more cases than that rule explains (up to 15 for one boat in a single run) - the real cause was found: the function reconstructs, for each decision, "what snapshot of the world the policy saw at that moment" by crossing the decision log with `historial_estados`, advancing a pointer by 1 each time the decision's minute changed. **If at some step both boats were busy (none free, nobody deciding anything), that step was not counted, and the pointer fell behind** - the lag accumulates for the rest of the run, so over time the function ended up reading an old snapshot, several steps back, which could show demand that had ALREADY been picked up before the decision actually occurred. It was fixed (`simulador/src/metricas.py`, `decisiones_por_barco`) by computing the frame index directly from the decision's minute (instead of counting jumps), without assuming that each new decision is exactly one step after the previous one. Result, same run, before/after the fix:

| | Before (with the bug) | After (fixed) |
|---|---|---|
| Base, `veces_espero_con_demanda_local` | 22 cases (10 with no real explanation) | **1 case** (fleet coordination, legitimate) |
| Agent, `veces_espero_con_demanda_local` | - | **2 cases** (same reason) |

With the fix, almost all the "unexplained" cases disappear (verified case by case: crossing each "waited with local demand" against whether there was ANOTHER free boat at the same node in the same step - 12 of the 22 original cases were already explained this way even with the bug; with the fix, the remaining 10, which had no other free boat, turn out to be readings of an old frame, not real waits with demand available). The number that remains (1 for the base, 2 for the agent) is exactly what the coordination rule predicts: rare cases of two boats free at the same time at the same node.

**By origin-destination pair** - `output/por_par_comparado.csv`: the agent matches or exceeds the base on several pairs (`kleppesto->bryggen` 91.4% vs. 80.8%, `laksevag->sandviken` tied at 60%) and falls below on others (`kleppesto->sandviken` 70.5% vs. 100%, `bryggen->sandviken` 77.8% vs. 100%) - it does not win uniformly across all pairs, consistent with the base policy also being competent (`politica_base/README.md` section 2), not an easy floor to beat everywhere.

**Charts** - `output/heatmaps_comparados.html` (% served by pair, side by side) and `output/reward_por_semilla.html` (reward bars by seed).

---

## 3. Full detail, two seeds (1001 and 1004)

The same package of 5 metrics/charts produced by `politica_base/notebooks/01_escalon_1_verificacion.ipynb` (`simulador/README.md` section 5), run for both policies on TWO specific seeds - not just one - chosen because they are opposite cases: 1001 is one of the seeds where the base wins, 1004 is where the agent wins by the largest margin (section 2). All of this lives in a single function (`mostrar_detalle_semilla`, `05_comparacion_agente_vs_base.ipynb`), called once per seed, so as not to duplicate thirty cells for each one.

| | Seed 1001 - Agent | Seed 1001 - Base | Seed 1004 - Agent | Seed 1004 - Base |
|---|---|---|---|---|
| Mean wait | 11.8 min | 9.4 min | **18.2 min** | 27.7 min |
| % served (reference) | 85.2% (115/135) | 91.9% (124/135) | **91.7%** (198/216) | 84.3% (182/216) |
| Conservation | OK | OK | OK | OK |

**The two seeds tell opposite stories, on purpose.** In 1001 the base waits less (9.4 vs. 11.8 min) - one of the 4 cases (out of 5) where the aggregated reward favors the base (section 2). In 1004 the opposite happens, and by a lot: the agent waits 18.2 min against 27.7 for the base, and serves more people (91.7% vs 84.3%) - this is the seed where the base has its worst run of the ten (reward -4091, section 2), and here you can see why: at minute 460 the base's `K->B` queue has 33 people who have been waiting **37.9 minutes**, against a more evenly spread backlog on the agent's side at the same instant (`output/percentiles_semilla1004.csv`, `output/percentiles_semilla1001.csv`).

**Percentiles, each seed separately:**

| Metric | Percentile | 1001 - Agent | 1001 - Base | 1004 - Agent | 1004 - Base |
|---|---|---|---|---|---|
| Wait | p50 | 11.3 min | 8.5 min | 14.3 min | 23.5 min |
| Wait | p90 | 26.4 min | 20.3 min | 33.7 min | 52.8 min |
| Wait | p95 | 28.3 min | 25.3 min | **37.3 min** | **59.2 min** |
| System | p95 | 36.3 min | 31.3 min | **49.3 min** | **69.2 min** |
| System | max | 38.4 min | 31.3 min | 66.8 min | 69.7 min |

In 1004, the agent beats the base by almost **22 minutes in p95 of time in system** (49.3 vs. 69.2) - the same distribution-tail advantage seen in the aggregate (section 2.1), but in a specific run, not as an average of five.

**Charts** (`output/*_semilla{1001,1004}.html`, agent and base separately): `wait_profile_*`, `fleet_occupancy_*`, `pct_served_heatmap_comparado_*`, `reward_breakdown_*`, `backlog_by_pair_*` - same filenames as before, with the seed number at the end.

**Step-by-step visualization:** `animacion_{agente,base}_semilla{1001,1004}.gif` (the complete steps of each run), inspector at specific minutes, and the full interactive player.

- **Inspector - bug fixed.** The minutes requested are RELATIVE to the start of the run (`hora_ini_min + 40`, `hora_ini_min + 100` - 400 and 460 for `escalon_1`), not absolute minutes of the day. Previously `inspeccionar(40, ...)` / `inspeccionar(100, ...)` were requested plainly - since `escalon_1` starts at minute 360 (06:00), both fell well before the run began, and the inspector always showed the same first frame (minute 360) no matter which of the two was requested - it looked like it "showed nothing". With the corrected minutes, each call shows a real and distinct instant of the run (see the notebook output, each with real boats, queues and reward, different from one another).
- **Player - why it kept loading for a long time.** Computing each frame is fast (~35 ms measured), so that was not the cause. The most likely cause: `IntSlider` (the time bar) fires an update for EVERY intermediate position while being dragged with the mouse, not only on release - dragging quickly from one end of the run to the other can queue up dozens of redraws, and even though each one is fast, the queue takes minutes to drain. It was fixed (`simulador/src/visualizacion.py`, `reproductor_interactivo`): the bar now only updates on mouse RELEASE (`continuous_update=False`) - the automatic playback button `▶` is not affected (it advances one step at a time with its own timer, not by dragging). In addition, the map is now rendered to an explicit PNG instead of matplotlib's default "rich display" protocol, which is lighter and more predictable in remote notebooks. **It only works with a live Jupyter kernel** - a notebook that has already been executed and saved (like the ones `nbconvert` produces, which is how the ones in this project were run) has no kernel running, so the buttons will not respond there.

---

**What model this is:** `n_steps=512`, 300 000 timesteps / 585 PPO updates (`modelo_rl/output/modelo_final/`, `modelo_rl/README.md` section 4.2). `n_steps=90` (one full episode) was also tested as a better-justified alternative, on a budget of 150 000 steps - it generalized worse on the same 5 seeds despite training similarly (`modelo_rl/README.md` section 4.1); that specific comparison was not repeated when scaling up to 300 000, it is not used here.

## 4. How to run

```bash
cd comparacion/notebooks
jupyter nbconvert --to notebook --execute --inplace 05_comparacion_agente_vs_base.ipynb
```

It needs `modelo_rl/output/modelo_final/modelo_ppo.zip` and `vecnormalize.pkl` already saved (`modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb` run first).

---

## 5. Assumptions and limitations

- **The reward comparison uses the RL formula (no cap, `peso_movimiento=0.003`) for both policies**, not the production formula reported by `politica_base/` and stages 1-3 - necessary for the number to be comparable 1:1, but it means that "who wins on reward" depends on which formula was chosen to train the agent, it is not a neutral measure independent of that choice.
- **5 seeds is a small sample** - section 2 already shows that the aggregated result can be dominated by a single outlier case (seed 1004). It was not run on more seeds or on larger instances (`escalon_2`/`escalon_3`) - the agent was trained and evaluated solely on `escalon_1` (`modelo_rl/README.md` section 6).
- **"Winning" on % served is not the only reasonable criterion** - see section 3: in the seed shown in detail, the base serves more people, but the agent handles the worst case of that same run better at the reward level. Which of the two matters more depends on what one wants to optimize in a real deployment (average compliance, or protecting the worst case?) - an open question, not resolved by this project.
