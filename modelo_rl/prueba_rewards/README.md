# Reward test - eight formulations, compared against real metrics

**What this is, in one sentence:** eight conceptually distinct reward functions (not eight weights of the same formula), each trained with PPO under the same conditions, compared ONLY by what the simulator actually measures (time in system, movements), never by the value of each reward itself.

**A-D** are the first sequence (already closed - Little's objective, quadratic with tolerance, that same one plus potential shaping, and the production parameters as-is). **E-H** are four new variants, added afterward without touching A-D: pure linear time, pure quadratic time (unnormalized), incremental time per step, and a 95%/5% scalarized multi-objective (quadratic time + movement) - see section 2 for the complete formulas and section 7 for an important correctness finding discovered while implementing them (it also affects C, already trained).

This experiment is self-contained: it does not modify `simulador/`, `politica_base/`, or the already-closed notebooks/README of `modelo_rl/`. It reuses the engine (`simulador/src/`), the base policy (`politica_base/src/politica_base.py`), the fresh-demand wrapper (`modelo_rl/src/entrenamiento.py`), and the metrics/charts (`simulador/src/metricas.py`, `visualizacion.py`) as-is - none of that is reimplemented.

---

## 1. How it is organized

```
modelo_rl/prueba_rewards/
├── README.md                                  # this file
├── config/instance.yaml                       # definition of A-H, timesteps, seed
├── src/
│   ├── recompensas_alternativas.py            # calcular_recompensa_{A,C,E,F,G,H}, potencial(), suma_tiempo_activo()
│   │                                            # (B and D: the ALREADY EXISTING function from recompensa.py, reused)
│   └── entorno_recompensa_intercambiable.py    # EntornoRecompensaIntercambiable -- plugs in any of the 8
├── notebooks/
│   ├── 00_verificar_formulas.ipynb             # Step 1: the 8 formulas, one example step, sanity checks
│   ├── 01_entrenar_cuatro.ipynb                # Step 2: 8 PPO trainings, same hyperparameters
│   └── 02_comparar_resultados.ipynb            # Step 3: evaluation against base, real metrics, charts
└── output/
    ├── modelos/{A,B,C,D,E,F,G,H}/              # modelo_ppo.zip, vecnormalize.pkl, monitor.monitor.csv
    └── comparacion/                            # final tables and charts
```

---

## 2. The eight rewards

The eight are computed on the same `EstadoSimulacion` that `env.py` already builds at every step - none of them reimplements the simulation mechanics, only how that state is translated into a number changes.

### 2.A - Pure objective, derived from Little's identity

$$r_t = -\left[\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot \Delta t \;+\; w_{\text{move}}^A \cdot m_t\right], \qquad w_{\text{move}}^A \approx 0.003 \cdot \Delta t = 0.006$$

No tolerance, no cap, no square: each active person (waiting or aboard, not yet delivered) costs, per step, its size multiplied by the duration of the step (Δt = 2 min).

**Why this is "the real objective", not a proxy.** The identity behind Little's Law (L = λW) is a double-counting argument: the area under the curve of "how many people are in the system at each instant," integrated over time, is exactly equal to the sum of the times each individual person spent in the system (each person "contributes" their own time in system to the area, regardless of when they entered or left). In its discrete version:

$$\sum_t N(t) \cdot \Delta t \;=\; \sum_{i} \text{tamaño}_i \cdot W_i$$

where $N(t)$ is the active people at step $t$ and $W_i$ is the time person $i$ spent in the system. The term on the left is, except for the sign, exactly what reward A accumulates over an episode. Maximizing the accumulated reward A is, by this identity, directly minimizing $\sum_i \text{tamaño}_i \cdot W_i$ - the total time in system, the metric that really matters (`espera_media_min`/`sistema_medio_min` in `metricas.py`) - without going through any tolerance or intermediate penalty curve.

### 2.B - Convex weighted with tolerance, no cap

The formula already used in the rest of `modelo_rl/` (winner of the already-closed A/B/C1/C2 sequence - see `modelo_rl/README.md` section 3), reused as-is via `recompensa.calcular_recompensa`:

$$r_t = -\left[\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot \left(\frac{\max(0, s_{i,t}-12)}{18}\right)^2 \;+\; 0.003 \cdot m_t\right]$$

Serves as a known reference: it penalizes only the excess over tolerance (12 min), quadratically (grows faster the longer it takes), with no per-person cap.

### 2.C - B + *potential-based reward shaping* (Ng, Harada & Russell 1999)

$$r_t = r_t^B + \underbrace{\big(\gamma \cdot \Phi(s_{t+1}) - \Phi(s_t)\big)}_{\text{shaping term}}, \qquad \Phi(s) = -\eta \cdot N(s)$$

with $N(s)$ = active people (weighted by size) in state $s$, and $\eta$ (`eta_potencial`) configurable.

**Why this form preserves the optimal policy - the theorem of Ng, Harada & Russell (1999).** Given an MDP $M$ with reward $R$, and a modified MDP $M'$ with reward $R' = R + F$, the authors prove that $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$ for any potential function $\Phi: S \to \mathbb{R}$ is, under mild conditions, **necessary and sufficient** to guarantee that every optimal policy of $M'$ is also optimal in $M$ (and vice versa) - for ANY base reward $R$, not just this project's. The proof shows that $Q^*_{M'}(s,a) = Q^*_M(s,a) - \Phi(s)$: shaping shifts the value of ALL actions in the same state by the SAME constant $\Phi(s)$ (it does not depend on $a$), so the ordering among actions - and therefore the optimal action - does not change in any state.

**What it is expected to contribute, then, if it does not change the optimum.** Learning speed, not the destination. Without shaping, the only signal comes from the discomfort penalty, which accumulates slowly and sparsely (it is only noticeable well after someone has been waiting a while). With shaping, every step that reduces the queue gives an immediate reward ($\Phi$ goes up, less negative), and every step that lets it grow gives an immediate penalty - a dense signal, correlated with progress, available from the first step. The theoretical guarantee concerns the optimum with infinite training; with this experiment's short budget (30 000 timesteps, same as the other three), the empirical question is whether that denser signal actually helps convergence happen faster - exactly what this experiment measures.

### 2.D - Current code (production parameters)

$$r_t = -\left[\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot \min\!\left(1.0, \left(\frac{\max(0,s_{i,t}-12)}{18}\right)^2\right) \;+\; 0.1 \cdot m_t\right]$$

Exactly `calcular_recompensa()` with the values currently living in `simulador/config/instance.yaml` → `recompensa:` (cap=1.0, `peso_movimiento`=0.1) - the ones that govern the base policy and escalones 1-3. **Real coincidence, not fabricated:** the already-closed experiment sequence (A/B/C1/C2, in `modelo_rl/notebooks/02_secuencia_experimentos_reward.ipynb`) never used `peso_movimiento=0.1` nor left the cap active - these production values had never been trained with RL until this experiment. It serves as a "poorly scaled" contrast (low cap + expensive movement) without needing to invent a fifth variant.

**Note (2026-09-17):** D's config was updated by the user's decision to a purely quadratic, tolerance-free time formulation (`tolerancia_incomodidad_min=0`, `sobrante_normalizador_min=12`, effectively infinite cap, `peso_movimiento=0`) - $r_t = -\sum_i \text{tamaño}_i \cdot (s_{i,t}/12)^2$, with no movement term. D was retrained with this config at 150 000 timesteps (versus the original 30 000) - the results in section 5 reflect this version, not the one described above in the mathematical formula (which documents the *original* D of this experiment, with cap=1.0 and `peso_movimiento`=0.1). See `config/instance.yaml` for the current values.

### 2.E - Pure linear time (unnormalized, no movement)

$$r_t = -\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot T_{i,t}, \qquad T_{i,t} = t - \text{minuto\_llegada}_i$$

The TOTAL time each active person has been in the system at this instant (not a fixed cost per step, like A - A charges `Δt` per active person each step; E charges the complete accumulated time, which grows with every step that person remains active). No tolerance, no normalizer, no cap, no movement term - no parameter besides time itself, deliberately, to isolate the effect of the functional form (linear vs. quadratic) without any other term contaminating it.

### 2.F - Pure quadratic time (unnormalized, no movement)

$$r_t = -\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot T_{i,t}^2$$

Same as D but without dividing by any normalizer (D uses $(T_i/12)^2$) - a raw version, to isolate whether normalizing matters once `VecNormalize` is already active in PPO (which normalizes the complete reward signal, with its own running statistics). No tolerance, no cap, no movement. Motivation: evaluate whether increasingly penalizing long times (quadratic > linear for $T>1$) improves agent behavior relative to E.

### 2.G - Incremental time (delta T per step)

$$r_t = -\sum_{i} \Delta T_{i,t}$$

The INCREASE in time in system that occurred during this 2-min step - not the total accumulated time (unlike E). It is computed as:

$$\Delta T_{\text{total}} = \big(S(s_{t+1}) - S(s_t)\big) + \sum_{i \in D_t} \text{tamaño}_i \cdot (t_{t+1} - \text{minuto\_llegada}_i), \qquad S(s) = \sum_{i \in \mathcal{A}(s)} \text{tamaño}_i \cdot (t(s) - \text{minuto\_llegada}_i)$$

where $D_t$ are the units delivered IN this step (no longer in $\mathcal{A}(s_{t+1})$, so the subtraction $S(s_{t+1})-S(s_t)$ loses them - the second term adds them back). This formula correctly handles the three cases that can occur in a step:

- **Remains active, not delivered:** its contribution is exactly $\Delta t$ (2 min) - it was already active before and remains active after, the whole step counts.
- **Arrives mid-step:** its contribution is only the fraction since it arrived ($t_{t+1} - \text{minuto\_llegada}_i < \Delta t$) - it was not in the system before arriving, so $S(s_t)$ does not include it and $S(s_{t+1})$ does, for its partial time.
- **Is delivered this step:** its contribution is $t_{t+1} - \max(t_t, \text{minuto\_llegada}_i)$ - only the time of THIS step, not its full history (the algebra above reduces exactly to this; see verification in `00_verificar_formulas.ipynb`).

No double counting or leakage: the sum of $\Delta T_{\text{total}}$ over a complete episode is identical, bit for bit, to the total time in system of each person (served or active at the close) - verified exactly in `00_verificar_formulas.ipynb` (difference 0.000000000 over a 90-step episode, seed 1001).

### 2.H - Scalarized multi-objective, 95% quadratic time + 5% movement

$$r_t = -\big[w_T \cdot J_T + w_M \cdot J_M\big], \qquad w_T = 0.95,\ w_M = 0.05$$

$$J_T = \sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot T_{i,t}^2 \ (\text{same as F, unnormalized}), \qquad J_M = m_t \ (\text{boats navigating this step, same definition as A/B/C/D})$$

Explicit weighted combination of two objectives (time and movement), intended to ask the agent to prioritize time over movement in a nominal 95/5 proportion. **See section 8 for the real, measured scale problem between $J_T$ and $J_M$ with these weights - documented, not manually corrected.**

---

## 3. Training - same conditions for all 8

| | Value | Where it comes from |
|---|---|---|
| Instance | `escalon_1` (2 boats, morning window) | `simulador/config/instance.yaml` |
| `total_timesteps` | 150 000 | `prueba_rewards/config/instance.yaml` |
| `semilla_entrenamiento` | 123 | `prueba_rewards/config/instance.yaml` |
| `VecNormalize` (obs and reward) | active | `simulador/config/instance.yaml` → `agente.hiperparametros` |
| `ent_coef` | 0.01 | same |
| `learning_rate` | 0.0003 | same |
| `n_steps` | 512 | same (the validated one - see `modelo_rl/README.md` section 4.1) |

The ONLY thing that changes across the 8 runs is the reward (`EntornoRecompensaIntercambiable`, `tipo_recompensa="A".."H"`). **Historical note:** `total_timesteps` was originally 30 000 (to iterate quickly); it was raised to 150 000 (292 PPO updates, same as the final agent - `modelo_rl/README.md` section 4.2) before adding E-H, and A/B/C/D were retrained at this longer budget - the numbers in section 5 are from that run, not from the 30 000 one (which is no longer documented, to avoid confusion with the current one).

**Actual measured time, by type** (each `output/modelos/{A..H}/metadata_entrenamiento.json` saves it separately, with `time.time()` around `model.learn(...)` - fixed data, not an estimate):

| Type | PPO updates (150 000/512) | Measured time |
|---|---|---|
| A | 292 | 14.7 min |
| B | 292 | 14.9 min |
| C | 292 | 12.9 min |
| D | 292 | 15.3 min |
| E | 292 | 9.3 min |
| F | 292 | 8.9 min |
| G | 292 | 9.2 min |
| H | 292 | 9.2 min |

E-H train noticeably faster than A-D with the same number of updates - consistent with none of the 4 new ones having the shaping/potential term nor the threshold/cap logic of B/C/D (less work per step in computing the reward, not in the simulator itself, which is the same for all 8).

---

## 4. Anti-Goodhart validation

The 8 rewards have different scales by design (A/E sum raw minutes, B/C/D/F/H use quadratic curves - some normalized and capped, others not, G sums time increments per step) - comparing their absolute values against each other means nothing, and doing so would be exactly the mistake sometimes known as "reward hacking"/Goodhart's Law misapplied (optimizing a proxy metric until it stops reflecting the real objective). That is why the Step 3 comparison uses ONLY the simulator's real metrics, reusing `metricas.reporte_completo` without modifying it:

- **Time in system, mean, maximum, and percentiles (p50/p90/p95)** - what the eight rewards, in theory, try to reduce.
- **Mean wait.**
- **Total movements** - the cost that is being avoided from overpaying.
- **Mean occupancy.**
- **% served** - for reference only, never as the main criterion: the simulator never loses anyone (nobody ever leaves), so that number depends partly on where the evaluation window cuts off, not only on the policy (same argument as in `comparacion/README.md`).

---

## 5. Results

150 000 timesteps per type, 5 evaluation seeds (1001-1005), conservation OK across all 8 runs + base.

**Note (2026-09-17):** this table replaces an earlier version calculated at 30 000 timesteps (initial exploratory budget) - A/B/C/D were retrained at 150 000 before adding E-H, so these A-D numbers no longer match those of the first version of this README. The reading in section 6 (Conclusion) was written based on the 30 000 run and **was not rewritten** for this longer run - its qualitative claims (e.g. "A is the worst in mean time", "D did not collapse") remain true with the numbers below, but the exact numeric values quoted in the text are the old ones. It is left as is (rewriting that section was not requested) and the discrepancy is documented rather than left implicit.

| Type | Mean system (min) | Max system (min) | System p95 (min) | Mean wait (min) | Wait p95 (min) | Movements | Mean occupancy | % served (ref.) |
|---|---|---|---|---|---|---|---|---|
| A | 37.17 | 122.07 | 87.95 | 22.45 | 59.33 | 164 | 3.00 | 78.1% |
| B | 29.02 | 60.77 | 54.62 | 17.06 | 37.46 | 191 | 3.50 | 88.0% |
| C | 29.25 | 74.07 | 56.77 | 20.21 | 44.77 | 193 | 3.39 | 84.5% |
| D | **25.73** | **51.92** | **51.10** | **16.21** | **41.34** | 174 | 3.21 | 84.5% |
| E | 41.06 | 123.13 | 115.10 | 24.41 | 62.55 | 172 | 2.85 | 74.8% |
| F | 33.05 | 81.13 | 69.46 | 22.78 | 57.46 | 183 | 3.00 | 75.9% |
| G | 33.46 | 109.74 | 74.55 | 20.72 | 61.71 | 169 | 3.02 | 74.4% |
| H | 37.15 | 103.74 | 76.55 | 24.94 | 64.43 | **194** | 3.20 | 81.4% |
| base | 26.05 | 69.74 | 58.77 | 16.42 | 46.06 | 155 | 3.50 | 88.0% |

Full table with more percentiles (p50/p90/p95 of system and wait): `output/comparacion/tabla_comparativa.csv`.

**Among A/B/C/D (analysis already closed, section 6), the ranking by sum of ranks (mean + maximum + movements) still gives D as the winner** with this longer run - see `02_comparar_resultados.ipynb`, cell "¿Cuál ganó?".

**E-H, no conclusion on which is best (explicitly requested this way):** with this single run of 150 000 timesteps and 5 evaluation seeds, no winning reward is declared among the 8 - the numbers above are left for a later joint analysis. Some specific observations, without being a ranking:

- **G (incremental time) has the lowest mean and p50 time among E-H** (33.46 min mean, 25.18 min p50 - very close to D's p50, 23.36), but a p95 considerably higher than its own median (74.55 min) - suggesting a long-tailed distribution: most passengers are served quickly, but a subset waits much longer.
- **E (pure linear time) has the worst maximum time and p95 of the whole group** (123.13 min max, 115.10 min p95 - comparable to A's worst case). Consistent with the same lack of penalty structure already documented for A in section 6: with no curve that penalizes more strongly those who have already waited a long time, there is no additional incentive to prioritize the most delayed cases.
- **F (pure quadratic time) improves over E on almost all metrics** (33.05 vs 41.06 mean, 81.13 vs 123.13 maximum) - consistent with the motivation of testing whether increasingly penalizing long times helps relative to a linear penalty.
- **H (95/5 multi-objective) behaves almost indistinguishably from F** (37.15 vs 33.05 mean, 103.74 vs 81.13 maximum - same order of magnitude, no clear improvement in movements despite the 5% nominal weight placed there: 194 movements, the highest value of the 8). Consistent with the finding in section 8: with these weights, the movement term is arithmetically insignificant compared to the time term, so H in practice optimizes almost the same thing as F alone.

Charts: training curves for all 8 (`output/comparacion/curvas_entrenamiento.html`), A-D+base metric comparison (`comparacion_metricas.html`), full A-H+base comparison with percentiles (`comparacion_metricas_completa.html`), and full detail (heatmap per pair + wait profile + fleet occupancy) for the two tied among A-D, C and D (`heatmap_por_par_{C,D}.html`, `perfil_espera_{C,D}.html`, `ocupacion_flota_{C,D}.html`).

---

## 6. Conclusion

**No result contradicts the theory - but none confirms the naive reading of "A is the real objective, so it should win".** The four give a coherent reading, read together with what each formula actually rewards:

**D (the production one) did not collapse - and that, by itself, is a result.** This project's original narrative (`modelo_rl/README.md`, historical diagnosis) is that these same parameters (low cap, high `peso_movimiento`) produced a degenerate, nearly paralyzed policy. That did not happen here: D trained with `VecNormalize` active (mandatory for the comparison among the 4 to be fair), and `VecNormalize` is precisely the fix that, in that earlier diagnosis, corrected the collapse. With normalization in place, the raw imbalance between the discomfort and movement weights matters much less than it did without it - D not only does not collapse, it ends up with the best mean time and the best mean wait of the four. This does not invalidate the original diagnosis (the collapse was real, under those conditions) - what it shows is that the underlying cause was the lack of normalization, not the weight itself, something this experiment makes much clearer than before.

**A (the "true" objective by Little's identity) was the worst of the four in mean time - exactly what it should minimize best in the limit.** The explanation does not contradict the theory, it completes it: Little's identity guarantees that minimizing A is minimizing total time in system *in the limit*, with sufficient training - it says nothing about how easy it is to learn that signal with a short budget. A charges the same for every minute any person is active, regardless of whether they are already past tolerance or just arrived - there is no "cheap" zone near the tolerance limit that the agent can exploit quickly, unlike B/C/D, where someone served within 12 minutes costs nothing. With only 30 000 steps, that lack of structure apparently outweighs the theoretical advantage of optimizing the correct objective - a real distinction between "the correct objective function" and "the objective function that is easy to learn quickly", and a concrete reason not to assume that the most direct formulation is automatically the best practical choice.

**C (B + potential shaping) delivers exactly what Ng et al.'s theorem promises: it does not change what is optimal, it helps find it better with a short budget - and here this shows up in the worst case and in efficiency, not in the average.** Shaping gives the agent an immediate signal for reducing the queue, at every step, instead of only the delayed discomfort penalty - with 30 000 steps (a fifth of the final budget), that dense signal appears to have helped the policy find a more deliberate dispatch pattern: considerably fewer movements (72, almost half of A/B/D) without sacrificing the worst case (104.75 min, the best of the four) - it protects exactly what a cap *without* shaping (like B) fails to protect (B has the worst maximum of the four, 139.93 min). This is consistent with the theoretical guarantee: C's optimal policy is the same as B's (same underlying `r_B`), so any observed difference is a difference in how fast/well a good policy is reached with the same training budget, not a difference in which policy is "correct".

**General reading:** the choice of reward matters, and not always in the direction that theoretical intuition would suggest at first glance - exactly the reason this project compares on real metrics and not on the value of each reward (section 4). If the goal of a real deployment were to minimize the typical case, D is the defensible choice with this evidence; if the goal is to protect the worst case and be efficient in movements, C is - neither of the two readings is "the answer", they are two legitimately different service objectives.

---

## 7. Technical note: aliasing bug discovered while implementing G (also affects C, already trained)

When implementing G, "what the world looked like before this step" was needed - the same requirement C already had (to compute $\Phi(s_t)$). The first implementation of both saved the complete `EstadoSimulacion` object from the previous step (`self._estado_previo = estado_despues`) to read it in the next step. **This had a real, silent bug, that had been present since C was first trained:**

`EstadoSimulacion.colas`/`.barcos` (`simulador/src/estado.py`) are built in `env.py` → `_construir_estado()` passing `self.colas`/`self.barcos` **by reference**, not by copy:

```python
EstadoSimulacion(t_actual_min=self.t_actual_min, dia_semana=self.dia_semana,
                  barcos=self.barcos, colas=self.colas, atendidas_historico=self.atendidas_historico)
```

An `EstadoSimulacion` saved from a previous step **is not frozen**: its `colas`/`barcos` are literally the same objects the simulator keeps mutating at every subsequent step, so reading it later returns the simulator's CURRENT state, not the historical one from the moment it was saved. Only `t_actual_min` (a simple float, copied by value) remained correct.

**Confirmed with a direct test:** `estado_0` was saved at `env.reset()`, 8 more steps were run, and `estado_0` was read again - it had 19 active units (it should have 0, the empty initial state) while `estado_0.t_actual_min` correctly remained at 360.0 (the starting minute). `id(estado_0.colas) == id(env.colas)` gave `True` - literally the same dictionary.

**Real impact:** for G, this produced incorrect reward values at every step with new arrivals (confirmed with a conservation identity that should have given a difference of 0 and did not). For C, the shaping term $\gamma\Phi(s_{t+1})-\Phi(s_t)$ was computed with a $\Phi(s_t)$ that was actually $\Phi$ of the CURRENT state (post-mutation), not the one from before the step - **C has already been trained and its results are already reported in section 5 under this bug.** The policy invariance theorem of Ng et al. (section 2.C) still guarantees that C's optimal policy remains the same as B's *if* the term were a valid potential-based shaping term - with the bug, the term actually injected is not $\gamma\Phi(s')-\Phi(s)$ but a related yet different quantity (computed with two readings of the same mutating state), so that guarantee does not strictly apply to the already-trained C. C is not being retrained retroactively (it was not requested, and it would violate "do not modify what already exists") - this finding is left explicitly documented instead of hidden, and it remains a decision for later whether it is worth retraining C with the fix.

**The fix (used in C and G from now on, and in every new run of both):** instead of saving the `EstadoSimulacion` object, a **scalar** is computed and saved (`potencial_antes` for C, `s_tiempo_antes` for G, via the `suma_tiempo_activo()` helper) at the exact moment the state is still fresh - a float is immune to future mutations because it does not share memory with anything. Implemented in `EntornoRecompensaIntercambiable._actualizar_escalares_previos()`, called at the end of every `reset()`/`step()`. Verified: with this fix, G's conservation identity gives an exact difference of 0.000000000 over a complete episode, and `C` with `eta_potencial=0` again matches `B` bit for bit over the 90 steps of a test episode (see `00_verificar_formulas.ipynb`).

## 8. Scale problem in H, measured

`H` weights $J_T$ (quadratic time) and $J_M$ (movement) with nominal weights 95%/5%. The real magnitude of each term was measured, unweighted, by running the base policy for a complete episode (seed 1001, `00_verificar_formulas.ipynb`):

| Term | Accumulated (full episode) | Average per step |
|---|---|---|
| $J_T$ (quadratic time) | 229 567.00 | 2 550.74 |
| $J_M$ (movement) | 96.00 | 1.07 |

**Ratio $J_T/J_M \approx 2391\times$ before weighting.** After applying $w_T=0.95$/$w_M=0.05$, the real contribution to the weighted sum is:

| Term | Weighted contribution | % real share of the signal |
|---|---|---|
| Time (95% nominal) | 218 088.65 | **99.9978%** |
| Movement (5% nominal) | 4.80 | **0.0022%** |

**The nominal 95/5 does not, in practice, represent a 95%/5% contribution - it effectively represents 100%/0%.** The movement term is arithmetically insignificant compared to the time term with these weights: $J_T$ is already ~2391 times larger than $J_M$ before weighting, so even with only a 5% nominal weight it still completely dominates. By explicit decision, this scale problem **is not corrected by hand** (the weights are not rescaled nor is $J_M$/$J_T$ normalized to force a real 95/5) - it is documented as is, because it is part of what this experiment measures: if a multi-objective is defined with nominal weights without first checking that both terms are on comparable scales, the practical result can be radically different from the nominal intent. H, as currently configured, is in practice nearly indistinguishable from F (pure quadratic time) in terms of what it optimizes - an observation to keep in mind when reading its results in section 5.

## 9. Bibliography

- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Little, J. D. C. - the L=λW identity. Reference notes: Columbia University, IEOR 4404, *Little's Law*. http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-LL.pdf ; MathWorld, *Little's Law*. https://mathworld.wolfram.com/LittlesLaw.html
- AdaPool (2021). *Adaptive Fleet Rebalancing via Reinforcement Learning.* arXiv:2104.00203. https://arxiv.org/abs/2104.00203
- RAST-MoE-RL (2025). arXiv:2512.13727 (reward hacking / anti-gaming in RL). https://arxiv.org/abs/2512.13727
- *Multi-Objective Rebalancing for Vehicle Sharing Systems* (2020). arXiv:2007.06801. https://arxiv.org/abs/2007.06801
- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd ed.). http://incompleteideas.net/book/the-book-2nd.html
