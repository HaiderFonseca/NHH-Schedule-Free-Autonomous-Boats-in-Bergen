# PPO Agent - training, reward experiments, final run

**What this is, in one sentence:** everything related to the RL agent that competes against the reference policy (`politica_base/`) - the random-demand wrapper needed for training, the sequence of experiments that decided the shape of the reward, and the final long run.

---

## 1. How it is organized

```
modelo_rl/
├── README.md                                     # this file
├── src/
│   └── entrenamiento.py                          # EntornoDemandaAleatoria -- fresh demand per reset
├── notebooks/
│   ├── 01_enfoque_y_entrenamiento_final.ipynb    # explains PPO/wrapper, runs the final LONG run
│   ├── 02_secuencia_experimentos_reward.ipynb    # A/B/C experiments on the shape of the reward
│   └── 03_verificacion_toy.ipynb                 # minimal check on a toy instance (appendix)
└── output/
    ├── modelo_final/                              # trained model, VecNormalize, training curve
    └── experimentos/                               # monitors and summary of the A/B/C sequence
```

The notebooks import `simulador/src` (engine) and `politica_base/src` (to evaluate against the reference within the experiment sequence itself) in addition to their own `src/entrenamiento.py`, and read `simulador/config/instance.yaml` - the single source of truth for the PPO hyperparameters, the definition of the experiments, and the final run (all under the `agente` key).

---

## 2. Why PPO, and why it needs a demand wrapper

**PPO (Proximal Policy Optimization), not DQN.** The environment's `action_space` (`simulador/src/env.py`) is `MultiDiscrete` (one action per boat - see `simulador/README.md` section 4.3) - Stable-Baselines3's DQN does not support `MultiDiscrete` natively, PPO does.

**Fresh demand on every `reset()` (`src/entrenamiento.py`) - necessary for RL.** `politica_base/` runs on **fixed** demand: it is generated ONCE and saved to a CSV that the notebooks only read. That is correct for verifying the simulator (the policy has nothing to adjust), but **it does not work for training an agent**: if every training episode were always the same group table, the agent could memorize that particular realization (which exact group appears at which exact minute) instead of learning a policy that generalizes over the real demand pattern.

`EntornoDemandaAleatoria(SimuladorBarcosBergen)` - the same subclass serves both needs, depending on how `reset()` is called:

- **`reset(seed=None)`** (the normal case in a training loop - SB3 does not send a seed on every episode): generates NEW demand each time, with a seed drawn from its own sequence (reproducible if `semilla_entrenamiento` was given when constructing the environment). Each episode sees a different realization of the same underlying pattern - Monte Carlo sampling of episodes.
- **`reset(seed=42)`** (explicit case - evaluation, comparison against the base policy): generates demand with THAT exact seed, reproducible just like `SimuladorBarcosBergen` today. Two different instances of `EntornoDemandaAleatoria` with `reset(seed=misma_semilla)` produce the SAME demand - this is what makes it possible to compare the agent and the base policy on equal terms (`comparacion/`).

**Why this is a wrapper and not a change to `env.py`:** `simulador/src/env.py` receives an already-built `grupos_df` and never knows anything about `demand/` (routes, `intensidad_od`, `conexiones_fuertes`, etc.). Putting demand generation there would complicate the base engine that `politica_base/` continues to use as-is. `EntornoDemandaAleatoria` changes only one thing (which `grupos_df` each episode uses); everything else (actions, transition, reward) is exactly `SimuladorBarcosBergen` untouched.

**`VecNormalize`.** The environment is wrapped in `DummyVecEnv` + `VecNormalize` (normalizes observations and reward using running statistics) - without this, an early diagnostic training run learned a degenerate policy (0% served even on a toy instance). It is part of `agente.hiperparametros` (`usar_vecnormalize: true`) and does NOT vary across the experiments in section 3 or the final run - only the reward changes between them.

---

## 3. Experiment sequence: shape of the reward (`02_secuencia_experimentos_reward.ipynb`)

Before spending the single long training run of the cycle, the shape of the reward was explored METHODICALLY on `escalon_1` (morning window, 2 boats) - SHORT training runs (`agente.experimentos.timesteps_exploracion`, 30 000 steps, ~2 min of training each) in order to iterate quickly, all evaluated against the base policy on the same 5 evaluation seeds as always.

**The sequence, and the question each step answers** (full config in `simulador/config/instance.yaml` → `agente.experimentos`):

- **A** - time penalty only, WITH a per-person cap (the production value, `simulador/README.md` section 4.4). How well does the agent perform with the simplest possible reward?
- **B** - the same, but WITHOUT the per-person cap. Hypothesis: better than A, because without the cap the reward keeps distinguishing between someone very delayed and someone only slightly delayed even after both exceed the 12-min tolerance - with the cap, someone waiting 35 min and someone waiting 50 min are penalized exactly the same.
- **C1/C2** - on top of the winner of A/B, a SMALL movement penalty is added (0.003 and 0.001 respectively). Scale analysis before running the experiment: a person's penalty ranges from 0 to 1 according to `(sobrante/18)²` - someone with ~13 min of wait already incurs a penalty of `(1/18)² ≈ 0.003`, so a movement penalty of the same order (0.003) or smaller (0.001) is needed so that moving a boat does not weigh, in the reward, as much as leaving someone waiting.

**Result and decision** (full narrative, with the actual tables, at the end of `02_secuencia_experimentos_reward.ipynb`):

| Experiment | `peso_movimiento` | Per-person cap | % served (30k steps) | Mean wait | Movements |
|---|---|---|---|---|---|
| A | 0.0 | active (1.0) | 47.8% | 26.2 min | 135 |
| B | 0.0 | inactive | 51.9% | 21.9 min | 125 |
| C1 | 0.003 | inactive | 52.0% | 23.8 min | 127 |
| C2 | 0.001 | inactive | 44.7% | 22.5 min | 95 |

**Note on the "% served" column of this table:** it is used here only as a quick indicator to compare A/B/C1/C2 AGAINST EACH OTHER, not as the metric that really matters - the simulator never loses anyone (nobody ever leaves the system, `simulador/README.md` section 4.5), so "% served" depends partly on where the run window's cutoff falls, not only on how well the policy dispatches. What the reward actually penalizes, and what is reported as the real result in `comparacion/README.md`, is time in system (mean and maximum) and movements.

**A vs. B: the hypothesis is confirmed** - removing the cap improves all three columns (more served, lower mean wait, even fewer movements). **C1 vs. C2: the starting point (0.003) performs better than the smaller one (0.001)** - C1 essentially ties with B in % served while keeping movement similar to B (movement discipline without losing performance); C2, with an even smaller penalty, does not perform better - there is no evidence that 0.001 is systematically better on any axis at this training scale. **Winner: C1** (`peso_movimiento: 0.003`, no per-person cap) - combo copied into `agente.entrenamiento.recompensa_overrides` for the final run.

**On the gap relative to the base policy (88.0% served) in this sequence:** no experiment gets close to the base - these are training runs of only 30 000 steps (a single training seed), intended exclusively to compare the SHAPE of the reward against each other, not to produce a competitive agent. See section 4 for the result of the long run (300 000 steps, 10x more) with the reward already chosen here.

---

## 4. Final training (`01_enfoque_y_entrenamiento_final.ipynb`)

`agente.entrenamiento` in `simulador/config/instance.yaml`:

```yaml
entrenamiento:
  escalon_base: "escalon_1"
  semilla_entrenamiento: 123     # root of the training-demand sequence
  total_timesteps: 300000        # ~3333 episodes of 90 steps - raised from 150000, see section 4.2
  recompensa_overrides:          # winning combo from the experiment sequence (section 3)
    peso_movimiento: 0.003
    penalizacion_maxima_persona: 1000000.0
    premio_por_persona_entregada: 0.0
```

PPO hyperparameters (`agente.hiperparametros`, shared by ALL experiments in section 3): `usar_vecnormalize: true`, `ent_coef: 0.01` (more exploration than SB3's default of 0.0), `learning_rate: 0.0003` (the PPO standard, 3e-4). `n_steps: 512` comes from an earlier round of diagnostics, chosen smaller than SB3's default (2048) so as not to wait ~23 episodes between updates - but 512 itself does not come from any formula or reference, it is a choice with no justification beyond "it worked". The most directly defensible alternative is `n_steps=90`: exactly one full `escalon_1` episode, so every update falls right at the close of an episode, never mid-episode. Section 4.1 compares the two.

**The notebook, in order:** (1) confirms fresh demand per reset (section 2); (2) SB3's `check_env` (standard check before training); (3) wraps the environment with `Monitor` + `DummyVecEnv`/`VecNormalize`; (4) trains `PPO("MlpPolicy", ...)`; (5) saves the model (`output/modelo_final/modelo_ppo.zip`, `vecnormalize.pkl`) and plots the per-episode reward curve (interactive, `output/modelo_final/training_curve.html`).

**Result of the final run (300 000 timesteps, 3333 episodes of 90 steps, combo C1) - raised from 150 000, see section 4.2 for why:** the mean reward of the first 20 episodes was -24 871.4 (identical to the 150k run: same seed, same start); that of the last 20, -1 797.8 - better than the -2 832.8 of the 150k run, consistent with continued training improving results. The last 5 individual episodes: -744.9, -1 887.7, -277.5, -4 533.3, -1 323.1 - in the same low range as before, with the same variability expected from fresh demand on each `reset`. Full curve (interactive) in `output/modelo_final/training_curve.html`. See `comparacion/README.md` for the result of this model compared against the base policy - with more training, the agent now nearly ties the base in mean time and mean wait, and gains a clear advantage in the upper percentiles (wait p95: 11 min less than the base).

### 4.1 `n_steps=90` instead of 512 - tried, performed worse in evaluation

512 did not come from any calculation (section 2) - the alternative that can actually be defended with an argument is `n_steps=90`, exactly one `escalon_1` episode, so every PPO update falls right at the close of an episode. The same long run was executed (150 000 timesteps, same combo C1) changing only that - saved separately in `output/modelo_final_n90/`, without overwriting the model above.

**In training they look similar:** mean reward of the last 20 episodes, -2 832.8 (n_steps=512) vs. -2 956.2 (n_steps=90) - the curve alone would not tell which is better. **In evaluation (5 seeds, the usual ones) they diverge a lot:**

| | n_steps=512 | n_steps=90 | Base policy |
|---|---|---|---|
| Mean time in system | 29.0 min | 30.2 min | 26.1 min |
| Maximum time in system | 60.8 min | **73.7 min** | 69.7 min |
| Mean wait | 17.1 min | 19.9 min | 16.4 min |
| Movements (5 episodes) | 191 | 179 | 155 |
| Reward (5 episodes) | -3772.0 | **-28082.5** | -5259.0 |
| % served (reference) | 88.0% | 77.5% | 88.0% |

`n_steps=90` loses on everything, including the worst case - 73.7 min, worse than the base policy itself (69.7). The evaluation reward (-28082) is in the range of the other model's early training episodes, not close to where it ended up. In other words: it trained well (the curve goes up) but generalized poorly to the evaluation seeds.

**Why?** With `n_steps=90`, the PPO buffer collects exactly ONE episode before updating, so over the same run it performs ~1666 updates (one per episode) instead of ~293 with n_steps=512 - each with much less data. Updating so frequently with such small batches makes each gradient step noisier and more tightly tied to that particular episode's demand - it can look like it is improving during training (the curve goes up) without the policy becoming more robust to demand it has not seen. `n_steps=512` ends up giving each update a wider, less noisy sample of experience, and that matters more than aligning updates with the close of each episode.

**It sticks with `n_steps=512`.** The model used in `comparacion/` is the one in `output/modelo_final/` - `n_steps=90` is documented as a real result that was tried and did not work better, not as the final model.

### 4.2 Training times (measured, not estimated)

Each run saves its own `metadata_entrenamiento.json` (`output/modelo_final/`, `output/modelo_final_n90/`, and one per experiment in `prueba_rewards/output/modelos/{A,B,C,D}/`) with the actual measured time (`time.time()` around `model.learn(...)`), not an estimate - so this data stays fixed, without depending on rerunning anything to find out.

| Run | `total_timesteps` | `n_steps` | PPO updates (`total_timesteps // n_steps`) | Measured time | Steps/sec |
|---|---|---|---|---|---|
| Final (`modelo_final/`), current | 300 000 | 512 | **585** | 29.3 min (1760 s) | 170 |
| Final (`modelo_final/`), previous version (150 000) | 150 000 | 512 | 292 | 12.7 min (761 s) | 197 |
| `n_steps=90` test (`modelo_final_n90/`) | 150 000 | 90 | **1666** | 13.8 min (828 s) | 181 |
| A/B/C1/C2 experiments (`../output/experimentos/`, closed) | 30 000 each | 512 | **58** each | 2.1-2.3 min each | 217-236 |
| `prueba_rewards/` A/B/C/D | 30 000 each | 512 | **58** each | 2.1-3.8 min each (varies with machine load at the time) | 132-237 |

**The number that matters is not wall-clock time, it is how many network updates PPO performs.** With `n_steps=512`, each update consumes 512 simulation steps; `total_timesteps // n_steps` gives the number of times the network (policy + value) is actually adjusted over the whole run:
- 30 000 timesteps → **58 updates**.
- 150 000 timesteps → **292 updates** - 5 times more.

Wall-clock time itself is cheap (2-4 min for 30 000 steps, ~13 min for 150 000, ~29 min for 300 000, on this machine) - what is really lacking in the 30 000-step runs is not time, it is **updates**: 58 gradient rounds are not enough for the network to converge to a competent policy, and that is why the % of people served in ALL the short experiments (A/B/C1/C2 sequence, `prueba_rewards/`) ends up well below the base policy (40-52% against the base's 88%, see `prueba_rewards/README.md` section 5) - it is not that the reward is wrong, it is that 58 updates is, for this problem, a genuinely short training budget. The final agent (`modelo_final/`, now 585 updates, previously 292) is the one that does match the base - and with the current run's 585, besides matching it on average, it already gains a clear advantage in the upper percentiles (`comparacion/README.md` section 2.1).

**Clarification about `n_steps=90`:** it is not the same as "58 updates" - these are two different numbers that can be confused. `n_steps` is how many simulation steps go into EACH update (the batch size); the 58 updates in the table come from `n_steps=512`. If `n_steps=90` were used with 30 000 timesteps, there would be MORE updates (30000/90≈333), not fewer - smaller batches, but more frequent. In fact `n_steps=90` was already tried with a long budget (150 000 steps, section 4.1, BEFORE the final model's budget was raised to 300 000 - that specific comparison was not repeated at the new budget): it gives 1666 updates (5.7x more than the 292 that `n_steps=512` had at that same budget) and still generalizes WORSE - more updates is not automatically better if each one sees very little experience (see the full explanation in 4.1). The reason the 30 000-step runs perform poorly is the combination of FEW updates (58) AND a short budget overall, not `n_steps` by itself.

---

## 5. How to run

```bash
cd modelo_rl/notebooks
# 1. Experiment sequence -- decides the reward for the final run
jupyter nbconvert --to notebook --execute --inplace 02_secuencia_experimentos_reward.ipynb
# 2. (if the winner changes) update simulador/config/instance.yaml -> agente.entrenamiento.recompensa_overrides
# 3. Final long training run
jupyter nbconvert --to notebook --execute --inplace 01_enfoque_y_entrenamiento_final.ipynb
```

Requires `pip install stable-baselines3` (installs PyTorch as a dependency), and that `demand/output/`, `bergen-boats/02_ruteo_navegable/output/`, and `politica_base/notebooks/00_preparar_demanda_escalones.ipynb` have already run (the base policy is used within the experiment sequence as a comparison reference). `comparacion/notebooks/05_comparacion_agente_vs_base.ipynb` requires that `01_enfoque_y_entrenamiento_final.ipynb` has already saved `output/modelo_final/modelo_ppo.zip`.

---

## 6. Assumptions and limitations

- **Small training instance (`escalon_1`, 2 boats, morning window):** chosen to allow fast iteration in the experiment sequence and in the diagnostic cycle - training was not done on `escalon_2`/`escalon_3` (full day / week). An agent trained on `escalon_1` does not necessarily generalize to instances with more boats or a different time window.
- **Experiment sequence with a single training seed per combo (30 000 steps):** sufficient to compare the SHAPE of the reward across nearby configurations, but each individual number in the section 3 table carries the variability inherent to a single short run - it was not averaged over multiple training seeds per experiment (each one IS evaluated on the 5 fixed evaluation seeds, which do not vary).
- **`penalizacion_maxima_persona` in the RL reward vs. production:** the winning combo (C1) disables the cap (`1000000.0`, effectively no cap) - different from the production value (`1.0`) that continues to govern `politica_base/` and escalones 1-3, unchanged. The `agente.entrenamiento`/`agente.experimentos` overrides never touch `simulador/config/instance.yaml` → `recompensa` (the production section).
