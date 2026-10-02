# Comparison H0 vs H3 vs PPO, small scenario (2 boats, Step 1)

**What this is, in one line:** the same kind of comparison as `comparacion/dia_completo/`, but on the simplest scenario in the project (Step 1: 2 boats, 06:00-09:00, about 0.8% of the population), easier to follow in a presentation, extending `comparacion/notebooks/05_comparacion_agente_vs_base.ipynb` (which only compared PPO vs. H0) to also include **H3**.

---

## 1. How it is organized

```
comparacion/escalon_1_h0_h3_ppo/
├── README.md                                          # this file
├── notebooks/
│   └── 01_comparacion_h0_h3_ppo_2barcos.ipynb         # generates EVERYTHING below, already executed
├── figuras/                                            # PNG + HTML + VIDEO (.mp4)
│   ├── barras_servicio.png / barras_tiempos.png / barras_operacion_flota.png
│   ├── dashboard_comparativo_h0_h3_ppo.png             # all 14 metrics in one image
│   ├── perfil_espera_comparado_semilla*.png/.html      # H0+H3+PPO overlaid, one chart
│   ├── desglose_recompensa_comparado_semilla*.png/.html
│   ├── heatmap_cumplimiento_comparado_semilla*.png/.html
│   ├── ocupacion_flota_{H0,H3,PPO}_semilla*.png/.html
│   ├── backlog_{H0,H3,PPO}_semilla*.png/.html
│   ├── video_{H0,H3,PPO}_semilla{mejor,peor}_*.mp4      # individual video, can be paused in PowerPoint
│   └── video_h0_h3_ppo_comparado_semilla*_*.mp4         # the three side by side, same video
└── resultados/
    ├── resultados_por_semilla_h0_h3_ppo.csv             # 15 rows (3 policies x 5 seeds)
    ├── tabla_comparativa_h0_h3_ppo_completa.csv          # official aggregate table (mean + std)
    └── reward_por_semilla.csv
```

---

## 2. Why video (.mp4) instead of GIF

A GIF inserted into PowerPoint cannot be reliably paused during a presentation. An `.mp4` can, with PowerPoint's native playback controls (play/pause/scrub bar). It was generated with `imageio` + `imageio-ffmpeg` (installed for this in the environment; the second one bundles its own ffmpeg binary, it does not depend on a system install), reusing `simulador/src/visualizacion.py::dibujar_frame` for every frame, the exact same drawing used by the GIFs in the rest of the project, only the output format changes (`visualizacion.py` itself was never modified).

---

## 3. Scenario (identical for all three policies)

| | |
|---|---|
| Boats | 2 |
| Hours | 06:00-09:00 (Step 1, short episodes) |
| Demand | about 0.8% of the population |
| Evaluation seeds | 1001, 1002, 1003, 1004, 1005 |
| PPO model | `modelo_rl/prueba_rewards/output/modelos/D/` (**not** `modelo_rl/output/modelo_final/`, used in an earlier version of this notebook) |

**On the model change:** `modelo_rl/output/modelo_final/` and `modelo_rl/prueba_rewards/output/modelos/D/` share the same nominal config (reward D, Step 1, 150,000 timesteps, `semilla_entrenamiento=123`) but are two DIFFERENT training runs, confirmed by SHA-256 hash (different weights, not the same file). PPO's own stochasticity produced two different policies from the "same" config. `prueba_rewards/D` gives a notably higher `pct_atendidas` (87.17% vs. 81.78% for the other checkpoint, closer to H0's 89.34%), but note this is not a uniform improvement: waiting times (`espera_media_min`, `espera_p95_min`, `espera_max_min`) are slightly WORSE than with the earlier checkpoint. Reported in full, without hiding the nuance.

**Key methodological difference from `dia_completo/`:** here all three policies run on the SAME environment (`EntornoDemandaAleatoria`) with the SAME reward formula (Reward D, `agente.entrenamiento.recompensa_overrides`), the same pattern already established in `comparacion/notebooks/05`. That makes the reward breakdown **comparable across policies here** (unlike `dia_completo/`, where H0/H3 ran under the production formula). Even so, the verdict is led by the operational metrics, not by reward, the same anti-Goodhart standard used throughout the rest of the project.

---

## 4. How to run it

```bash
cd comparacion/escalon_1_h0_h3_ppo/notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_comparacion_h0_h3_ppo_2barcos.ipynb
```

Requires `modelo_rl/prueba_rewards/output/modelos/D/{modelo_ppo.zip, vecnormalize.pkl}` and the outputs of `demand/` (`masas_por_nodo.csv`, `matriz_intensidad_od.csv`) already generated.

**Note:** every time it is run again, the best/worst seed for the agent can change (it depends on `sistema_medio_min`, which can vary if the model changes); the detail files (videos, per-seed figures) are named after that seed, and files from a previous run tagged with a different seed become stale and must be deleted by hand (or run again into an empty `figuras/` folder).

---

## 5. Interactive player (step by step, English state panel)

For each individual policy (best and worst seed) the notebook adds a player with a slider, buttons, and an English text panel (each boat's state, queues, the decision taken, the step reward); for H3 the panel also shows the active reservations (which queue each boat is committed to). It reuses the pattern from `simulador/src/visualizacion.py::reproductor_interactivo` (without modifying that file), rewritten in English and extended with reservations. **It only works with a live Jupyter kernel**: open the notebook in VS Code / Jupyter Lab and run those cells; in this already-executed, saved copy the buttons will not respond (same as the original player).

---

## 6. Assumptions and limitations

- The illustrative detail (profiles, heatmaps, videos, interactive player) uses only the best and worst seed FOR THE AGENT (by `sistema_medio_min`, not by reward); the official aggregate table (results section) does cover all 5 seeds.
- Deliberately small scenario (2 boats, 3 hours), meant to explain the mechanism in a presentation, not to draw conclusions about scale; that is what `comparacion/dia_completo/` (12 boats, full day) is for.
- See the executed notebook for the full comparative table and the automatic verdict (not duplicated here, to avoid going out of sync if it is run again).
