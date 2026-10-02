# Schedule-Free Autonomous Boats in Bergen

Simulation and dispatch-logic design for a small-boat, on-demand water transport service in Bergen, Norway. The service does not exist yet: this project does not improve something that already runs, it builds the model of how such a service would operate, with no fixed timetable, using a dispatch policy that recomputes itself every few minutes.

**Full report (in English):** [`docs/informe/informe.pdf`](docs/informe/informe.pdf). This is the document that summarizes the whole project end to end, with the math and the results. This README is only a quick map of the repository.

Advisors: Julio Goez (optimization) and Stein W. Wallace (stochastic programming). Starting point: Gu & Wallace (2021), *Operational benefits of autonomous vessels in logistics*, the static water-taxi model this project builds a real-time version of.

## Project order

The work moves in the same order as the folders below, each one built on top of the previous:

| Folder | What it holds |
|---|---|
| [`bergen-boats/`](bergen-boats/) | The base instance: the 4 stops, the real navigable routing (follows the coastline), and the resulting travel-time matrix. |
| [`demand/`](demand/) | The synthetic demand model, anchored in official population and employment data (SSB). |
| [`simulador/`](simulador/) | The engine: the simulation environment (state, actions, reward), the metrics, and the visualization used by everything else. |
| [`politica_base/`](politica_base/) | The reference policy H0 ("nearest-available") and the simulator's verification at three scales. |
| [`heuristicas/`](heuristicas/) | Heuristics H1, H2, and H3, each built on top of the previous, with their verification cases and the fleet-size sweep. |
| [`modelo_rl/`](modelo_rl/) | The reinforcement-learning agent (PPO): the reward-formula search and the training runs, at small scale and at full scale. |
| [`comparacion/`](comparacion/) | The final comparisons between the agent and the heuristics, with tables, figures, and animations. |
| [`docs/`](docs/) | The final report (`docs/informe/`) and the simulator's technical specification. |
| [`papers/`](papers/) | The reference papers cited in the report. |

Each folder is self-contained: it has its own `README.md` with the detail for that part, and its own output folder (`output/` or `outputs/`) with the results and figures. Shared parameters (fleet, reward, evaluation seeds) live in a single file, `simulador/config/instance.yaml`, which every folder reads without copying it.

## Where to start

The fastest way to understand the whole project is to read the report (`docs/informe/informe.pdf`): it covers routing, demand, the simulator, the four heuristics, the RL agent, and the final comparison in one ordered document. For the code and the detailed results behind any one part, the `README.md` in the matching folder has the detail and instructions to run it.

**Note:** the folder-level `README.md` files and the code comments are in Spanish (the language this project was developed in); the final report above is in English.
