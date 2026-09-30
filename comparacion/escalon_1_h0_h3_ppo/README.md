# Comparación H0 vs H3 vs PPO -- escenario chico (2 barcos, escalón 1)

**Qué es esto, en una frase:** el mismo tipo de comparación que `comparacion/dia_completo/`, pero en el escenario más simple del proyecto (`escalón 1`: 2 barcos, 06:00-09:00, ~0.8% de la población) -- más fácil de seguir en una presentación -- extendiendo `comparacion/notebooks/05_comparacion_agente_vs_base.ipynb` (que solo comparaba PPO vs. H0) para incluir también **H3**.

---

## 1. Cómo está organizado

```
comparacion/escalon_1_h0_h3_ppo/
├── README.md                                          # este archivo
├── notebooks/
│   └── 01_comparacion_h0_h3_ppo_2barcos.ipynb         # genera TODO lo de abajo, ya ejecutado
├── figuras/                                            # PNG + HTML + VIDEO (.mp4)
│   ├── barras_servicio.png / barras_tiempos.png / barras_operacion_flota.png
│   ├── dashboard_comparativo_h0_h3_ppo.png             # las 14 métricas en una sola imagen
│   ├── perfil_espera_comparado_semilla*.png/.html      # H0+H3+PPO superpuestos, mismo gráfico
│   ├── desglose_recompensa_comparado_semilla*.png/.html
│   ├── heatmap_cumplimiento_comparado_semilla*.png/.html
│   ├── ocupacion_flota_{H0,H3,PPO}_semilla*.png/.html
│   ├── backlog_{H0,H3,PPO}_semilla*.png/.html
│   ├── video_{H0,H3,PPO}_semilla{mejor,peor}_*.mp4      # video individual, pausable en PowerPoint
│   └── video_h0_h3_ppo_comparado_semilla*_*.mp4         # los 3 lado a lado, mismo video
└── resultados/
    ├── resultados_por_semilla_h0_h3_ppo.csv             # 15 filas (3 políticas x 5 semillas)
    ├── tabla_comparativa_h0_h3_ppo_completa.csv          # tabla oficial agregada (media+std)
    └── reward_por_semilla.csv
```

---

## 2. Por qué video (.mp4) y no GIF

Un GIF insertado en PowerPoint no se puede pausar de forma confiable durante la presentación. Un `.mp4` sí -- controles de reproducción nativos de PowerPoint (play/pausa/barra de tiempo). Se generó con `imageio` + `imageio-ffmpeg` (instalados para esto en el entorno; el segundo trae su propio binario de ffmpeg, no depende de una instalación del sistema), reusando `simulador/src/visualizacion.py::dibujar_frame` para cada cuadro -- el mismo dibujo exacto que usan los GIF del resto del proyecto, solo cambia el formato de salida (nunca se modificó `visualizacion.py`).

---

## 3. Escenario (idéntico para las tres políticas)

| | |
|---|---|
| Barcos | 2 |
| Horario | 06:00-09:00 (escalón 1, episodios cortos) |
| Demanda | ~0.8% de la población |
| Semillas de evaluación | 1001, 1002, 1003, 1004, 1005 |
| Modelo PPO | `modelo_rl/prueba_rewards/output/modelos/D/` (**no** `modelo_rl/output/modelo_final/`, usado en una versión anterior de este notebook) |

**Sobre el cambio de modelo:** `modelo_rl/output/modelo_final/` y `modelo_rl/prueba_rewards/output/modelos/D/` tienen la misma config nominal (reward D, escalón 1, 150 000 timesteps, `semilla_entrenamiento=123`) pero son dos corridas de entrenamiento DISTINTAS -- confirmado por hash SHA-256 (pesos distintos, no el mismo archivo). La estocasticidad de PPO produjo dos políticas distintas a partir de la "misma" config. `prueba_rewards/D` da un `pct_atendidas` notablemente más alto (87.17% vs. 81.78% del otro checkpoint, más cerca del 89.34% de H0) -- pero OJO, no es una mejora uniforme: los tiempos de espera (`espera_media_min`, `espera_p95_min`, `espera_max_min`) son ligeramente PEORES que con el checkpoint anterior. Se reporta completo, sin ocultar el matiz.

**Diferencia metodológica clave con `dia_completo/`:** aquí las tres políticas corren sobre el MISMO entorno (`EntornoDemandaAleatoria`) con la MISMA fórmula de recompensa (Reward D, `agente.entrenamiento.recompensa_overrides`) -- mismo patrón ya establecido en `comparacion/notebooks/05`. Eso hace que aquí el desglose de recompensa **sí sea comparable entre políticas** (a diferencia de `dia_completo/`, donde H0/H3 corrían con la fórmula de producción). Aun así, el veredicto lidera con las métricas operativas, no con el reward -- mismo criterio anti-Goodhart del resto del proyecto.

---

## 4. Cómo correr

```bash
cd comparacion/escalon_1_h0_h3_ppo/notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_comparacion_h0_h3_ppo_2barcos.ipynb
```

Requiere `modelo_rl/prueba_rewards/output/modelos/D/{modelo_ppo.zip, vecnormalize.pkl}` y las salidas de `demand/` (`masas_por_nodo.csv`, `matriz_intensidad_od.csv`) ya generadas.

**Nota:** cada vez que se vuelve a correr, la mejor/peor semilla para el agente puede cambiar (depende de `sistema_medio_min`, que puede variar si el modelo cambia) -- los archivos de detalle (videos, gráficas por semilla) quedan nombrados con esa semilla, y los de la corrida anterior con una semilla distinta quedan obsoletos y hay que borrarlos a mano (o volver a correr sobre una carpeta `figuras/` vacía).

---

## 5. Reproductor interactivo (paso a paso, panel de estado en inglés)

Para cada política individual (mejor y peor semilla) el notebook agrega un reproductor con slider + botones + panel de texto en inglés (estado de cada barco, colas, decisión tomada, recompensa del paso) -- para H3 el panel también muestra las reservas activas (a qué cola está comprometido cada barco). Reusa el patrón de `simulador/src/visualizacion.py::reproductor_interactivo` (sin modificar ese archivo), reescrito en inglés y extendido con reservas. **Solo funciona con un kernel de Jupyter vivo** -- abrir el notebook en VS Code / Jupyter Lab y correr esas celdas; en esta copia ya ejecutada y guardada los botones no van a responder (igual que el reproductor original).

---

## 6. Supuestos y limitaciones

- El detalle ilustrativo (perfiles, heatmaps, videos, reproductor interactivo) usa solo la mejor y la peor semilla PARA EL AGENTE (por `sistema_medio_min`, no por reward) -- la tabla oficial agregada (sección de resultados) sí cubre las 5 semillas.
- Escenario deliberadamente chico (2 barcos, 3h) -- pensado para explicar el mecanismo en una presentación, no para sacar conclusiones de escala; para eso está `comparacion/dia_completo/` (12 barcos, día completo).
- Ver el notebook ejecutado para la tabla comparativa completa y el veredicto automático (no se duplica aquí para no desincronizarse si se vuelve a correr).
