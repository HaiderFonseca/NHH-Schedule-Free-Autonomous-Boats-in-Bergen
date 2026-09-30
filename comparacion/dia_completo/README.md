# Comparación completa -- PPO vs H0 vs H3 (día completo, 12 barcos, demanda 10%)

**Qué es esto, en una frase:** compara el agente PPO final (`modelo_rl/ppo_dia_10pct_12barcos_reward_D_1M/`) contra las heurísticas H0 y H3 sobre el mismo escenario exacto y las mismas 5 semillas de evaluación, con tablas, gráficas (estáticas e interactivas) y una animación GIF de la mañana, todo en una sola carpeta lista para copiar a la presentación.

---

## 1. Cómo está organizado

```
comparacion/dia_completo/
├── README.md                                          # este archivo
├── notebooks/
│   └── 01_comparacion_completa_ppo_h0_h3.ipynb        # genera TODO lo de abajo, ya ejecutado
├── figuras/                                            # PNG (para pegar en diapositivas) + HTML (interactivo) + GIF
│   ├── barras_servicio.png
│   ├── barras_tiempos.png
│   ├── barras_operacion_flota.png
│   ├── dashboard_comparativo_h0_h3_ppo.png             # las 12 métricas en una sola imagen -- una diapositiva de resumen
│   ├── perfil_espera_{H0,H3,PPO}.png / .html
│   ├── ocupacion_flota_{H0,H3,PPO}.png / .html
│   ├── heatmap_cumplimiento_{H0,H3,PPO}.png / .html
│   ├── heatmap_cumplimiento_comparado.png / .html      # los 3 lado a lado
│   ├── desglose_recompensa_{H0,H3,PPO}.png / .html     # NO comparable entre políticas, ver sección 4
│   ├── backlog_{H0,H3,PPO}.png / .html
│   ├── animacion_h0_h3_ppo_manana_semilla1001.gif      # los 3 paneles lado a lado, 06:00-08:00 (GIF)
│   ├── animacion_PPO_semilla1001_manana.gif            # PPO solo (GIF)
│   ├── animacion_{H0,H3}_semilla1001_manana.gif        # copiados de heuristicas/outputs/figuras/ (mismo escenario, GIF)
│   ├── video_{H0,H3,PPO}_semilla1001_manana.mp4        # los mismos, en .mp4 -- pausable en PowerPoint
│   └── video_h0_h3_ppo_comparado_semilla1001_manana.mp4  # los 3 lado a lado, en .mp4
└── resultados/
    └── tabla_comparativa_h0_h3_ppo_completa.csv        # tabla oficial agregada (media+std, 5 semillas)
```

El notebook importa `simulador/src` (motor + métricas + visualización), `politica_base/src` y `heuristicas/h3_costo_global/src` (H0/H3, sin modificar) y `modelo_rl/ppo_dia_10pct_12barcos_reward_D_1M/` (`entrenar.py`/`evaluar.py`, sin reentrenar ni reevaluar la tabla oficial). No se modificó el simulador, las heurísticas, el modelo entrenado, ninguna métrica existente, ni GitHub -- solo se importó código existente y se escribieron archivos nuevos dentro de esta carpeta.

---

## 2. Escenario (idéntico para las tres políticas)

| | |
|---|---|
| Barcos | 12 |
| Capacidad por barco | 30 personas |
| Demanda | 10% de la población oficial |
| Horario | 06:00-24:00 (día completo, 540 pasos de 2 min) |
| Semillas de evaluación | 1001, 1002, 1003, 1004, 1005 |

---

## 3. Metodología, en dos niveles

**1. Comparación oficial (agregada, 5 semillas) -- la que cuenta para el veredicto.**
H0 y H3 se REUSAN tal cual de `heuristicas/outputs/resultados/resultados_h0_h1_h2_h3.csv` (generado por `heuristicas/experimento_fleet_sweep.py`) -- **nunca recalculados**. PPO se reusa de `modelo_rl/ppo_dia_10pct_12barcos_reward_D_1M/resultados_evaluacion_ppo.csv` (generado por `evaluar.py`) -- tampoco se recalcula aquí. Este notebook solo lee esos dos CSV y arma la tabla/gráficas comparativas.

**2. Detalle ilustrativo (semilla 1001 únicamente) -- para las gráficas de serie de tiempo y el GIF.**
Esas gráficas necesitan el `env` completo paso a paso (`historial_estados`), que no está guardado en los CSV agregados. Para esto el notebook SÍ vuelve a *correr* H0 y H3 (copiando el bucle exacto de `experimento_fleet_sweep.py::correr()`, código sin modificar una sola línea, sobre la misma demanda pre-generada `grupos_seed1001.csv`) y vuelve a *evaluar* PPO para esa semilla (misma función `evaluar.correr_ppo()`, sin reentrenar nada). **Esto no cambia ningún número de la tabla oficial** -- es solo para poder graficar/animar un episodio completo.

**Chequeo de transparencia (demanda "misma semilla" entre caminos de código distintos):** H0/H3 leen la demanda de un CSV pre-generado; PPO la genera on-the-fly en `reset(seed=1001)` vía `EntornoDemandaAleatoria`. Ambos caminos usan el mismo generador (`demand/src/llegadas.py`) pero por rutas de código distintas -- el notebook imprime el total de personas generadas por las tres para confirmar (o refutar, sin esconderlo) que la realización de demanda es exactamente la misma. Ver la salida de la celda correspondiente en el notebook ejecutado.

---

## 4. Resultado oficial (media ± std, 5 semillas)

| métrica | H0 | H3 | PPO |
|---|---|---|---|
| pct_atendidas | 99.58 | 99.61 | 99.39 |
| backlog_final_total | 34.2 | 31.8 | **50.2** |
| espera_media_min | 6.18 | 4.81 | **12.76** |
| espera_p95_min | 15.71 | 12.27 | **37.25** |
| espera_max_min | 28.03 | 18.36 | **105.40** |
| viaje_medio_min | 9.79 | 9.79 | 9.79 |
| sistema_medio_min | 15.97 | 14.60 | **22.55** |
| movimientos_totales | 1268.6 | 803.2 | 1149.8 |
| movimientos_vacios | 704.0 | 200.6 | 660.6 |
| movimientos_con_carga | 564.6 | 602.6 | 489.2 |
| ocupación_%capacidad | 16.31 | 16.32 | 16.30 |
| pct_esperando_flota | 26.28 | 54.25 | 31.72 |

(`sistema_p95_min`/`sistema_max_min` solo están calculadas para PPO -- 46.17 / 115.40 min; `experimento_fleet_sweep.py` no las guardó para H0/H3, no se recalculan aquí.)

**Sobre la comparación de reward:** cada política corrió con una fórmula de recompensa distinta -- H0/H3 con la fórmula de producción (`simulador/config/instance.yaml -> recompensa`), PPO con **Reward D** (`agente.entrenamiento.recompensa_overrides`: sin tolerancia, sin premio por entrega, sin penalización de movimiento). Las gráficas de "desglose de recompensa" de `figuras/` están, por eso, en escalas distintas por política y **no se usan, en ningún punto de este análisis, como criterio de comparación**. No se declara a PPO mejor (ni peor) por su reward.

---

## 5. Veredicto honesto

**PPO no supera a H0 ni a H3 en este escenario.** Es casi comparable en `pct_atendidas` (99.39% vs. 99.58-99.61%), pero pierde claramente en las métricas de experiencia del usuario: la espera media, p95 y máxima son 2-4x peores que ambas heurísticas, y el backlog final también es mayor. La única cifra donde PPO queda "en el medio" es movimientos totales/vacíos, sin que eso se traduzca en mejor servicio. H3 es, con margen, la política con mejor tiempo de espera; H0 es la que más mueve la flota (más movimientos vacíos) pero aun así compite de cerca con H3 en espera.

---

## 6. Video (.mp4) además del GIF, y reproductor interactivo

Un GIF insertado en PowerPoint no se puede pausar de forma confiable durante la presentación; un `.mp4` sí (controles nativos de reproducción). Se generó con `imageio`+`imageio-ffmpeg`, reusando `visualizacion.dibujar_frame` para cada cuadro (mismo patrón que `comparacion/escalon_1_h0_h3_ppo/`) -- nunca se modificó ese archivo salvo dos textos que se tradujeron a inglés ("waiting:" y el título del mapa), para que todas las animaciones del proyecto queden en inglés.

Además, para cada política individual (H0, H3, PPO, semilla 1001) hay un **reproductor interactivo paso a paso** (slider + botones + panel de texto en inglés: estado de cada barco, decisiones, colas, recompensa del paso -- y para H3, las reservas activas), cubriendo el DÍA COMPLETO (no solo la ventana de la mañana del video/GIF). Reusa el patrón de `visualizacion.reproductor_interactivo` sin modificar ese archivo. **Solo funciona con un kernel de Jupyter vivo** -- abrir el notebook y correr esas celdas; en esta copia ya ejecutada los botones no responden.

---

## 7. Cómo correr

```bash
cd comparacion/dia_completo/notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_comparacion_completa_ppo_h0_h3.ipynb
```

Requiere que ya existan: `heuristicas/outputs/resultados/resultados_h0_h1_h2_h3.csv`, `modelo_rl/ppo_dia_10pct_12barcos_reward_D_1M/{modelo_ppo.zip, vecnormalize.pkl, resultados_evaluacion_ppo.csv}`, y `politica_base/output/escalon_dia_10pct/grupos_seed1001.csv`. Tarda varios minutos (descarga el mapa base una vez, corre 3 episodios completos de 540 pasos, renderiza ~61 frames del GIF combinado).

---

## 8. Supuestos y limitaciones

- El detalle ilustrativo y el GIF usan **una sola semilla (1001)** -- no representan la variabilidad entre semillas, solo el resultado oficial agregado (sección 4) la captura.
- El GIF cubre solo la ventana 06:00-08:00 (mañana), no el día completo -- mismo criterio que `heuristicas/notebooks/04_visualizacion_heuristicas.ipynb`, para mantener el archivo liviano y legible.
- Las gráficas de desglose de recompensa NO son comparables entre políticas (fórmulas distintas, ver sección 4) -- se incluyen solo como reporte individual de cada política.
- Si el chequeo de transparencia de la sección 3 muestra totales de personas generadas distintos entre H0/H3 y PPO para la "semilla 1001", significa que ambos caminos de generación de demanda no producen exactamente la misma realización pese a compartir el número de semilla -- una limitación conocida de tener dos caminos de código para generar demanda (CSV pre-generado vs. on-the-fly), documentada aquí sin ocultarla.
