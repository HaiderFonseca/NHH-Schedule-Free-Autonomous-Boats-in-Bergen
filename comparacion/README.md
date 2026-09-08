# Agente PPO vs. política base

**Qué es esto, en una frase:** corre el agente entrenado (`modelo_rl/`) y la política de referencia (`politica_base/`) sobre las mismas 5 semillas de evaluación, agregado y con el detalle completo de una semilla concreta.

---

## 1. Cómo está organizado

```
comparacion/
├── README.md                                  # este archivo
├── notebooks/
│   └── 05_comparacion_agente_vs_base.ipynb    # comparación agregada (5 semillas) + detalle semilla 1001
└── output/                                     # tablas, gráficas interactivas, animaciones, GIFs
```

El notebook importa `simulador/src` (motor), `politica_base/src` (`asignar_flota`) y `modelo_rl/src` (`EntornoDemandaAleatoria`), y lee `simulador/config/instance.yaml`. Necesita que `modelo_rl/output/modelo_final/modelo_ppo.zip` ya exista (`modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb` ya corrido).

**Cómo se evalúa:** las dos políticas corren sobre las mismas 5 semillas (`agente.evaluacion.semillas`: 1001-1005, fuera del rango de `semilla_entrenamiento`), vía `EntornoDemandaAleatoria.reset(seed=...)` -- misma semilla, misma demanda para las dos, así la comparación es justa. El reward se calcula con la fórmula de RL (`agente.entrenamiento.recompensa_overrides`: sin techo, `peso_movimiento=0.003`) para AMBAS políticas, no la de producción -- si cada una usara una fórmula distinta, comparar el reward total no significaría nada.

---

## 2. Comparación agregada (5 semillas)

| Métrica | Agente PPO | Política base | Diferencia |
|---|---|---|---|
| % atendidas | 88.0% | 88.0% | 0.0 pp |
| Espera media | 17.1 min | 16.4 min | +0.6 min |
| Tiempo en sistema medio | 29.0 min | 26.1 min | +3.0 min |
| Tiempo en sistema máximo | 60.8 min | 69.7 min | **-9.0 min** |
| Movimientos totales (5 episodios) | 191 | 155 | +36 |
| Ocupación media | 3.50 | 3.50 | ~0 |
| Reward total (5 episodios, fórmula RL) | **-3772.0** | -5259.0 | **+1487.0** |

**El agente iguala exactamente a la base en % atendidas** (748/850 unidades en los dos casos, sobre las 5 semillas combinadas) -- ya no hay brecha de servicio. Gana claramente en el reward que se entrenó a optimizar (-3772 vs. -5259) y en el peor caso de tiempo en sistema (60.8 vs. 69.7 min) -- exactamente el efecto esperado de entrenar sin techo en la penalización (`modelo_rl/README.md` sección 3): el agente aprendió a que dejar a alguien esperar mucho tiempo sigue costando cada vez más, así que prioriza no dejar a nadie en el peor de los casos. El costo de esto es una espera media levemente mayor (17.1 vs. 16.4 min) y bastantes más movimientos (191 vs. 155, 36 de más) -- el agente mueve la flota de forma más activa para lograrlo.

**Por semilla, el resultado no es parejo** (`output/reward_por_semilla.csv`):

| Semilla | Reward agente | Reward base | Diferencia |
|---|---|---|---|
| 1001 | -546.9 | -132.8 | -414.1 (gana la base) |
| 1002 | -366.7 | -617.1 | +250.4 (gana el agente) |
| 1003 | -564.6 | -318.8 | -245.8 (gana la base) |
| 1004 | -1525.6 | -4091.1 | **+2565.4** (gana el agente, por mucho) |
| 1005 | -768.3 | -99.3 | -668.9 (gana la base) |

El agente pierde en 3 de las 5 semillas individuales, pero gana el promedio porque en la semilla 1004 la base tiene una corrida particularmente mala (-4091.1, la peor de las 10 corridas) mientras el agente la maneja razonablemente (-1525.6) -- consistente con la lectura de arriba: el agente es más parejo en el peor caso, la base puede ser mejor en el caso típico pero tiene más varianza hacia abajo.

**¿El agente simplemente no mueve la flota?** No -- es justo lo contrario. `metricas.decisiones_por_barco` (cruza el log de decisiones con el estado en ese momento -- ver `simulador/README.md` sección 5) muestra que el agente espera en solo **5.4%** de sus decisiones (11 de 202), contra **47.5%** de la política base (140 de 295) -- casi 9 veces menos. Tabla completa por barco en `output/decisiones_por_barco_comparado.csv` y `output/por_barco_comparado.csv` (movimientos, tiempo navegado/esperando, ocupación por barco).

**Por par origen-destino** -- `output/por_par_comparado.csv`: el agente iguala o supera a la base en varios pares (`kleppesto->laksevag`, `laksevag->bryggen`, ambos 100%/93% de sus destinos) y queda por debajo en otros (`kleppesto->sandviken` 88.6% vs. 100%, `bryggen->sandviken` 77.8% vs. 100%) -- no gana de forma pareja en todos los pares, consistente con que la política base también es competente (`politica_base/README.md` sección 2), no un piso fácil de superar en todos lados.

**Gráficas** -- `output/heatmaps_comparados.html` (% atendidas por par, lado a lado) y `output/reward_por_semilla.html` (barras de reward por semilla).

---

## 3. Detalle completo, semilla 1001

Mismo paquete de 5 métricas/gráficas que produce `politica_base/notebooks/01_escalon_1_verificacion.ipynb` (`simulador/README.md` sección 5), corrido para las dos políticas sobre la semilla 1001 -- para ver el comportamiento CONCRETO de un caso, no solo el agregado.

| | Agente | Base |
|---|---|---|
| % atendidas (esta semilla) | 81.5% (110/135) | 91.9% (124/135) |
| Espera media | 17.4 min | 9.4 min |
| Conservación | OK | OK |

**En esta semilla concreta, la base sirve MÁS gente y con menos espera que el agente** -- lo opuesto de lo que el agregado de la sección 2 podría sugerir a primera vista. Es la razón exacta por la que este detalle importa: el agregado (5 semillas) favorece al agente por el reward (que castiga fuerte el peor caso, sin techo), no porque el agente sirva sistemáticamente mejor semilla por semilla -- la semilla 1001 es una de las 3 donde gana la base (sección 2). Ambas lecturas son correctas a la vez: el agente es mejor "en el peor caso agregado", la base puede ser mejor "en un caso típico concreto" como este.

**Gráficas** (`output/*_semilla1001.html`, agente y base por separado): `wait_profile_{agente,base}_semilla1001.html` (personas esperando en el tiempo), `fleet_occupancy_{agente,base}_semilla1001.html` (ocupación de cada barco), `pct_served_heatmap_comparado_semilla1001.html` (heatmap lado a lado, solo esta semilla), `reward_breakdown_{agente,base}_semilla1001.html` (desglose de recompensa en el tiempo), `backlog_by_pair_{agente,base}_semilla1001.html` (backlog al final).

**Visualización paso a paso:** `animacion_{agente,base}_semilla1001.gif` (los 90 pasos completos), inspector en minutos concretos, y el reproductor interactivo completo -- mismo patrón que `politica_base/notebooks/01_escalon_1_verificacion.ipynb` (botones de paso a paso, **solo funcionan con un kernel de Jupyter vivo**).

---

## 4. Cómo correr

```bash
cd comparacion/notebooks
jupyter nbconvert --to notebook --execute --inplace 05_comparacion_agente_vs_base.ipynb
```

Necesita `modelo_rl/output/modelo_final/modelo_ppo.zip` y `vecnormalize.pkl` ya guardados (`modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb` corrido primero).

---

## 5. Supuestos y limitaciones

- **La comparación de reward usa la fórmula de RL (sin techo, `peso_movimiento=0.003`) para las dos políticas**, no la fórmula de producción que reportan `politica_base/` y los escalones 1-3 -- necesario para que el número sea comparable 1:1, pero significa que "quién gana en reward" depende de qué fórmula se eligió para entrenar al agente, no es una medida neutral independiente de esa elección.
- **5 semillas es una muestra chica** -- la sección 2 ya muestra que el resultado agregado puede estar dominado por un solo caso atípico (semilla 1004). No se corrió sobre más semillas ni sobre instancias más grandes (`escalon_2`/`escalon_3`) -- el agente se entrenó y evaluó únicamente sobre `escalon_1` (`modelo_rl/README.md` sección 6).
- **"Ganar" en % atendidas no es el único criterio razonable** -- ver sección 3: en la semilla mostrada en detalle, la base sirve más gente, pero el agente maneja mejor el peor caso de esa misma corrida a nivel reward. Cuál de los dos importa más depende de qué se quiera optimizar en un despliegue real (¿cumplimiento promedio, o proteger el peor caso?) -- pregunta abierta, no resuelta por este proyecto.
