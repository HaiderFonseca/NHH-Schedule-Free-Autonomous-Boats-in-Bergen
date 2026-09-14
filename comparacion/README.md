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

**Modelo actual: 300 000 timesteps (585 actualizaciones de PPO)** -- se subió desde los 150 000 (292 actualizaciones) de la versión anterior. Ver `modelo_rl/README.md` sección 4.2 para el tiempo real medido de esta corrida.

**Por qué no lideramos con % atendidas.** El simulador no pierde a nadie -- nadie se retira nunca (`simulador/README.md` sección 4.5), así que alguien que no subió a un barco cuando termina la corrida no está "perdido", sigue en cola y se habría atendido si la corrida seguía un rato más. Eso quiere decir que "% atendidas" mezcla dos cosas distintas: qué tan bien despacha la política, y cuánto quedaba de ventana operativa cuando llegó la última gente -- lo segundo es arbitrario, no dice nada de la política. Lo que la recompensa realmente castiga (y lo que de verdad nos importa) es cuánto tiempo pasa la gente en el sistema -- medio y, sobre todo, máximo -- y cuántos movimientos le cuesta a la flota lograrlo. Por eso la tabla de abajo lidera con esas dos cosas, y deja % atendidas al final, solo como referencia.

| Métrica | Agente PPO | Política base | Diferencia |
|---|---|---|---|
| Tiempo en sistema medio | 26.1 min | 26.1 min | +0.05 min (empate) |
| Tiempo en sistema máximo | 66.8 min | 69.7 min | **-3.0 min** |
| Espera media | 16.3 min | 16.4 min | -0.07 min (empate) |
| Movimientos totales (5 episodios) | 185 | 155 | +30 |
| Reward total (5 episodios, fórmula RL) | **-3754.1** | -5259.0 | **+1504.9** |
| % atendidas (referencia, ver arriba) | 87.8% | 88.0% | -0.2 pp |

Con más entrenamiento, el agente ya prácticamente EMPATA a la base en tiempo medio y espera media (antes le costaba 3 min más de media; ahora la diferencia es ruido), y sigue ganando en el peor caso (66.8 vs. 69.7 min) -- el efecto esperado de entrenar sin techo en la penalización (`modelo_rl/README.md` sección 3): al agente le sigue costando cada vez más dejar a alguien esperando mucho, así que prioriza no dejar a nadie en el peor de los casos, sin sacrificar ya el promedio como antes. El costo que queda es movimientos (185 vs. 155, 30 de más) -- mueve la flota más activamente. Ver sección 2.1 para por qué esto se ve todavía más claro en los percentiles que en el promedio.

### 2.1 Percentiles (la base real de una garantía de servicio)

`metricas.metricas_por_usuario` -- percentiles sobre las unidades ATENDIDAS, sin censurar (el simulador no pierde a nadie, así que son datos reales, no una muestra sesgada hacia los casos rápidos). Es lo que hace falta para poder decir algo como "servimos al 95% de la gente en X minutos" con sustento, no solo el promedio -- y es donde la ventaja del agente se ve mucho más clara que en la tabla de arriba:

| Métrica | Percentil | Agente PPO | Política base | Diferencia |
|---|---|---|---|---|
| Espera | p50 | 14.1 min | 13.9 min | +0.1 min |
| Espera | p90 | 31.9 min | 37.2 min | **-5.3 min** |
| Espera | p95 | 35.0 min | 46.1 min | **-11.1 min** |
| Espera | máx | 54.8 min | 59.7 min | **-5.0 min** |
| Sistema | p50 | 23.5 min | 23.0 min | +0.5 min |
| Sistema | p90 | 40.7 min | 37.8 min | +2.9 min |
| Sistema | p95 | 44.2 min | 58.8 min | **-14.6 min** |
| Sistema | máx | 66.8 min | 69.7 min | -3.0 min |

En la mediana (p50) las dos políticas son prácticamente iguales -- la diferencia real está en la COLA de la distribución: en p95 de espera, el agente le saca casi 11 minutos a la base; en p95 de tiempo en sistema, casi 15 minutos. Esto es consistente con la teoría (`modelo_rl/README.md` sección 3, reward sin techo): el agente no está optimizado para el caso típico, está optimizado para que NINGÚN caso individual crezca sin límite -- y eso es exactamente lo que un percentil alto mide. Tabla completa: `output/percentiles_agregado.csv`.

**Por semilla, el resultado no es parejo** (`output/reward_por_semilla.csv`):

| Semilla | Reward agente | Reward base | Diferencia |
|---|---|---|---|
| 1001 | -216.0 | -132.8 | -83.1 (gana la base) |
| 1002 | -640.1 | -617.1 | -23.0 (gana la base) |
| 1003 | -787.7 | -318.8 | -469.0 (gana la base) |
| 1004 | -1699.3 | -4091.1 | **+2391.7** (gana el agente, por mucho) |
| 1005 | -411.1 | -99.3 | -311.7 (gana la base) |

El agente pierde en 4 de las 5 semillas individuales, pero gana el promedio porque en la semilla 1004 la base tiene una corrida particularmente mala (-4091.1, la peor de las 10 corridas) mientras el agente la maneja bien (-1699.3) -- misma lectura que antes: el agente es más parejo en el peor caso, la base puede ser mejor en el caso típico pero tiene más varianza hacia abajo. Con más entrenamiento el agente ahora pierde en una semilla más que antes (4 de 5, era 3 de 5) -- vale la pena tenerlo presente: ganar el agregado por un solo caso extremo es una base más frágil de lo que "el agente gana en promedio" sugiere a primera vista.

**¿El agente simplemente no mueve la flota?** No -- es justo lo contrario. `metricas.decisiones_por_barco` (cruza el log de decisiones con el estado en ese momento -- ver `simulador/README.md` sección 5) muestra que el agente espera en solo **4.1%** de sus decisiones (8 de 193), contra **47.5%** de la política base (140 de 295) -- unas 12 veces menos. Tabla completa por barco en `output/decisiones_por_barco_comparado.csv` y `output/por_barco_comparado.csv` (movimientos, tiempo navegado/esperando, ocupación por barco).

### 2.2 `veces_espero_con_demanda_local` -- por qué la base "esperaba con demanda local" (y ya no)

La regla de la política base es estricta: **si hay alguien esperando en el nodo donde el barco está, SIEMPRE lo recoge antes que cualquier otra cosa** (`politica_base/README.md` sección 2.1) -- no hay ningún caso legítimo donde decida esperar teniendo demanda local, salvo uno: cuando OTRO barco también está libre en el mismo nodo en el mismo paso, y la coordinación de flota (`asignar_flota`, `politica_base/README.md` sección 2.2) ya le asignó esa demanda al primero -- el segundo, correctamente, no tiene nada que recoger (aunque la foto general del nodo todavía muestre gente esperando, ya está "reclamada").

Antes de esta revisión, `metricas.decisiones_por_barco` reportaba muchos más casos de los que esa regla explica (hasta 15 para un barco en una corrida) -- se encontró la causa real: la función reconstruye, para cada decisión, "qué foto del mundo vio la política en ese momento" cruzando el log de decisiones con `historial_estados`, avanzando un puntero de a 1 cada vez que el minuto de la decisión cambiaba. **Si en algún paso los dos barcos estaban ocupados (ninguno libre, nadie decide nada), ese paso no quedaba contado, y el puntero se atrasaba** -- el atraso se acumula para el resto de la corrida, así que con el tiempo la función terminaba leyendo una foto vieja, de varios pasos atrás, que podía mostrar demanda que YA había sido recogida antes de que la decisión ocurriera de verdad. Se corrigió (`simulador/src/metricas.py`, `decisiones_por_barco`) calculando el índice del frame directamente a partir del minuto de la decisión (en vez de contar saltos), sin asumir que cada decisión nueva es exactamente un paso después de la anterior. Resultado, misma corrida, antes/después del fix:

| | Antes (con el bug) | Después (corregido) |
|---|---|---|
| Base, `veces_espero_con_demanda_local` | 22 casos (10 sin ninguna explicación real) | **1 caso** (coordinación de flota, legítimo) |
| Agente, `veces_espero_con_demanda_local` | -- | **2 casos** (mismo motivo) |

Con el fix, casi todos los casos "sin explicación" desaparecen (verificado caso por caso: cruzando cada "esperó con demanda local" contra si había OTRO barco libre en el mismo nodo en el mismo paso -- 12 de los 22 casos originales ya se explicaban así incluso con el bug; con el fix, los 10 restantes, que no tenían otro barco libre, resultan ser lecturas de un frame viejo, no esperas reales con demanda disponible). El número que queda (1 para la base, 2 para el agente) es exactamente lo que la regla de coordinación predice: casos raros, de dos barcos libres a la vez en el mismo nodo.

**Por par origen-destino** -- `output/por_par_comparado.csv`: el agente iguala o supera a la base en varios pares (`kleppesto->bryggen` 91.4% vs. 80.8%, `laksevag->sandviken` empatado 60%) y queda por debajo en otros (`kleppesto->sandviken` 70.5% vs. 100%, `bryggen->sandviken` 77.8% vs. 100%) -- no gana de forma pareja en todos los pares, consistente con que la política base también es competente (`politica_base/README.md` sección 2), no un piso fácil de superar en todos lados.

**Gráficas** -- `output/heatmaps_comparados.html` (% atendidas por par, lado a lado) y `output/reward_por_semilla.html` (barras de reward por semilla).

---

## 3. Detalle completo, semilla 1001

Mismo paquete de 5 métricas/gráficas que produce `politica_base/notebooks/01_escalon_1_verificacion.ipynb` (`simulador/README.md` sección 5), corrido para las dos políticas sobre la semilla 1001 -- para ver el comportamiento CONCRETO de un caso, no solo el agregado.

| | Agente | Base |
|---|---|---|
| Espera media | 11.8 min | 9.4 min |
| % atendidas, esta semilla (referencia) | 85.2% (115/135) | 91.9% (124/135) |
| Conservación | OK | OK |

**En esta semilla concreta, la base espera un poco menos que el agente** (9.4 vs. 11.8 min) -- lo opuesto de lo que el reward agregado de la sección 2 podría sugerir a primera vista, aunque la brecha ya es mucho más chica que con el modelo anterior (antes era 9.4 vs. 17.4 min). Es la razón por la que este detalle importa: la semilla 1001 es una de las 4 (de 5) donde el reward de la base le gana al agente -- el agregado favorece al agente porque el reward castiga fuerte el peor caso entre las 5 semillas, no porque el agente sirva sistemáticamente mejor semilla por semilla. Las dos lecturas son correctas a la vez: el agente es mejor "en el peor caso agregado", la base puede ser mejor en un caso típico concreto como este.

**Percentiles, esta misma semilla** (`output/percentiles_semilla1001.csv`):

| Métrica | Percentil | Agente | Base |
|---|---|---|---|
| Espera | p50 | 11.3 min | 8.5 min |
| Espera | p90 | 26.4 min | 20.3 min |
| Espera | p95 / máx | 28.3 min | 25.3 min |
| Sistema | p50 | 20.1 min | 16.5 min |
| Sistema | p90 | 36.3 min | 28.3 min |
| Sistema | p95 | 36.3 min | 31.3 min |
| Sistema | máx | 38.4 min | 31.3 min |

Acá SÍ gana la base en toda la tabla, en esta semilla puntual -- consistente con que 1001 es una de las semillas donde el reward le da la razón a la base (sección 2). Es la contraparte necesaria de la sección 2.1: el agregado de 5 semillas favorece al agente en la cola de la distribución, pero en una corrida concreta cualquiera de las dos puede ganar -- por eso se muestran las dos vistas, no solo el agregado.

**Gráficas** (`output/*_semilla1001.html`, agente y base por separado): `wait_profile_{agente,base}_semilla1001.html` (personas esperando en el tiempo), `fleet_occupancy_{agente,base}_semilla1001.html` (ocupación de cada barco), `pct_served_heatmap_comparado_semilla1001.html` (heatmap lado a lado, solo esta semilla), `reward_breakdown_{agente,base}_semilla1001.html` (desglose de recompensa en el tiempo), `backlog_by_pair_{agente,base}_semilla1001.html` (backlog al final).

**Visualización paso a paso:** `animacion_{agente,base}_semilla1001.gif` (los 90 pasos completos), inspector en minutos concretos, y el reproductor interactivo completo -- mismo patrón que `politica_base/notebooks/01_escalon_1_verificacion.ipynb` (botones de paso a paso, **solo funcionan con un kernel de Jupyter vivo**).

---

**Sobre qué modelo es este:** `n_steps=512`, 300 000 timesteps / 585 actualizaciones de PPO (`modelo_rl/output/modelo_final/`, `modelo_rl/README.md` sección 4.2). Se probó también `n_steps=90` (un episodio completo) como alternativa mejor justificada, a un presupuesto de 150 000 pasos -- generalizó peor en las mismas 5 semillas pese a entrenar parecido (`modelo_rl/README.md` sección 4.1); esa comparación específica no se repitió al subir a 300 000, no se usa acá.

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
