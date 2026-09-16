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

## 3. Detalle completo, dos semillas (1001 y 1004)

Mismo paquete de 5 métricas/gráficas que produce `politica_base/notebooks/01_escalon_1_verificacion.ipynb` (`simulador/README.md` sección 5), corrido para las dos políticas sobre DOS semillas concretas -- no solo una -- elegidas porque son casos opuestos: 1001 es una de las semillas donde gana la base, 1004 es donde el agente gana por más margen (sección 2). Todo esto vive en una sola función (`mostrar_detalle_semilla`, `05_comparacion_agente_vs_base.ipynb`), llamada una vez por semilla, para no duplicar treinta celdas por cada una.

| | Semilla 1001 -- Agente | Semilla 1001 -- Base | Semilla 1004 -- Agente | Semilla 1004 -- Base |
|---|---|---|---|---|
| Espera media | 11.8 min | 9.4 min | **18.2 min** | 27.7 min |
| % atendidas (referencia) | 85.2% (115/135) | 91.9% (124/135) | **91.7%** (198/216) | 84.3% (182/216) |
| Conservación | OK | OK | OK | OK |

**Las dos semillas cuentan historias opuestas, a propósito.** En 1001 la base espera menos (9.4 vs. 11.8 min) -- uno de los 4 casos (de 5) donde el reward agregado le da la razón a la base (sección 2). En 1004 pasa lo contrario, y por mucho: el agente espera 18.2 min contra 27.7 de la base, y atiende más gente (91.7% vs 84.3%) -- es la semilla donde la base tiene su peor corrida de las diez (reward -4091, sección 2), y acá se ve por qué: a minuto 460 la cola `K->B` de la base tiene 33 personas esperando hace **37.9 minutos**, contra un backlog más repartido del lado del agente en el mismo instante (`output/percentiles_semilla1004.csv`, `output/percentiles_semilla1001.csv`).

**Percentiles, cada semilla por separado:**

| Métrica | Percentil | 1001 -- Agente | 1001 -- Base | 1004 -- Agente | 1004 -- Base |
|---|---|---|---|---|---|
| Espera | p50 | 11.3 min | 8.5 min | 14.3 min | 23.5 min |
| Espera | p90 | 26.4 min | 20.3 min | 33.7 min | 52.8 min |
| Espera | p95 | 28.3 min | 25.3 min | **37.3 min** | **59.2 min** |
| Sistema | p95 | 36.3 min | 31.3 min | **49.3 min** | **69.2 min** |
| Sistema | máx | 38.4 min | 31.3 min | 66.8 min | 69.7 min |

En 1004, el agente le saca a la base casi **22 minutos en p95 de tiempo en sistema** (49.3 vs. 69.2) -- la misma ventaja de cola de distribución que se ve en el agregado (sección 2.1), pero en una corrida concreta, no como promedio de cinco.

**Gráficas** (`output/*_semilla{1001,1004}.html`, agente y base por separado): `wait_profile_*`, `fleet_occupancy_*`, `pct_served_heatmap_comparado_*`, `reward_breakdown_*`, `backlog_by_pair_*` -- mismos nombres de archivo que antes, con el número de semilla al final.

**Visualización paso a paso:** `animacion_{agente,base}_semilla{1001,1004}.gif` (los pasos completos de cada corrida), inspector en minutos concretos, y el reproductor interactivo completo.

- **Inspector -- bug corregido.** Los minutos que se piden son RELATIVOS al inicio de la corrida (`hora_ini_min + 40`, `hora_ini_min + 100` -- 400 y 460 para `escalon_1`), no minutos absolutos del día. Antes se pedían `inspeccionar(40, ...)` / `inspeccionar(100, ...)` a secas -- como `escalon_1` arranca en el minuto 360 (06:00), los dos caían muy antes de que la corrida empezara, y el inspector siempre mostraba el mismo primer frame (minuto 360) sin importar cuál de los dos se pidiera -- parecía que "no mostraba nada". Con los minutos corregidos, cada llamada muestra un instante real y distinto de la corrida (ver la salida del notebook, cada una con barcos, colas y recompensa reales, distintas entre sí).
- **Reproductor -- por qué se quedaba cargando mucho tiempo.** El cómputo de cada frame es rápido (~35 ms medido), así que no era la causa. La causa más probable: `IntSlider` (la barra de tiempo) dispara una actualización por CADA posición intermedia mientras se arrastra con el mouse, no solo al soltar -- arrastrar rápido de punta a punta de la corrida puede encolar decenas de redibujados, y aunque cada uno sea rápido, la cola tarda minutos en drenarse. Se corrigió (`simulador/src/visualizacion.py`, `reproductor_interactivo`): la barra ahora solo actualiza al SOLTAR el mouse (`continuous_update=False`) -- el botón `▶` de reproducción automática no se ve afectado (avanza un paso a la vez con su propio temporizador, no por arrastre). Además, el mapa ahora se renderiza a PNG explícito en vez de con el protocolo de "rich display" por defecto de matplotlib, más liviano y predecible en notebooks remotos. **Solo funciona con un kernel de Jupyter vivo** -- un notebook ya ejecutado y guardado (como los que produce `nbconvert`, que es como se corrieron los de este proyecto) no tiene kernel corriendo, así que ahí los botones no van a responder.

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
