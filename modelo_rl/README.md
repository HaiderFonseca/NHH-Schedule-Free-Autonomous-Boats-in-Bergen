# Agente PPO -- entrenamiento, experimentos de recompensa, corrida final

**Qué es esto, en una frase:** todo lo del agente de RL que compite contra la política de referencia (`politica_base/`) -- el wrapper de demanda aleatoria necesario para entrenar, la secuencia de experimentos que decidió la forma de la recompensa, y la corrida larga final.

---

## 1. Cómo está organizado

```
modelo_rl/
├── README.md                                     # este archivo
├── src/
│   └── entrenamiento.py                          # EntornoDemandaAleatoria -- demanda fresca por reset
├── notebooks/
│   ├── 01_enfoque_y_entrenamiento_final.ipynb    # explica PPO/wrapper, corre la corrida LARGA final
│   ├── 02_secuencia_experimentos_reward.ipynb    # experimentos A/B/C sobre la forma de la recompensa
│   └── 03_verificacion_toy.ipynb                 # chequeo mínimo en una instancia de juguete (apéndice)
└── output/
    ├── modelo_final/                              # modelo entrenado, VecNormalize, curva de entrenamiento
    └── experimentos/                               # monitores y resumen de la secuencia A/B/C
```

Los notebooks importan `simulador/src` (motor) y `politica_base/src` (para evaluar contra la referencia dentro de la propia secuencia de experimentos) además de `src/entrenamiento.py` propio, y leen `simulador/config/instance.yaml` -- única fuente de verdad de los hiperparámetros de PPO, la definición de los experimentos, y la corrida final (todos bajo la clave `agente`).

---

## 2. Por qué PPO, y por qué necesita un wrapper de demanda

**PPO (Proximal Policy Optimization), no DQN.** El `action_space` del entorno (`simulador/src/env.py`) es `MultiDiscrete` (una acción por barco -- ver `simulador/README.md` sección 4.3) -- el DQN de Stable-Baselines3 no soporta `MultiDiscrete` de forma nativa, PPO sí.

**Demanda fresca en cada `reset()` (`src/entrenamiento.py`) -- necesario para RL.** `politica_base/` corre sobre demanda **fija**: se genera UNA VEZ y se guarda en un CSV que los notebooks solo leen. Eso es correcto para verificar el simulador (la política no tiene nada que ajustar), pero **no sirve para entrenar un agente**: si todos los episodios de entrenamiento fueran siempre la misma tabla de grupos, el agente podría memorizar esa realización particular (qué grupo exacto aparece en qué minuto exacto) en vez de aprender una política que generaliza sobre el patrón de demanda real.

`EntornoDemandaAleatoria(SimuladorBarcosBergen)` -- misma subclase sirve para las dos cosas que hacen falta, según cómo se llame `reset()`:

- **`reset(seed=None)`** (el caso normal en un loop de entrenamiento -- SB3 no manda una semilla en cada episodio): genera una demanda NUEVA cada vez, con una semilla sacada de una secuencia propia (reproducible si se dio `semilla_entrenamiento` al construir el entorno). Cada episodio ve una realización distinta del mismo patrón de fondo -- muestreo Montecarlo de episodios.
- **`reset(seed=42)`** (caso explícito -- evaluación, comparación contra la política base): genera la demanda con ESA semilla exacta, reproducible igual que `SimuladorBarcosBergen` hoy. Dos instancias distintas de `EntornoDemandaAleatoria` con `reset(seed=misma_semilla)` producen la MISMA demanda -- es lo que hace posible comparar el agente y la política base en igualdad de condiciones (`comparacion/`).

**Por qué es un wrapper y no un cambio a `env.py`:** `simulador/src/env.py` recibe un `grupos_df` ya construido y nunca sabe nada de `demand/` (rutas, `intensidad_od`, `conexiones_fuertes`, etc.). Meter la generación de demanda ahí complicaría el motor base que `politica_base/` sigue usando tal cual. `EntornoDemandaAleatoria` solo cambia una cosa (qué `grupos_df` usa cada episodio); todo lo demás (acciones, transición, recompensa) es exactamente `SimuladorBarcosBergen` sin tocar.

**`VecNormalize`.** El entorno se envuelve en `DummyVecEnv` + `VecNormalize` (normaliza observaciones y recompensa con estadísticas corridas) -- sin esto, un primer entrenamiento de diagnóstico aprendió una política degenerada (0% atendidas incluso en una instancia de juguete). Es parte de `agente.hiperparametros` (`usar_vecnormalize: true`) y NO varía entre los experimentos de la sección 3 ni la corrida final -- solo cambia la recompensa entre ellos.

---

## 3. Secuencia de experimentos: forma de la recompensa (`02_secuencia_experimentos_reward.ipynb`)

Antes de gastar el único entrenamiento largo del ciclo, se exploró METÓDICAMENTE la forma de la recompensa sobre `escalon_1` (franja mañana, 2 barcos) -- entrenamientos CORTOS (`agente.experimentos.timesteps_exploracion`, 30 000 pasos, ~2 min de entrenamiento cada uno) para poder iterar rápido, todos evaluados contra la política base sobre las mismas 5 semillas de evaluación de siempre.

**La secuencia, y la pregunta que responde cada paso** (config completo en `simulador/config/instance.yaml` → `agente.experimentos`):

- **A** -- solo penalización de tiempo, CON techo por persona (el valor de producción, `simulador/README.md` sección 4.4). ¿Qué tan bien sirve el agente con la recompensa más simple posible?
- **B** -- lo mismo, pero SIN techo por persona. Hipótesis: mejor que A, porque sin techo la recompensa sigue distinguiendo entre alguien muy atrasado y alguien apenas atrasado incluso después de que ambos superen la tolerancia de 12 min -- con techo, alguien que espera 35 min y alguien que espera 50 min penalizan exactamente igual.
- **C1/C2** -- sobre el ganador de A/B, se agrega una penalización de movimiento PEQUEÑA (0.003 y 0.001 respectivamente). Análisis de escala antes de correr el experimento: la penalización de una persona va de 0 a 1 según `(sobrante/18)²` -- alguien con ~13 min de espera ya penaliza `(1/18)² ≈ 0.003`, así que una penalización de movimiento del mismo orden (0.003) o más chica (0.001) es necesaria para que mover un barco no pese, en la recompensa, tanto como dejar a alguien esperando.

**Resultado y decisión** (narrativa completa, con las tablas reales, al final de `02_secuencia_experimentos_reward.ipynb`):

| Experimento | `peso_movimiento` | Techo por persona | % atendidas (30k pasos) | Espera media | Movimientos |
|---|---|---|---|---|---|
| A | 0.0 | activo (1.0) | 47.8% | 26.2 min | 135 |
| B | 0.0 | inactivo | 51.9% | 21.9 min | 125 |
| C1 | 0.003 | inactivo | 52.0% | 23.8 min | 127 |
| C2 | 0.001 | inactivo | 44.7% | 22.5 min | 95 |

**Nota sobre el % atendidas de esta tabla:** se usa acá solo como un indicador rápido para comparar A/B/C1/C2 ENTRE SÍ, no como la métrica que de verdad importa -- el simulador no pierde a nadie (nadie se retira nunca, `simulador/README.md` sección 4.5), así que "% atendidas" depende en parte de dónde cae el corte de la ventana de la corrida, no solo de qué tan bien despacha la política. Lo que la recompensa realmente penaliza, y lo que se reporta como resultado real en `comparacion/README.md`, es tiempo en sistema (medio y máximo) y movimientos.

**A vs. B: se confirma la hipótesis** -- quitar el techo mejora las tres columnas (más atendidas, menos espera media, incluso menos movimientos). **C1 vs. C2: el punto de partida (0.003) rinde mejor que el más chico (0.001)** -- C1 prácticamente empata con B en % atendidas mientras mantiene movimiento similar a B (disciplina de movimiento sin perder desempeño); C2, con una penalización aún más chica, no rinde mejor -- no hay evidencia de que 0.001 sea sistemáticamente mejor en ningún eje a esta escala de entrenamiento. **Ganador: C1** (`peso_movimiento: 0.003`, sin techo por persona) -- combo copiado a `agente.entrenamiento.recompensa_overrides` para la corrida final.

**Sobre la brecha frente a la política base (88.0% atendidas) en esta secuencia:** ningún experimento se acerca a la base -- son entrenamientos de solo 30 000 pasos (una sola semilla de entrenamiento), pensados exclusivamente para comparar la FORMA de la recompensa entre sí, no para producir un agente competitivo. Ver sección 4 para el resultado de la corrida larga (150 000 pasos, 5x más) con la recompensa ya elegida aquí.

---

## 4. Entrenamiento final (`01_enfoque_y_entrenamiento_final.ipynb`)

`agente.entrenamiento` en `simulador/config/instance.yaml`:

```yaml
entrenamiento:
  escalon_base: "escalon_1"
  semilla_entrenamiento: 123     # raíz de la secuencia de demandas de entrenamiento
  total_timesteps: 150000        # ~1666 episodios de 90 pasos
  recompensa_overrides:          # combo ganador de la secuencia de experimentos (sección 3)
    peso_movimiento: 0.003
    penalizacion_maxima_persona: 1000000.0
    premio_por_persona_entregada: 0.0
```

Hiperparámetros de PPO (`agente.hiperparametros`, compartidos con TODOS los experimentos de la sección 3): `usar_vecnormalize: true`, `ent_coef: 0.01` (más exploración que el default 0.0 de SB3), `learning_rate: 0.0003` (el estándar de PPO, 3e-4). `n_steps: 512` viene de una ronda de diagnóstico anterior, elegido más chico que el default de SB3 (2048) para no esperar ~23 episodios entre actualizaciones -- pero 512 en sí no sale de ninguna fórmula ni referencia, es una elección sin más justificación que "funcionó". La alternativa más directa de defender es `n_steps=90`: exactamente un episodio completo de `escalon_1`, así que cada actualización cae justo al cerrar un episodio, nunca a mitad de uno. Sección 4.1 compara las dos.

**El notebook, en orden:** (1) confirma demanda fresca por reset (sección 2); (2) `check_env` de SB3 (chequeo estándar antes de entrenar); (3) envuelve el entorno con `Monitor` + `DummyVecEnv`/`VecNormalize`; (4) entrena `PPO("MlpPolicy", ...)`; (5) guarda el modelo (`output/modelo_final/modelo_ppo.zip`, `vecnormalize.pkl`) y grafica la curva de recompensa por episodio (interactiva, `output/modelo_final/training_curve.html`).

**Resultado de la corrida final (150 000 timesteps, 1666 episodios de 90 pasos, combo C1):** la recompensa media de los primeros 20 episodios fue -24 871.4; la de los últimos 20, -2 832.8 -- una mejora de casi un orden de magnitud. Los últimos 5 episodios individuales: -17 540.6, -1 905.5, -651.3, -1 007.3, -958.0 -- la mayoría ya estabilizados en un rango bajo, con episodios ocasionales peores (demanda más pesada según la semilla de ese episodio, dentro de lo esperable con demanda fresca en cada `reset`). Curva completa (interactiva) en `output/modelo_final/training_curve.html`. Ver `comparacion/README.md` para el resultado de este modelo comparado contra la política base.

### 4.1 `n_steps=90` en vez de 512 -- se probó, empeoró en evaluación

512 no salía de ninguna cuenta (sección 2) -- la alternativa que sí se puede defender con un argumento es `n_steps=90`, exactamente un episodio de `escalon_1`, así que cada actualización de PPO cae justo al cerrar un episodio. Se corrió la misma corrida larga (150 000 timesteps, mismo combo C1) cambiando solo eso -- guardada aparte en `output/modelo_final_n90/`, sin pisar el modelo de arriba.

**En entrenamiento se ven parecidos:** recompensa media de los últimos 20 episodios, -2 832.8 (n_steps=512) vs. -2 956.2 (n_steps=90) -- la curva sola no diría cuál es mejor. **En evaluación (5 semillas, mismas de siempre) se separan mucho:**

| | n_steps=512 | n_steps=90 | Política base |
|---|---|---|---|
| Tiempo en sistema medio | 29.0 min | 30.2 min | 26.1 min |
| Tiempo en sistema máximo | 60.8 min | **73.7 min** | 69.7 min |
| Espera media | 17.1 min | 19.9 min | 16.4 min |
| Movimientos (5 episodios) | 191 | 179 | 155 |
| Reward (5 episodios) | -3772.0 | **-28082.5** | -5259.0 |
| % atendidas (referencia) | 88.0% | 77.5% | 88.0% |

`n_steps=90` pierde en todo, incluido el peor caso -- 73.7 min, peor que la propia política base (69.7). El reward de evaluación (-28082) está en el rango de los primeros episodios de entrenamiento del otro modelo, no cerca de donde terminó. O sea: entrenó bien (la curva sube) pero generalizó mal a las semillas de evaluación.

**¿Por qué?** Con `n_steps=90`, el buffer de PPO junta exactamente UN episodio antes de actualizar, así que en la misma corrida hace ~1666 actualizaciones (una por episodio) en vez de ~293 con n_steps=512 -- cada una con muchos menos datos. Actualizar tan seguido con lotes tan chicos hace que cada paso de gradiente sea más ruidoso y esté más pegado a la demanda particular de ESE episodio -- puede parecer que mejora durante el entrenamiento (la curva sube) sin que la política se vuelva más robusta a demanda que no vio. `n_steps=512` termina dándole a cada actualización una muestra más ancha y menos ruidosa de experiencia, y eso pesa más que alinear las actualizaciones con el cierre de cada episodio.

**Se queda con `n_steps=512`.** El modelo que se usa en `comparacion/` es el de `output/modelo_final/` -- `n_steps=90` queda documentado como un resultado real que se probó y no funcionó mejor, no como el modelo final.

### 4.2 Tiempos de entrenamiento (medidos, no estimados)

Cada corrida guarda su propio `metadata_entrenamiento.json` (`output/modelo_final/`, `output/modelo_final_n90/`, y uno por experimento en `prueba_rewards/output/modelos/{A,B,C,D}/`) con el tiempo real medido (`time.time()` alrededor de `model.learn(...)`), no una estimación -- así este dato queda fijo, sin depender de volver a correr nada para saberlo.

| Corrida | `total_timesteps` | `n_steps` | Actualizaciones de PPO (`total_timesteps // n_steps`) | Tiempo medido | Pasos/seg |
|---|---|---|---|---|---|
| Final (`modelo_final/`) | 150 000 | 512 | **292** | 12.7 min (761 s) | 197 |
| Prueba `n_steps=90` (`modelo_final_n90/`) | 150 000 | 90 | **1666** | 13.8 min (828 s) | 181 |
| Experimentos A/B/C1/C2 (`../output/experimentos/`, cerrados) | 30 000 c/u | 512 | **58** c/u | 2.1-2.3 min c/u | 217-236 |
| `prueba_rewards/` A/B/C/D | 30 000 c/u | 512 | **58** c/u | 2.1-3.8 min c/u (varía con la carga de la máquina en el momento) | 132-237 |

**La cuenta que importa no es el tiempo de reloj, es cuántas actualizaciones de la red hace PPO.** Con `n_steps=512`, cada actualización consume 512 pasos de simulación; `total_timesteps // n_steps` da el número de veces que la red (política + valor) realmente se ajusta durante toda la corrida:
- 30 000 timesteps → **58 actualizaciones**.
- 150 000 timesteps → **292 actualizaciones** -- 5 veces más.

El tiempo de reloj en sí es barato (2-4 min para 30 000 pasos, ~13 min para 150 000, en esta máquina) -- lo que de verdad falta en las corridas de 30 000 no es tiempo, son **actualizaciones**: 58 rondas de gradiente no alcanzan para que la red converja a una política competente, y por eso el % de gente atendida en TODOS los experimentos cortos (secuencia A/B/C1/C2, `prueba_rewards/`) queda muy por debajo de la política base (40-52% contra el 88% de la base, ver `prueba_rewards/README.md` sección 5) -- no es que la recompensa esté mal, es que 58 actualizaciones es, para este problema, un presupuesto de entrenamiento genuinamente corto. El agente final (`modelo_final/`, 292 actualizaciones) es el que sí iguala a la base.

**Aclaración sobre `n_steps=90`:** no es lo mismo que "58 actualizaciones" -- son dos números distintos que se pueden confundir. `n_steps` es cuántos pasos de simulación entran en CADA actualización (el tamaño del lote); las 58 actualizaciones de la tabla salen de `n_steps=512`. Si se usara `n_steps=90` con 30 000 timesteps, saldrían MÁS actualizaciones (30000/90≈333), no menos -- lotes más chicos, pero más frecuentes. De hecho ya se probó `n_steps=90` con el presupuesto largo (150 000 pasos, sección 4.1): da 1666 actualizaciones (5.7x más que las 292 de `n_steps=512`) y aun así generaliza PEOR -- más actualizaciones no es automáticamente mejor si cada una ve muy poca experiencia (ver la explicación completa en 4.1). La causa de que las corridas de 30 000 anden mal es la combinación de POCAS actualizaciones (58) Y presupuesto corto en general, no `n_steps` por sí solo.

---

## 5. Cómo correr

```bash
cd modelo_rl/notebooks
# 1. Secuencia de experimentos -- decide la recompensa de la corrida final
jupyter nbconvert --to notebook --execute --inplace 02_secuencia_experimentos_reward.ipynb
# 2. (si el ganador cambia) actualizar simulador/config/instance.yaml -> agente.entrenamiento.recompensa_overrides
# 3. Entrenamiento largo final
jupyter nbconvert --to notebook --execute --inplace 01_enfoque_y_entrenamiento_final.ipynb
```

Necesita `pip install stable-baselines3` (instala PyTorch como dependencia), y que `demand/output/`, `bergen-boats/02_ruteo_navegable/output/` y `politica_base/notebooks/00_preparar_demanda_escalones.ipynb` ya hayan corrido (la política base se usa dentro de la secuencia de experimentos como referencia de comparación). `comparacion/notebooks/05_comparacion_agente_vs_base.ipynb` necesita que `01_enfoque_y_entrenamiento_final.ipynb` ya haya guardado `output/modelo_final/modelo_ppo.zip`.

---

## 6. Supuestos y limitaciones

- **Instancia de entrenamiento chica (`escalon_1`, 2 barcos, franja mañana):** elegida para poder iterar rápido en la secuencia de experimentos y en el ciclo de diagnóstico -- no se entrenó sobre `escalon_2`/`escalon_3` (día completo / semana). Un agente entrenado en `escalon_1` no necesariamente generaliza a instancias con más barcos o distinta franja horaria.
- **Secuencia de experimentos con una sola semilla de entrenamiento por combo (30 000 pasos):** suficiente para comparar la FORMA de la recompensa entre configuraciones cercanas, pero cada número individual de la tabla de la sección 3 tiene la variabilidad propia de una sola corrida corta -- no se promedió sobre múltiples semillas de entrenamiento por experimento (sí se evalúa cada uno sobre las 5 semillas de evaluación fijas, que no varían).
- **`penalizacion_maxima_persona` en la recompensa de RL vs. producción:** el combo ganador (C1) desactiva el techo (`1000000.0`, efectivamente sin techo) -- distinto del valor de producción (`1.0`) que sigue gobernando `politica_base/` y los escalones 1-3, sin cambios. Los overrides de `agente.entrenamiento`/`agente.experimentos` nunca tocan `simulador/config/instance.yaml` → `recompensa` (la sección de producción).
