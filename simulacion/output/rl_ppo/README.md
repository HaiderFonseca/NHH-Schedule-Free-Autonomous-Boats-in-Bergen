# Qué hay en esta carpeta, y cómo se generó

Esta carpeta (`simulacion/output/rl_ppo/`) es la salida del entrenamiento del agente
de RL (PPO, `notebooks/04_entrenamiento_rl.ipynb`, y su verificación previa
`04a_verificacion_toy_rl.ipynb`). Este documento explica, a fondo, qué entra, qué
sale, qué semillas se usan y cómo está configurado cada parámetro -- para no tener
que ir a leer el código para entender qué se corrió.

---

## 1. Qué archivos hay acá y de dónde sale cada uno

| Archivo | Lo genera | Qué es |
|---|---|---|
| `modelo_ppo.zip` | `04_entrenamiento_rl.ipynb` | El modelo PPO entrenado (pesos de la red + hiperparámetros), formato nativo de Stable-Baselines3 (`PPO.save`/`PPO.load`). |
| `vecnormalize.pkl` | `04_entrenamiento_rl.ipynb` (solo si `usar_vecnormalize: true`) | Las estadísticas corridas (media/varianza) de `VecNormalize` -- necesarias para evaluar el modelo con la MISMA normalización que vio durante el entrenamiento (ver sección 4). |
| `monitor.monitor.csv` | `04_entrenamiento_rl.ipynb` (vía `stable_baselines3.common.monitor.Monitor`) | Una fila por episodio de entrenamiento: recompensa total, duración, tiempo transcurrido. Es la fuente de la curva de entrenamiento. |
| `training_curve.html` | `04_entrenamiento_rl.ipynb` | Curva de recompensa por episodio (interactiva -- zoom, hover, media móvil de 20 episodios), de `monitor.monitor.csv`. Solo `.html`, sin `.png` -- ver nota técnica en `simulacion/README.md` sección 14.1 sobre por qué se sacó la exportación estática (kaleido). |
| `monitor_toy.monitor.csv` | `04a_verificacion_toy_rl.ipynb` | Igual que `monitor.monitor.csv`, pero de la corrida de verificación mínima sobre `escalon_toy` (B.1 del diagnóstico) -- **no se mezcla** con la de arriba: cada notebook lee su propio archivo por nombre, nunca la carpeta completa, para no juntar corridas distintas por accidente. |
| `bitacora_experimentos.md` | Manual, durante el diagnóstico (no lo genera un notebook) | Registro fila por fila de cada experimento del ciclo de diagnóstico (B.1-B.7): qué cambió, y el resultado agente-vs-base de cada uno. Es la traza completa de "qué se probó y qué mejoró" -- este README explica el SETUP; la bitácora explica el PROCESO que llevó a ese setup. |
| `README.md` | Este archivo | -- |

---

## 2. Qué es `EntornoDemandaAleatoria` y por qué existe (el input del agente)

El agente entrena sobre `simulacion/src/entrenamiento.py`, clase `EntornoDemandaAleatoria`
-- una subclase de `SimuladorBarcosBergen` (el mismo entorno que usan los escalones
1-3 y la política base) que cambia UNA sola cosa: **regenera la demanda en cada
`reset()`** en vez de usar una tabla fija.

- `reset(seed=None)` (el caso normal durante el entrenamiento -- SB3 no manda una
  semilla en cada episodio): genera una demanda NUEVA, distinta a la de episodios
  anteriores, con una semilla sacada de una secuencia propia (ver sección 3).
- `reset(seed=X)` (caso explícito -- evaluación, verificación): genera la demanda
  con ESA semilla exacta, reproducible.

**Por qué hace falta esto:** si el agente entrenara siempre sobre la misma tabla de
grupos, podría memorizar esa realización particular (qué persona exacta aparece en
qué minuto exacto) en vez de aprender una política que generaliza sobre el patrón
de demanda real. Con demanda fresca, cada episodio de entrenamiento es una muestra
Montecarlo distinta del mismo patrón de fondo (gravedad + SSB + Poisson, igual que
`demand/`) -- el agente tiene que aprender la ESTRUCTURA del problema, no una
instancia de ella. Ver `simulacion/README.md`, sección 11, para el detalle completo.

---

## 3. Semillas -- cuáles hay, para qué sirve cada una, y por qué no se cruzan

| Semilla / secuencia | Vive en | Para qué |
|---|---|---|
| `agente.entrenamiento.semilla_entrenamiento` (config, hoy `123`) | Raíz de la secuencia de demandas de ENTRENAMIENTO. `EntornoDemandaAleatoria` la usa para inicializar un `numpy.random.default_rng(semilla_entrenamiento)` propio; cada `reset()` sin semilla explícita saca el SIGUIENTE número de esa secuencia para generar la demanda de ese episodio. | Reproducibilidad de la SECUENCIA completa de episodios de entrenamiento (mismo `semilla_entrenamiento` -> mismos ~1000+ episodios de demanda, en el mismo orden) -- útil para poder repetir un entrenamiento exacto si hace falta depurar algo. |
| `agente.entrenamiento.semilla_entrenamiento` (reusada) también como `seed=` de `PPO(...)` | Semilla de la RED (inicialización de pesos, muestreo de acciones durante el entrenamiento) | Se reusa el mismo número por simplicidad -- son dos fuentes de aleatoriedad independientes (una para la demanda, otra para PyTorch/PPO), no hay razón para que compartan el número salvo conveniencia; si hace falta desacoplarlas más adelante, son campos separados en el código. |
| `agente.evaluacion.semillas` (config, hoy `[1001, 1002, 1003, 1004, 1005]`) | 5 semillas FIJAS, usadas por `05_comparacion_agente_vs_base.ipynb` para generar 5 demandas de evaluación, una por semilla, vía `reset(seed=semilla)`. | Comparar el agente y la política base sobre la MISMA demanda (misma semilla -> misma tabla de grupos para los dos, ver `entrenamiento.py`) -- así la comparación es justa. Deliberadamente fuera del rango que recorre `semilla_entrenamiento` (que empieza en 123 y avanza con `default_rng`, nunca llega a la zona de 1001-1005 en las corridas de esta escala) -- el agente nunca vio estas demandas exactas durante el entrenamiento. |
| `42` (semilla de verificación puntual, usada en varias celdas de `04_entrenamiento_rl.ipynb` y en `experimento_toy.py`) | Una demanda de referencia fija, para confirmar a ojo que `reset(seed=X)` es reproducible, y para comparaciones rápidas fuera del ciclo formal de evaluación. | No se usa para el resultado final reportado (eso es siempre el promedio sobre `agente.evaluacion.semillas`) -- es solo una demanda de referencia para checks rápidos durante el desarrollo. |

---

## 4. Parámetros de config -- qué hace cada uno, y por qué tiene ese valor

Todo vive en `simulacion/config/instance.yaml` -> `agente` y `recompensa`. Nada
hardcodeado en los notebooks (ver tabla de la sección 3 del README principal,
"de dónde sale cada parámetro").

### `agente.gamma` (0.99)

Factor de descuento estándar de RL -- ya estaba declarado desde antes de conectar
PPO (la especificación §10 lo pedía). Se pasa tanto a `PPO(gamma=...)` como a
`VecNormalize(gamma=...)` (la normalización de recompensa también necesita saber
el horizonte de descuento para escalar bien).

### `agente.entrenamiento`

| Campo | Qué hace | Valor |
|---|---|---|
| `escalon_base` | De qué escalón saca horas/porcentaje de población/número de barcos la instancia de entrenamiento -- **el mismo valor lo lee `04_entrenamiento_rl.ipynb` (para entrenar) y `05_comparacion_agente_vs_base.ipynb` (para evaluar)**, y desde el diagnóstico hay un `assert` explícito en el notebook 05 que compara la FORMA del vector de observación del modelo cargado contra la de la instancia de evaluación -- si algún día se entrena en un escalón y se evalúa en otro por error, esto lo detiene con un mensaje claro, no deja pasar un resultado silenciosamente incomparable. | `"escalon_1"` |
| `semilla_entrenamiento` | Ver sección 3. | `123` |
| `total_timesteps` | Cuántos pasos de simulación entrena PPO en total (no episodios -- un episodio de `escalon_1` son 90 pasos, así que esto son ~timesteps/90 episodios). | `150000` (B.7 -- se probó 300000 primero, pero a la velocidad medida en esta máquina (~73 steps/seg en `escalon_1`) hubiera tomado ~68 min; 150000 (~34 min) fue el compromiso práctico, ver bitácora) |
| `usar_vecnormalize` | Si `true`, envuelve el entorno en `DummyVecEnv` + `VecNormalize` (normaliza observaciones y recompensa con estadísticas corridas) antes de pasárselo a PPO. **Se agregó durante el diagnóstico** -- la primera corrida (sin esto) aprendió una política degenerada; con esto, incluso en la instancia de juguete el agente pasó de 0% a igualar la política base. Ver `bitacora_experimentos.md`. | `true` |
| `ent_coef` | Coeficiente de entropía de PPO -- sube la exploración (el default de SB3 es `0.0`, sin ningún incentivo extra a explorar). | `0.01` |
| `learning_rate` | Tasa de aprendizaje del optimizador de PPO. | `0.0003` (3e-4, el estándar de PPO -- no se tocó salvo que algún experimento mostrara inestabilidad, ver bitácora) |
| `n_steps` | Cuántos pasos de simulación junta PPO antes de cada actualización de la red. El default de SB3 (2048) implica esperar a que se acumulen ~22 episodios completos de `escalon_1` (90 pasos c/u) antes de aprender nada -- se bajó para actualizar con más frecuencia relativa al horizonte corto de este problema. | `512` (~5-6 episodios por actualización) |

### `agente.evaluacion.semillas`

Ver sección 3. `[1001, 1002, 1003, 1004, 1005]`.

### `recompensa.premio_por_persona_entregada`

Término nuevo, positivo, agregado durante el diagnóstico (ver
`simulacion/src/recompensa.py`, historia de cambios en su docstring, y
`simulacion/README.md` sección de diagnóstico): antes de esto, la recompensa era de
puro castigo (nunca positiva) -- no había ninguna señal explícita de "esto salió
bien", solo distintos grados de "esto salió mal". Con este término, cada persona
que un barco entrega a destino en un paso suma `premio_por_persona_entregada` a la
recompensa de ESE paso. Default `0.0` (inactivo -- así los escalones 1-3 y la
política base, ya corridos antes de este cambio, siguen siendo válidos sin volver a
correrlos); se activa (>0) específicamente para las corridas de entrenamiento de
esta fase. Valor usado: ver `bitacora_experimentos.md`.

### `recompensa.peso_movimiento`

Ya existía (penalización por barco navegando en el paso, default `0.1`). Durante el
diagnóstico se probó también en `0.0` (B.4 -- "para que el agente aprenda a servir
antes de optimizar eficiencia", como pediste), **como experimento temporal, no como
valor de producción** -- si la bitácora muestra que ayuda, el valor final usado
para B.7 (y de ahí en adelante) queda documentado ahí, con la aclaración de que es
una decisión de esta fase de diagnóstico, revisable después.

---

## 5. Cómo se evalúa (qué compara contra qué)

`05_comparacion_agente_vs_base.ipynb`, para cada una de las 5 semillas de
`agente.evaluacion.semillas`:

1. Construye una demanda con esa semilla (misma función, mismos parámetros de
   `escalon_base`, para el agente y para la política base -- garantiza demanda
   IDÉNTICA para los dos).
2. Corre el agente (`model.predict(obs, deterministic=True)` -- política
   determinística, sin exploración, es la que se reporta) hasta que el episodio
   trunca. Si el modelo se entrenó con `VecNormalize`, la evaluación usa las
   estadísticas GUARDADAS (`vecnormalize.pkl`), congeladas (`training=False`, no se
   siguen actualizando con datos de evaluación) y comparando sobre recompensa CRUDA
   (`norm_reward=False`) -- para que el número sea comparable 1:1 contra la
   política base, que nunca pasa por ninguna normalización.
3. Corre la política base (`asignar_flota`) sobre la misma demanda.
4. Junta las 5 corridas de cada política con `metricas.combinar_corridas` (misma
   función que usa el escalón 3 para juntar una semana) y saca las métricas
   completas de siempre (`metricas.reporte_completo`) -- % atendidas, espera,
   tiempo en sistema, ocupación, backlog, todo lo que ya existía, cero funciones de
   métricas nuevas.

Resultado final: `simulacion/output/comparacion/tabla_comparativa.csv` (y su
narrativa en `simulacion/README.md`, sección 13).
