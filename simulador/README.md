# Simulador de despacho de barcos (motor: entorno Gymnasium, estado, recompensa, métricas)

**Qué es esto, en una frase:** el motor que modela minuto a minuto (pasos de 2 min) qué hacen los barcos y los pasajeros en Bergen -- nada de políticas ni de experimentos, eso vive en `politica_base/`, `modelo_rl/` y `comparacion/` (ver la raíz del proyecto para la vista de los 4 bloques).

**Documento fuente (manda sobre cualquier duda):** `docs/especificacion_simulador_rl.md`. Todo lo de abajo está mapeado a sus secciones (§1-§12).

---

## 1. Cómo está organizado

```
simulador/
├── README.md                 # este archivo
├── config/instance.yaml      # ÚNICA fuente de verdad de los parámetros compartidos por los 4 bloques
└── src/
    ├── unidades.py            # qué es una "Unidad" (persona o grupo) y cómo se generan
    ├── estado.py               # la foto del mundo en cada instante + el vector aplanado (§4)
    ├── recompensa.py           # la fórmula de puntaje (§7)
    ├── env.py                  # el simulador en sí -- Gymnasium (junta todo lo anterior, §3/§5/§6)
    ├── metricas.py              # verificación de conservación + métricas detalladas + combinar_corridas
    └── visualizacion.py         # mapa animado, gráficas interactivas, inspector de un minuto concreto
```

`config/instance.yaml` es leído (nunca copiado) por los otros 3 bloques vía ruta relativa (`../simulador/config/instance.yaml`) -- misma regla que ya usaba `demand/`: cada carpeta es dueña de sus propios parámetros, las demás los leen.

**Los otros 3 bloques del proyecto:**

- **`politica_base/`** -- la política de referencia (`asignar_flota`, regla fija "nearest-available" + coordinación de flota) y su verificación paso a paso (notebooks 00-03, los "escalones" de demanda creciente).
- **`modelo_rl/`** -- el agente PPO: el wrapper de demanda aleatoria para entrenar, la secuencia de experimentos sobre la forma de la recompensa, y el entrenamiento largo final.
- **`comparacion/`** -- agente PPO vs. política base, sobre las mismas semillas de evaluación.

**Cómo leer esto si nunca programaste en Python:** cada archivo de `src/` es una "pieza" con una responsabilidad. Nada se ejecuta solo con crear los archivos -- hace falta un notebook (en `politica_base/`, `modelo_rl/` o `comparacion/`) que las importe y las corra.

**Qué es esa primera línea `from __future__ import annotations`:** instrucción del lenguaje Python, no algo específico de este proyecto -- le dice a Python que trate las anotaciones de tipo (`nodos: list[str]`, `politica: str | None`) como texto en vez de evaluarlas de inmediato, evitando errores en casos donde una clase se refiere a sí misma antes de terminar de definirse. Práctica estándar en Python moderno.

---

## 2. Qué es Gymnasium (y qué NO es)

**Gymnasium no trae simuladores prearmados para elegir.** Es una librería que define un **molde estándar**: cualquier "entorno" (environment) debe ser una clase de Python con exactamente dos métodos:

- `reset()` → arranca el mundo desde cero, devuelve el estado inicial.
- `step(accion)` → recibe una acción, avanza el mundo un paso, devuelve `(nuevo_estado, recompensa, terminó, se_truncó, info_extra)`.

Ese molde es TODO lo que Gymnasium impone. Nosotros escribimos el 100% de la lógica real (qué hace un barco, cómo suben y bajan los pasajeros) en `src/env.py`, clase `SimuladorBarcosBergen`. Gymnasium no sabe nada de barcos ni de Bergen -- solo garantiza que nuestra clase "hable el mismo idioma" que espera cualquier librería de RL después (Stable-Baselines3, ya conectada -- ver `modelo_rl/README.md`). Es como un enchufe: Gymnasium estandariza la forma del enchufe, nosotros construimos el aparato.

**El "tipo" de simulación** (una pregunta distinta, sobre la mecánica interna, no sobre Gymnasium): es una **simulación de tiempo discreto en pasos fijos de 2 minutos** -- cada `step()` siempre avanza exactamente 2 minutos, nunca menos ni más (a diferencia de una simulación "por eventos", que salta directo al próximo momento interesante). La especificación (§3) pide explícitamente pasos fijos, por simplicidad y porque encaja natural con el `step()` de Gymnasium.

**¿Sí estamos usando Gymnasium de verdad, o solo el nombre?** Sí, de verdad: `src/env.py`, `class SimuladorBarcosBergen(gym.Env):` -- hereda literalmente de la clase base de la librería (`import gymnasium as gym`). Eso obliga (y verifica en tiempo de ejecución) a que la clase tenga `action_space` y `observation_space` bien declarados (`MultiDiscrete` para las acciones, `Box` para el vector aplanado) y los métodos `reset()`/`step()` con la firma exacta que la librería espera. Si `SimuladorBarcosBergen` no cumpliera el contrato, Stable-Baselines3 no podría usarla -- de hecho `check_env` (`modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb`) lo confirma en tiempo de ejecución.

**¿Qué semilla estamos usando?** `config/instance.yaml` → `semilla: 42`. Esa semilla se usa para regenerar la DEMANDA (`demand/src/llegadas.py`), no dentro de `env.py` -- el entorno en sí no tiene ningún sorteo propio, así que `env.reset(seed=...)` no cambia nada por sí solo; lo que realmente reproduce una corrida es generar la tabla de grupos con esa semilla y correr el entorno sobre esa misma tabla.

---

## 3. De dónde sale cada parámetro (nada se inventa ni se duplica)

Regla del proyecto (ya usada en `demand/`): cada carpeta es dueña de sus propios parámetros; las demás los **leen** de la fuente, nunca los copian.

| Parámetro | Vive en | Por qué ahí |
|---|---|---|
| Capacidad del barco (20), nodo inicial (Bryggen), conexiones fuertes | `bergen-boats/config/instance.yaml` → `flota`, `garantia` | Son propiedades de la flota/física, ya definidas en el paso de ruteo |
| Tiempos de viaje entre nodos | `bergen-boats/02_ruteo_navegable/output/matriz_tiempos_min.csv` | Ya calculados (Dijkstra sobre agua real), nunca se recalculan aquí |
| Geometría de las rutas (para animar) | `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson` | Igual -- ya calculada, persistida en ese mismo trabajo |
| Patrón de quién viaja a dónde y cuándo | `demand/src/llegadas.py` + `demand/output/matriz_intensidad_od.csv` | El generador de demanda real (gravedad + SSB + Poisson), no se toca |
| **Tolerancia (12 min), normalizador fijo (18 min), techo por persona, peso de movimiento (producción)** | `simulador/config/instance.yaml` → `recompensa` | Propios de este motor (fórmula de recompensa, sección 4.4) |
| **Tamaño de flota y % de demanda por escalón** | `simulador/config/instance.yaml` → `escalones` | Propios de la verificación (no existen en `demand/` ni en `bergen-boats/`) |
| **Paso de tiempo (2 min), semilla, unidad de demanda (personas/grupos)** | `simulador/config/instance.yaml` | Propios de este motor |
| **Hiperparámetros de PPO, experimentos de recompensa, entrenamiento final, semillas de evaluación** | `simulador/config/instance.yaml` → `agente` | Compartidos por `modelo_rl/` y `comparacion/` -- ver `modelo_rl/README.md` |

Si buscas un número y no está en `simulador/config/instance.yaml`, casi seguro es porque pertenece a otro paso y aquí solo se **lee** -- revisa la tabla de arriba.

---

## 4. El MDP explicado con ejemplos reales

### 4.1 Estado (§4) -- dos versiones, una para humanos y una para la red

Cada instante del mundo se guarda en un objeto `EstadoSimulacion` (`src/estado.py`). Tiene DOS formas de mostrarse:

- **`.to_dict()`** -- legible, para nosotros (y para las funciones de `metricas.py`/`visualizacion.py`, que leen esta forma). Ejemplo real de una corrida (recortado):
  ```json
  {
    "tiempo": {"minuto_del_dia": 412, "dia_semana": 0},
    "barcos": [
      {"origen": "L", "destino": "B", "min_para_llegar": 3.0, "ocupacion": 17, "libre": false},
      {"origen": "L", "destino": "K", "min_para_llegar": 1.0, "ocupacion": 0,  "libre": false}
    ],
    "demanda": {"K->B": {"personas": 20, "espera_max": 6.0}, "...": "..."}
  }
  ```
  `libre` sigue existiendo AQUÍ (es útil para logs/inspección humana) aunque ya no está en el vector aplanado -- ver más abajo.

- **Vector aplanado (`aplanar_estado`)** -- una lista de números de tamaño fijo (46 para 2 barcos), lo único que ve un agente de RL. `dimension_vector(num_barcos, num_nodos)` calcula el tamaño exacto: `10` valores por barco (one-hot origen + one-hot destino + `min_para_llegar` + ocupación, con 4 nodos) + `24` (12 pares O-D × 2: personas esperando y espera máxima) + `2` (tiempo: minuto del día y día de semana, cada uno un solo valor normalizado).

**Dos decisiones de diseño sobre el vector, explícitas (no accidentes):**

- **Sin codificación cíclica (seno/coseno) del tiempo.** Cada corrida simulada es un episodio INDEPENDIENTE -- nunca cruza medianoche (`env.reset()` vuelve a `hora_inicio_min`, `env` termina al llegar a `hora_fin_min`) ni cambia de día de semana a mitad de un episodio. La codificación cíclica (`sin(2π·minuto/1440)`, `cos(...)`) tiene sentido cuando el "final" y el "inicio" de una variable temporal están cerca en el mundo real (23:59 y 00:00 son el mismo instante, casi) y un agente necesita verlos como cercanos -- pero eso nunca ocurre acá: ningún episodio empieza en 23:58 y sigue hasta 00:02 del día siguiente. Usar seno/coseno solo agregaría 2 números más al vector (4 en vez de 2) sin ninguna información nueva que un valor lineal simple no dé ya. Por eso `aplanar_estado` usa `minuto_del_dia / 1440.0` y `dia_semana / 6.0` directamente -- dos escalares normalizados a `[0, 1]`, nada más.
- **Sin la bandera `libre` en el vector.** Es deducible de otros dos campos que YA están en el vector: un barco está libre exactamente cuando su nodo de origen y su nodo de destino son el mismo (el one-hot de origen == el one-hot de destino) y `min_para_llegar ≈ 0`. Repetir esa información como un tercer número no le da a la red nada que no pueda inferir de los otros dos -- se quitó para no cargar el vector con un valor redundante. (Sigue existiendo en `EstadoSimulacion.to_dict()`, donde sí es útil para logs y para que `politica_base`/`metricas.py` no tengan que re-derivarlo en cada lugar que lo necesitan.)

**Nota de compatibilidad:** este vector de 46 valores (2 barcos) es distinto en tamaño al de una versión anterior (50 valores, con seno/coseno y `libre`) -- cualquier modelo entrenado contra esa versión anterior queda obsoleto; los modelos de `modelo_rl/output/` son todos posteriores a este cambio.

### 4.2 Unidad de demanda: personas, no grupos

`demand/` genera **grupos** (ej. "3 personas que salen juntas de Kleppestø a las 6:23"). Este motor, por defecto, **explota** cada grupo en personas individuales (`src/unidades.py`, función `grupos_a_unidades`) -- cada persona hereda el mismo origen/destino/hora de llegada de su grupo, pero se trata como una solicitud independiente. Es un interruptor en el config (`unidad_demanda: "personas" | "grupos"`), no dos simuladores distintos.

`demand/` sigue calculando una paciencia por grupo (columna `espera_maxima_min`, 15/30 min ± jitter) porque es un dato propio de ese módulo -- pero este motor ya no la lee en ningún lado (sección 4.5): la `Unidad` de este paso no tiene ese campo.

### 4.3 Acciones -- viajes directos punto a punto

Cada barco libre recibe una de 5 órdenes: ir a cada uno de los nodos, o esperar. Un barco en ruta ignora cualquier orden y sigue hasta llegar (no se redirige a media ruta). Quién sube a cada barco es una regla fija del simulador (colas separadas por par origen-destino, embarque en orden de llegada hasta llenar capacidad) -- **quién decide el destino de cada barco es la política**, ver `politica_base/README.md` (regla de referencia, incluida la coordinación entre varios barcos libres a la vez) y `modelo_rl/README.md` (el agente).

**"Esperar" -- una decisión, dos formas equivalentes de tomarla.** Un barco libre puede recibir explícitamente la orden "esperar" (el índice de acción dedicado, `acciones_posibles[-1] = None`), o puede recibir la orden de ir a su PROPIO nodo actual (`destino_elegido == nodo_origen`) -- las dos son, mecánicamente, la misma cosa: en `env.py`, `step()`, ambas caen en el mismo `continue` (no-op), sin descontar nada de ninguna cola, sin actualizar `nodo_destino`, y por lo tanto **sin ninguna penalización de movimiento** (`recompensa.py` cuenta barcos en movimiento comparando `nodo_origen != nodo_destino` del estado YA actualizado tras el paso -- un barco que se quedó nunca cambia esa comparación). El log de decisiones (`env.log_eventos`, tipo `"decision"`) registra las dos formas por igual como `"esperar"` -- así ninguna métrica que cuente esperas por el string `"esperar"` (`metricas.decisiones_por_barco`) subcuenta el caso de "eligió su propio nodo".

### 4.4 Recompensa -- la fórmula exacta, con ejemplos de magnitud

```
sobrante      = max(0, tiempo_en_el_sistema - 12)             # 12 = tolerancia_incomodidad_min
penalizacion  = min(techo, (sobrante / 18) ** 2)               # 18 = sobrante_normalizador_min, fijo para todos
                                                                 # techo = penalizacion_maxima_persona

r = -[ Σ tamano × penalizacion (por cada persona/grupo activo: esperando O a bordo) + peso_movimiento × (barcos navegando) ]
    + premio_por_persona_entregada × (personas entregadas este paso)
```

Valores de PRODUCCIÓN (`config/instance.yaml` → `recompensa`, los que usan `politica_base/` y sus escalones 1-3): `tolerancia_incomodidad_min=12`, `sobrante_normalizador_min=18`, `penalizacion_maxima_persona=1.0`, `peso_movimiento=0.1`, `premio_por_persona_entregada=0.0`. Los experimentos de RL (`modelo_rl/`) usan **overrides** de esta misma fórmula (`agente.*.recompensa_overrides`) que NUNCA tocan estos valores de producción -- ver `modelo_rl/README.md`.

**¿Por qué existe un techo (`penalizacion_maxima_persona`)?** El simulador no tiene paciencia (sección 4.5) -- nadie se retira nunca, así que alguien puede esperar mucho tiempo si el patrón de demanda no le da prioridad a su par. Sin techo, esa persona sola podría dominar `r` con un número cada vez más grande, aplastando la señal de todos los demás. El techo evita eso: a partir de cierto tiempo de espera la penalización de esa persona ya no sigue creciendo, aunque siga esperando -- se le sigue "cobrando", pero al máximo, de forma estable. (Los experimentos de `modelo_rl/` evalúan explícitamente si mantener este techo activo o no cambia el comportamiento aprendido -- ver `modelo_rl/README.md`.)

**Ejemplos de magnitud** (para una persona con `tamano=1`, con el techo de producción = 1.0):

| Situación | Penalización de esa persona |
|---|---|
| Espera 12 min o menos (dentro de la tolerancia) | 0 |
| Espera 18 min (6 min de sobrante sobre 18) | (6/18)² ≈ 0.11 |
| Espera 24 min (12 de sobrante) | (12/18)² ≈ 0.44 |
| Espera 30 min (18 de sobrante = el normalizador completo) | (18/18)² = 1.0 (el techo) |
| Espera 60 min (48 de sobrante) | sin techo sería (48/18)² ≈ 7.1 -- con techo, sigue en 1.0 |

Un barco moviéndose cuesta `peso_movimiento` (0.1 en producción) -- deliberadamente chico para no desincentivar despachar barcos cuando sí hace falta. La penalización crece al cuadrado del sobrante hasta el techo, así que se queda chica mientras alguien está solo un poco pasado de la tolerancia y se acelera mientras más espera -- la presión se concentra en quien lleva más tiempo esperando, no se reparte parejo entre todos.

### 4.5 Por qué no hay paciencia ni pérdidas (decisión de diseño, no un bug)

La especificación original tenía una **paciencia** por persona (15 min en conexiones fuertes, 30 min en el resto, con ruido ±ε): si nadie la atendía antes de que se le acabara, se retiraba del sistema -- quedaba contada como "perdida", nunca llegaba a subir a un barco.

**Se eliminó por completo** (`unidades.py`, `env.py`, `recompensa.py`, `metricas.py` -- ningún archivo del motor tiene ya un campo o una lógica de paciencia). Ahora **nadie se va nunca**: toda persona/grupo generado espera hasta ser atendido, o hasta que la corrida se corta (queda contabilizado como "esperando al final" o "a bordo al final").

**¿Por qué?** La paciencia introducía un problema de **censura de datos**: alguien que llevaba mucho tiempo esperando y se perdía nunca llegaba a revelar cuánto habría esperado en realidad -- se "borraba" justo antes de que ese dato existiera, sesgando cualquier percentil calculado solo sobre atendidas hacia abajo. Sin paciencia, el tiempo de espera de TODOS los atendidos es un dato real y sin censurar, y los percentiles de `metricas.metricas_por_usuario` (p50/p90/p95/máx) se pueden usar directamente para definir una garantía de tiempo de servicio con sustento en datos, no en un supuesto.

El costo de este cambio: ya no hay una señal explícita de "esto no se puede seguir postergando" en la recompensa -- esa presión depende enteramente del término de incomodidad con techo (sección 4.4).

---

## 5. Métricas y visualización (funciones genéricas, no saben de políticas)

`src/metricas.py` -- punto de entrada único `reporte_completo(env)`, junta: `verificar_conservacion` (toda persona generada debe terminar contabilizada), `metricas_globales`, `metricas_por_par`, `metricas_por_barco` (movimientos, tiempo navegado vs. esperando -- ver siguiente párrafo --, ocupación), `metricas_por_usuario` (percentiles de espera/viaje/sistema), `sin_atender_al_final_por_par` (backlog, no "pérdidas": el simulador nunca pierde a nadie), `desglose_recompensa`. `combinar_corridas(envs)` junta N corridas independientes (días de una semana, semillas de evaluación) en un solo objeto con la misma interfaz, para poder llamar `reporte_completo` sobre el conjunto sin duplicar ninguna lógica.

**Esperando vs. navegando -- dos vistas independientes, deben coincidir aproximadamente.** `metricas_por_barco` reporta `tiempo_esperando_min`/`pct_esperando` contando, para cada paso de `env.historial_estados`, si el barco estaba libre en ese paso (medido en MINUTOS). `decisiones_por_barco` cuenta cuántas veces el log de decisiones (`env.log_eventos`, tipo `"decision"`) registró `"esperar"` para ese barco (medido en DECISIONES, ver sección 4.3 para por qué ambas formas de "quedarse" cuentan igual ahí). Un barco está "esperando" en un paso casi siempre porque decidió `"esperar"` el paso anterior -- salvo dos casos sin una decisión "esperar" propia detrás: el primer paso de la corrida (el barco arranca libre por construcción, antes de que nadie decida nada) y el paso exacto en que un barco recién llegado queda libre (se le pregunta qué hacer recién en el paso SIGUIENTE). Por eso `tiempo_esperando_min / paso_tiempo_min` es un poco MAYOR que `decisiones_por_barco()["veces_espero"]` -- la diferencia esperada es chica y explicable (uno por el arranque, uno por cada llegada de ese barco), y sirve como verificación cruzada: una diferencia mucho más grande señalaría un log de decisiones incompleto.

`src/visualizacion.py` -- interactivas (Plotly, no matplotlib): `graficar_perfil_espera` (personas esperando en el tiempo), `graficar_ocupacion_flota` (ocupación de cada barco en el tiempo, leyenda click-para-aislar), `graficar_heatmap_cumplimiento` (% atendidas por par O-D), `graficar_desglose_recompensa` (incomodidad/movimiento/entrega por paso), `graficar_sin_atender_al_final` (backlog por par). Todas reciben cualquier objeto con la interfaz de un `env` (uno solo, o una `CorridaCombinada`) -- mismas 5 funciones, usadas igual por `politica_base/`, `modelo_rl/` y `comparacion/`. Más geoespacial: `animar_corrida` (GIF), `inspeccionar` (un minuto concreto, funciona siempre), `reproductor_interactivo` (botones de paso a paso, solo con un kernel de Jupyter vivo).

**Dónde viven los logs, mientras dura la corrida** (atributos de `env`, en memoria, se llenan solos dentro de `env.step()`): `env.log_eventos` (sube/baja/decision/movimiento), `env.log_recompensa` (desglose por paso), `env.historial_estados` (una foto completa por paso), `env.atendidas_historico` (personas ya resueltas). Nada se guarda a disco automáticamente -- cada notebook exporta lo que necesita a su propia carpeta `output/`.

---

## 6. Supuestos y limitaciones (del motor)

- **El entorno en sí no tiene aleatoriedad propia.** Toda la aleatoriedad del pipeline vive en la generación de demanda (`demand/`, con su semilla, o `modelo_rl/src/entrenamiento.py` para RL). Una vez que la tabla de grupos está fija, el simulador es 100% determinista -- propiedad deseable para verificar, no un defecto.
- **Sin paciencia, sin pérdidas -- decisión de diseño explícita** (sección 4.5). Habilita percentiles de espera reales, sin censurar, pero también significa que la recompensa ya no distingue conexión fuerte de conexión normal (el normalizador es fijo para todos), y que la presión contra dejar a alguien esperando mucho depende enteramente del término de incomodidad con techo.
- **Tiempo de espera reportado** = tiempo hasta que el destino de la persona quedó resuelto (subió a un barco) -- no el tiempo de viaje a bordo después de subir. El tiempo en sistema sí suma ambos. Solo se calcula sobre atendidas.
- **Los percentiles de espera (`metricas_por_usuario`) son la base para definir garantías de tiempo de servicio con datos reales**, no supuestos -- ese es justamente el motivo de la sección 4.5.
