# Simulador de despacho de barcos (entorno Gymnasium + primer agente PPO)

**Qué es esto, en una frase:** un programa que modela minuto a minuto (en pasos de 2 min) qué hacen los barcos y los pasajeros en Bergen. Primero se verificó a fondo con una regla fija ("nearest-available", secciones 1-10) — franja mañana, día completo, y una semana — y ahora se conectó un primer agente de RL (PPO, secciones 11-13) para compararlo contra esa línea base; la sección 14 documenta un ciclo de diagnóstico (gráficas interactivas + un agente que aprendía mal, arreglado) que mejoró ambas partes.

**Documento fuente (manda sobre cualquier duda):** `docs/especificacion_simulador_rl.md`. Todo lo de abajo está mapeado a sus secciones (§1-§12).

---

## 1. Cómo está organizado

```
simulacion/
├── README.md                 # este archivo
├── config/instance.yaml      # parámetros PROPIOS de este paso (ver sección 3)
├── src/
│   ├── unidades.py           # qué es una "Unidad" (persona o grupo) y cómo se generan
│   ├── estado.py             # la foto del mundo en cada instante (§4 de la especificación)
│   ├── recompensa.py         # la fórmula de puntaje (§7)
│   ├── politica_base.py      # la regla fija "nearest-available" + coordinación de flota (§9)
│   ├── env.py                # el simulador en sí (junta todo lo anterior, §3/§5/§6)
│   ├── entrenamiento.py      # EntornoDemandaAleatoria -- demanda fresca por reset, para RL (sección 11)
│   ├── metricas.py           # verificación de conservación + métricas detalladas + combinar_corridas
│   └── visualizacion.py      # mapa animado, gráficas, e inspector de un minuto concreto
├── notebooks/
│   ├── 00_preparar_demanda_escalones.ipynb   # genera la demanda de prueba (escalones 1-3)
│   ├── 01_escalon_1_verificacion.ipynb        # franja mañana, verificación a ojo + animación
│   ├── 02_escalon_2_metricas.ipynb            # día completo, métricas agregadas (sin animación)
│   ├── 03_escalon_3_semana.ipynb              # semana completa (7 días), métricas agregadas
│   ├── 04_entrenamiento_rl.ipynb              # entrena PPO sobre la instancia chica (sección 12)
│   └── 05_comparacion_agente_vs_base.ipynb    # agente PPO vs. política base (sección 13)
└── output/
    ├── escalon1/              # todo lo que produce el notebook 01
    ├── escalon2/              # todo lo que produce el notebook 02, misma estructura de archivos
    ├── escalon3/               # todo lo que produce el notebook 03 (semana)
    ├── rl_ppo/                 # modelo entrenado, monitor.csv, curva de entrenamiento (notebook 04)
    └── comparacion/             # tablas y gráficas de agente vs. base (notebook 05)
```

Cada escalón tiene su propia carpeta en `output/` con los mismos nombres de archivo adentro (`grupos.csv`, `metricas_por_par.csv`, `heatmap_cumplimiento.png`, etc.) — así no hay que andar leyendo prefijos largos, y es fácil comparar un escalón contra el otro abriendo la carpeta correspondiente. La única diferencia de contenido: `escalon1/` tiene `animacion.gif` e `inspeccion_minuto400.png` (la corrida es chica, 90 pasos, se puede animar); `escalon2/` y `escalon3/` no (corridas largas — un GIF de esa duración pesa y tarda mucho más, y la verificación visual paso a paso ya se hizo a fondo en el escalón 1, no hacía falta repetirla). `escalon3/` usa `grupos_semana.csv` en vez de `grupos.csv` (7 días concatenados, columna `dia`).

**Cómo leer esto si nunca programaste en Python:** cada archivo de `src/` es una "pieza" con una responsabilidad. Los notebooks son los que las juntan y las corren, mostrando resultados. Nada se ejecuta solo con crear los archivos — hay que abrir un notebook y correrlo (ya está corrido y guardado, puedes abrirlo y ver los resultados sin volver a correrlo).

**Qué es esa primera línea `from __future__ import annotations` que aparece en casi todos los archivos:** es una instrucción del lenguaje Python, no algo específico de este proyecto. Sin ella, cuando escribimos una "anotación de tipo" como `nodos: list[str]` o `politica: str | None` (para decir "esto debería ser una lista de strings", "esto puede ser un string o nada"), Python las evalúa de inmediato al cargar el archivo, y en algunas situaciones eso falla (por ejemplo, si una clase se refiere a sí misma en su propia anotación, antes de terminar de definirse). Esa línea le dice a Python "trata las anotaciones de tipo como texto, no las evalúes ahora" — evita ese problema y es una práctica estándar recomendada en Python moderno, no algo que inventamos para este proyecto.

---

## 2. Qué es Gymnasium (y qué NO es)

**Gymnasium no trae simuladores prearmados para elegir.** Es una librería que define un **molde estándar**: cualquier "entorno" (environment) debe ser una clase de Python con exactamente dos métodos:

- `reset()` → arranca el mundo desde cero, devuelve el estado inicial.
- `step(accion)` → recibe una acción, avanza el mundo un paso, devuelve `(nuevo_estado, recompensa, terminó, se_truncó, info_extra)`.

Ese molde es TODO lo que Gymnasium impone. Nosotros escribimos el 100% de la lógica real (qué hace un barco, cómo suben y bajan los pasajeros) en `src/env.py`, clase `SimuladorBarcosBergen`. Gymnasium no sabe nada de barcos ni de Bergen — solo garantiza que nuestra clase "hable el mismo idioma" que espera cualquier librería de RL después (Stable-Baselines3, mencionada en la especificación §11, ya conectada -- ver secciones 11-13). Es como un enchufe: Gymnasium estandariza la forma del enchufe, nosotros construimos el aparato.

**El "tipo" de simulación** (una pregunta distinta, sobre la mecánica interna, no sobre Gymnasium): es una **simulación de tiempo discreto en pasos fijos de 2 minutos** — cada `step()` siempre avanza exactamente 2 minutos, nunca menos ni más (a diferencia de una simulación "por eventos", que salta directo al próximo momento interesante). La especificación (§3) pide explícitamente pasos fijos, por simplicidad y porque encaja natural con el `step()` de Gymnasium.

**¿Sí estamos usando Gymnasium de verdad, o solo el nombre?** Sí, de verdad: `src/env.py`, línea 27, `class SimuladorBarcosBergen(gym.Env):` — hereda literalmente de la clase base de la librería (`import gymnasium as gym`). Eso obliga (y verifica en tiempo de ejecución) a que la clase tenga `action_space` y `observation_space` bien declarados (líneas 90-92: `MultiDiscrete` para las acciones, `Box` para el vector aplanado) y los métodos `reset()`/`step()` con la firma exacta que la librería espera. No es una simulación casera a la que le pusimos "Gymnasium" de nombre — si `SimuladorBarcosBergen` no cumpliera el contrato, Stable-Baselines3 (secciones 11-13) no podría usarla; de hecho `check_env` (sección 12) lo confirma en tiempo de ejecución.

**¿Qué semilla estamos usando?** `simulacion/config/instance.yaml` → `semilla: 42`. Aclaración importante: como se explica en la sección de reproducibilidad más abajo, esa semilla se usa para regenerar la DEMANDA (`demand/src/llegadas.py`), no dentro de `env.py` — el entorno en sí no tiene ningún sorteo propio, así que `env.reset(seed=...)` no cambia nada por sí solo; lo que realmente reproduce una corrida es generar la tabla de grupos con la semilla 42 (paso 00) y correr el entorno sobre esa misma tabla.

---

## 3. De dónde sale cada parámetro (nada se inventa ni se duplica)

Regla del proyecto (ya usada en `demand/`): cada carpeta es dueña de sus propios parámetros; las demás los **leen** de la fuente, nunca los copian.

| Parámetro | Vive en | Por qué ahí |
|---|---|---|
| Capacidad del barco (20), nodo inicial (Bryggen), conexiones fuertes | `bergen-boats/config/instance.yaml` → `flota`, `garantia` | Son propiedades de la flota/física, ya definidas en el paso de ruteo |
| Tiempos de viaje entre nodos | `bergen-boats/02_ruteo_navegable/output/matriz_tiempos_min.csv` | Ya calculados (Dijkstra sobre agua real), nunca se recalculan aquí |
| Geometría de las rutas (para animar) | `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson` | Igual — ya calculada, se persistió en este mismo trabajo para poder reusarla |
| Patrón de quién viaja a dónde y cuándo | `demand/src/llegadas.py` + `demand/output/matriz_intensidad_od.csv` | El generador de demanda real (gravedad + SSB + Poisson), no se toca |
| **Tolerancia (12 min), normalizador fijo (18 min), techo por persona (1.0), peso de movimiento (0.1)** | `simulacion/config/instance.yaml` → `recompensa` | Son propios de ESTE paso (la fórmula de recompensa, ver sección 4.4) |
| **Tamaño de flota y % de demanda por escalón** | `simulacion/config/instance.yaml` → `escalones` | Propios de la verificación (no existen en `demand/` ni en `bergen-boats/`) |
| **Paso de tiempo (2 min), semilla, unidad de demanda (personas/grupos)** | `simulacion/config/instance.yaml` | Propios de este paso |

Si buscas un número y no está en `simulacion/config/instance.yaml`, casi seguro es porque pertenece a otro paso y aquí solo se **lee** — revisa la tabla de arriba.

---

## 4. El MDP explicado con ejemplos reales

### 4.1 Estado (§4) — dos versiones, una para humanos y una para la red

Cada instante del mundo se guarda en un objeto `EstadoSimulacion` (`src/estado.py`). Tiene DOS formas de mostrarse:

- **`.to_dict()`** — legible, para nosotros. Ejemplo real de una corrida (recortado):
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
- **Vector aplanado (`aplanar_estado`)** — una lista de ~50 números (para 2 barcos), lo único que vería un agente de RL. Nadie lo lee a mano; existe porque una red neuronal necesita números de tamaño fijo, no un diccionario. `src/estado.py` explica cada número con comentarios.

### 4.2 Unidad de demanda: personas, no grupos (ajuste que pediste)

`demand/` genera **grupos** (ej. "3 personas que salen juntas de Kleppestø a las 6:23"). Este paso, por defecto, **explota** cada grupo en personas individuales (`src/unidades.py`, función `grupos_a_unidades`) — cada persona hereda el mismo origen/destino/hora de llegada de su grupo, pero se trata como una solicitud independiente. Es un interruptor en el config (`unidad_demanda: "personas" | "grupos"`), no dos simuladores distintos.

`demand/` sigue calculando una paciencia por grupo (columna `espera_maxima_min`, 15/30 min ± jitter) porque es un dato propio de ese módulo — pero `simulacion/` ya no la lee en ningún lado (ver sección 4.5): la `Unidad` de este paso no tiene ese campo.

### 4.3 Acciones (§5) — viajes directos punto a punto

Cada barco libre recibe una de 5 órdenes: ir a Kleppestø / Laksevåg / Bryggen / Sandviken, o esperar. Un barco en ruta ignora cualquier orden y sigue hasta llegar (no se redirige a media ruta, tal como pide la especificación).

**La política NO elige a quién recoge, elige el NODO destino.** Quién sube es una regla fija del simulador — pero, a diferencia de una primera versión, **las colas están separadas por par origen-destino** (12 colas, una por cada combinación, no 4 colas por nodo). Cuando un barco libre en A recibe la orden "ir a B", solo puede embarcar de la cola exacta A→B, en orden de llegada, hasta llenar el barco — nunca lleva gente con destinos mixtos, y al llegar a B baja a todos. Es un viaje directo punto a punto, no una ruta con paradas intermedias (ese modelo no existe todavía en este proyecto).

**¿Y si la gente que más necesita un barco está en OTRO nodo?** Aquí hay una regla que hay que entender bien porque no es la única forma razonable de hacerlo, y es importante para explicarla:

> **La política SIEMPRE prioriza la demanda del nodo donde el barco ya está, sobre cualquier demanda de otro nodo — sin importar cuánto tiempo lleve esperando la gente de otros nodos.**

Ejemplo concreto (el que preguntaste): un barco queda libre en Bryggen. Ahí mismo hay 5 personas que acaban de llegar (esperan 1 minuto) queriendo ir a Laksevåg. En Kleppestø hay 1 persona que lleva 20 minutos esperando ir a Sandviken. **La política manda el barco a buscar a las 5 de Laksevåg, no a la persona de Kleppestø** — aunque esa persona lleve muchísimo más tiempo esperando y esté mucho más cerca de perderse. La regla no compara "qué tan urgente es cada uno" de forma global; primero agota TODA la demanda local (por más pequeña o reciente que sea) y solo mira otros nodos cuando en el propio no queda absolutamente nadie esperando.

¿Por qué se diseñó así? Para no dejar "abandonada" gente que el barco ya podría atender de inmediato, a cambio de perseguir a alguien que todavía requiere viajar. Es una decisión de diseño razonable pero **no es la única posible** — una alternativa sería comparar la urgencia de TODOS los candidatos (locales y remotos) en una sola lista, y que el barco a veces se vaya a buscar a alguien lejano si es mucho más urgente que la demanda local. Esa alternativa no está implementada todavía; la actual es más simple de explicar y de verificar, pero puede llevar a que alguien muy urgente en otro nodo espere mucho más tiempo de lo razonable mientras el barco atiende demanda local menos urgente, como en el ejemplo de arriba. Como el simulador ya no purga a nadie por paciencia (sección 4.5), esa espera ya no tiene un límite implícito — puede crecer indefinidamente si el patrón de demanda no le da nunca prioridad a ese par. Queda declarado como límite conocido, no escondido (ver sección 9).

Mecánicamente, en `src/politica_base.py`, función `politica_base`: primero arma la lista `candidatos_directos` (solo pares que SALEN del nodo actual del barco, `o == A`); si esa lista no está vacía, elige de ahí el más urgente y punto — nunca mira `candidatos_reposicion` (demanda de otros nodos) a menos que `candidatos_directos` esté completamente vacío. Cuando sí se reposiciona, el barco viaja **vacío** (no recoge a nadie de camino) hasta el nodo con la demanda remota más urgente y ahí, en su siguiente momento libre, decide de nuevo con información fresca — es una decisión miope (no planea las dos etapas de una vez), consistente con que "nearest-available" es una regla simple, no un optimizador.

### 4.4 Recompensa (§7, SIMPLIFICADA) — la fórmula exacta, con ejemplos de magnitud

**Cambio sobre la especificación original:** ya no hay término de pérdida (nadie se pierde, sección 4.5), y `sobrante_max` es una sola constante fija para todos, no algo calculado por persona. Quedan solo dos términos:

```
sobrante      = max(0, tiempo_en_el_sistema - 12)             # 12 = tolerancia_incomodidad_min
penalizacion  = min(1.0, (sobrante / 18) ** 2)                 # 18 = sobrante_normalizador_min, fijo para todos
                                                                 # 1.0 = penalizacion_maxima_persona (techo)

r = -[ Σ tamano × penalizacion (por cada persona/grupo activo: esperando O a bordo) + 0.1 × (barcos navegando) ]
```

**¿Por qué no hay término de pérdida?** Directo: ya no existe el evento "pérdida" en ningún lado del simulador (sección 4.5) — nadie se retira nunca, así que no hay nada que penalizar en ese instante. La presión por no dejar a alguien esperando demasiado sigue estando, pero es enteramente la del término de incomodidad, que ahora puede crecer sin el límite implícito que antes le ponía la pérdida.

**¿Por qué `sobrante_max` es una constante fija (18) y no algo por persona?** Antes de quitar la paciencia del todo, hubo una versión intermedia donde `sobrante_max` se calculaba como `paciencia - 12` para cada quien — pero la paciencia en conexiones fuertes era 15 min con ruido (`jitter`) de hasta ±5, así que podía bajar a 10 min, por debajo de la tolerancia de 12, dando un `sobrante_max` negativo (sin sentido). Ese problema quedó completamente resuelto al eliminar la paciencia: 18 es simplemente el ancho de curva que se usa para TODOS, sin relación con ningún dato por persona (ya no existe ese dato). El costo es el mismo de antes: no se distingue entre alguien de una conexión más difícil de cubrir y alguien de una conexión fácil al calcular qué tan "grave" es su sobrante — todos se miden con la misma vara.

**¿Por qué existe un techo (`penalizacion_maxima_persona = 1.0`)?** Es nuevo desde que se quitó la paciencia. Antes, nadie podía esperar indefinidamente (se perdía tarde o temprano), así que `sobrante` tenía un límite natural. Ahora alguien puede quedar esperando mucho tiempo si el patrón de demanda no le da prioridad a su par (sección 4.3) — sin techo, esa persona sola podría dominar `r` con un número cada vez más grande, aplastando la señal de todos los demás. El techo evita eso: a partir de 30 min de espera (18 min de sobrante = el normalizador completo) la penalización de esa persona ya no sigue creciendo, aunque siga esperando. No se le deja de "cobrar" — se le cobra el máximo, de forma estable.

**Ejemplos de magnitud** (para tener intuición de qué tan grande es cada número, para una persona con `tamano=1`):

| Situación | Penalización de esa persona |
|---|---|
| Espera 12 min o menos (dentro de la tolerancia) | 0 |
| Espera 18 min (6 min de sobrante sobre 18) | (6/18)² ≈ 0.11 |
| Espera 24 min (12 de sobrante) | (12/18)² ≈ 0.44 |
| Espera 30 min (18 de sobrante = el normalizador completo) | (18/18)² = 1.0 (el techo) |
| Espera 60 min (48 de sobrante) | sin techo sería (48/18)² ≈ 7.1 — con techo, sigue en 1.0 |

Diez personas esperando ~30 min o más al mismo tiempo aportan como máximo `-10` a `r` en ese paso (nunca más, gracias al techo). Un barco moviéndose cuesta solo `0.1` — deliberadamente chico para no desincentivar despachar barcos cuando sí hace falta. La penalización crece al cuadrado del sobrante hasta el techo, así que se queda chica mientras alguien está solo un poco pasado de la tolerancia y se acelera mientras más espera — la idea es que la presión se concentre en quien lleva más tiempo esperando, no que se reparta parejo entre todos.

### 4.5 Por qué se quitó la paciencia (cambio de diseño, no un bug)

La especificación original (y las primeras versiones de este simulador) tenían una **paciencia** por persona (15 min en conexiones fuertes, 30 min en el resto, con ruido ±ε): si nadie la atendía antes de que se le acabara, se retiraba del sistema — quedaba contada como "perdida", nunca llegaba a subir a un barco.

**Se eliminó por completo** (`unidades.py`, `env.py`, `recompensa.py`, `politica_base.py`, `metricas.py` — ningún archivo del simulador tiene ya un campo o una lógica de paciencia). Ahora **nadie se va nunca**: toda persona/grupo generado espera hasta ser atendido, o hasta que la corrida se corta (queda contabilizado como "esperando al final" o "a bordo al final" — sección 8).

**¿Por qué?** La paciencia introducía un problema de **censura de datos**: alguien que llevaba mucho tiempo esperando y se perdía nunca llegaba a revelar cuánto habría esperado en realidad — se "borraba" justo antes de que ese dato existiera. Eso hacía imposible calcular, con fundamento, cuánto hay que esperar en el peor caso razonable (percentiles altos de espera) — la pérdida escondía sistemáticamente a los casos más lentos, sesgando cualquier percentil calculado solo sobre atendidas hacia abajo. Sin paciencia, el tiempo de espera de TODOS los atendidos es un dato real y sin censurar, y los percentiles de `metricas.metricas_por_usuario` (p50/p90/p95/máx) se pueden usar directamente para definir una garantía de tiempo de servicio con sustento en datos, no en un supuesto.

El costo de este cambio: ya no hay una señal explícita de "esto no se puede seguir postergando" en la recompensa (antes, perder a alguien era, indirectamente, el peor desenlace posible). Ahora esa presión depende enteramente del término de incomodidad con techo (sección 4.4) — se documenta como pregunta abierta para cuando se entrene el agente (sección 9).

---

## 5. La política de referencia ("nearest-available")

`src/politica_base.py` tiene dos funciones:

- **`politica_base(barco, estado, matriz_tiempos, cfg)`** — decide UN barco: prioriza la demanda que sale de su nodo actual (servible de inmediato); si no hay nada ahí, considera reposicionarse vacío hacia el nodo con la demanda más urgente del resto del sistema (sección 4.3).
- **`asignar_flota(barcos_libres, estado, matriz_tiempos, capacidad_barco, cfg)`** — aplica lo anterior a **varios barcos libres a la vez, uno por uno**, descontando localmente lo que cada barco ya "se llevaría" antes de decidir el siguiente. Es la forma correcta de usar la política con más de un barco (ver bug #3 abajo) — la que realmente se usa en los notebooks.

Es la "línea base" tonta contra la que se comparará el agente después — está escrita como funciones independientes para poder cambiarlas por el agente sin tocar el simulador.

---

## 6. Problemas encontrados y corregidos durante la verificación

Verificar el simulador a ojo antes de conectar el agente sirvió exactamente para esto — se encontraron 3 problemas reales, ninguno escondido:

1. **(Superado por el rediseño de colas, sección 4.3) La política ignoraba a los pasajeros ya a bordo.** En una primera versión, un barco podía llevar gente con destinos mixtos, y la política solo miraba las colas en tierra para decidir a dónde ir — un barco recién cargado podía ser mandado a buscar a alguien más, abandonando a los que ya llevaba. Se resolvió de raíz al pasar a colas por par origen-destino y viajes directos punto a punto: ahora un barco libre siempre está vacío por construcción, así que el problema no puede volver a ocurrir (no hace falta parchear la política para que "recuerde" a los de a bordo).
2. **(Superado por la eliminación total de la paciencia, sección 4.5) La fórmula de `sobrante_max` podía ser negativa.** La paciencia en conexiones fuertes era 15 min, pero con el ruido (`jitter`) que ya tenía `demand/` podía bajar a 10 min — por debajo de los 12 min de tolerancia de la fórmula. `sobrante_max = paciencia - 12` daba un número negativo (división sin sentido). Primero se corrigió con un piso (`sobrante_max_minimo`); después, al simplificar la recompensa a una constante fija, el caso límite desapareció; y finalmente, al quitar la paciencia del todo (ya no existe ese campo en ningún lado), el problema quedó completamente sin base — no hay ninguna paciencia de la que `sobrante_max` pudiera derivarse mal.
3. **Doble despacho: dos barcos libres a la vez elegían el mismo destino.** Con la política aplicada de forma independiente por barco (cada uno mirando la misma foto del mundo, antes de que cualquiera de los dos suba gente), dos barcos libres en el mismo nodo y momento podían decidir ambos "ir a Bryggen" pensando que había 11 personas esperando ahí — el primero en procesarse se las llevaba todas, el segundo viajaba **vacío** hasta Bryggen, pagando la penalización de movimiento sin servir a nadie. Se detectó inspeccionando el minuto 400 de la corrida de ejemplo (`visualizacion.inspeccionar`) y se confirmó en el log de texto. Se corrigió con `asignar_flota` (sección 5): asigna los barcos libres de a uno, con una copia local de las colas que se va descontando. Efecto medido en su momento (bajo el sistema de paciencia, antes de eliminarla del todo — sección 4.5): **el % de cumplimiento subió de 53% a 90%** — no era un detalle menor. (Las cifras actuales de la sección 8 ya no son comparables directamente contra ese 90%, porque hoy "% atendidas" mide algo distinto: nadie se pierde, así que sin este fix la diferencia se habría visto en el backlog al final y en las esperas, no en un % de pérdidas.)

---

## 7. Los escalones (instancias de prueba)

| | Escalón 1 | Escalón 2 | Escalón 3 |
|---|---|---|---|
| Cuándo | Franja mañana (6-9h) | Día completo (6-24h) | Semana completa (7 días, 6-24h c/u) |
| Barcos | 2 | 3 | 3 |
| Grupos / personas | 23 / 202 | 133 / 1106 | 620 / 5429 |
| Pasos de 2 min | 90 | 540 | 7 × hasta 540 (7 episodios independientes) |
| Para qué | Verificar a ojo, log de texto, animación | Métricas agregadas, sin animación (ver sección 8.1) | Métricas agregadas sobre entre-semana + fin de semana (ver sección 8.4) |
| Notebook | `01_escalon_1_verificacion.ipynb` | `02_escalon_2_metricas.ipynb` | `03_escalon_3_semana.ipynb` |

La demanda de cada escalón se generó con `demand/src/llegadas.py` **sin tocarlo** — solo se le pasó un `porcentaje_poblacion_dia` más bajo que el oficial (10%), calibrado por prueba y error hasta acercarse a "~20-30 grupos" (escalón 1) que pedía la especificación. `demand/config/instance.yaml` sigue intacto en 10%. El escalón 3 reusa la densidad y flota del escalón 2 (`porcentaje_poblacion_dia=0.012`, 3 barcos) sobre `generar_llegadas_semana` (`demand/src/llegadas.py`) en vez de `generar_llegadas_dia` -- 7 días (lunes=0..domingo=6), con el factor entre-semana/fin-de-semana (`factor_dia_semana`) ya resuelto adentro.

---

## 8. Corridas de ejemplo y métricas

Ver `notebooks/01_escalon_1_verificacion.ipynb` (log paso a paso) y `notebooks/02_escalon_2_metricas.ipynb` (solo métricas agregadas, ver sección 7) para el detalle completo.

**Verificación de conservación** (`metricas.verificar_conservacion`): toda persona generada debe terminar contabilizada. Confirmado en los dos escalones — y ahora, sin paciencia (sección 4.5), **el 100% de la gente generada termina atendida** en ambas corridas, porque nadie se retira nunca y las dos ventanas de tiempo alcanzan para que la flota agote todo el backlog antes de terminar:

| | Escalón 1 | Escalón 2 |
|---|---|---|
| Generadas | 202 | 1106 |
| = Atendidas + | 202 | 1106 |
| Esperando al final + | 0 | 0 |
| A bordo al final | 0 | 0 |

**Globales** (`metricas.metricas_globales`):

| Métrica | Escalón 1 | Escalón 2 |
|---|---|---|
| % atendidas | 100.0% | 100.0% |
| Espera media | 17.4 min | 10.9 min |
| Tiempo en sistema medio / máximo | 27.2 / 54.0 min | 20.5 / 53.6 min |
| Espera p50 / p90 / p95 | 15.6 / 33.4 / 33.5 min | 9.4 / 21.6 / 26.3 min |

**Comparado con la versión anterior (con paciencia):** antes de este cambio, el escalón 1 reportaba 89.6% de "cumplimiento" (181/202 atendidas, 21 "perdidas") y una espera media de 10.7 min; el escalón 2, 82.5% (912/1106) y 8.6 min. Esos números eran más bajos y las esperas medias más cortas **porque estaban censurados**: a la gente que esperaba demasiado se la retiraba antes de que su espera larga contara para el promedio — el promedio solo veía a los que tuvieron suerte de ser atendidos rápido. Ahora que nadie se retira, el 100% de la gente efectivamente es atendida (dado que ambas corridas tienen ventana suficiente para vaciar el backlog) y la espera media sube porque ahora sí incluye a quienes esperaron mucho — son los mismos episodios de espera larga que antes se "perdían" del cálculo, no un simulador que de repente funciona peor. Los percentiles altos (p90/p95) son justamente el dato nuevo que esto habilita: antes no existían con fundamento, porque los casos más lentos nunca llegaban a medirse completos.

El escalón 2, pese a tener 5.5× más demanda y 9× más pasos que el escalón 1 (con solo 1 barco más), termina con espera media y p90/p95 **más bajos** — tiene sentido: el escalón 1 concentra toda su demanda en 3h de franja pico, sin margen para que el backlog se drene entre picos, mientras que el escalón 2 cubre 18h con franjas de alta y baja intensidad alternándose, dando más oportunidades de que la flota se ponga al día en los valles.

**Por par origen-destino, por barco, por usuario (percentiles), y backlog al final de la corrida** — tablas completas en cada notebook (`metricas.metricas_por_par`, `metricas_por_barco`, `metricas_por_usuario`, `metricas.sin_atender_al_final_por_par`), guardadas también en `output/escalon1/metricas_por_par.csv` / `metricas_por_barco.csv` y su equivalente en `output/escalon2/`. En las dos corridas, `sin_atender_al_final_por_par` da una tabla vacía (nadie quedó sin atender) — consistente con el 100% de arriba.

**Gráficas** (`src/visualizacion.py`, interactivas -- Plotly, no matplotlib; `output/escalon1/*.html` y `output/escalon2/*.html`, además de `output/escalon3/`): perfil temporal de personas esperando (`wait_profile.html`), ocupación de la flota en el tiempo (`fleet_occupancy.html`), heatmap de % atendidas por par (`pct_served_heatmap.html`), desglose de recompensa en el tiempo (`reward_breakdown.html`), barras de backlog por par (`backlog_by_pair.html`) — mismas gráficas, misma función, para los tres escalones. Zoom/pan nativo, valores exactos al pasar el mouse, y la leyenda de `fleet_occupancy.html` permite aislar un barco (click) o volver a mostrarlos todos (doble click) -- ver sección 14 para por qué se rehicieron en Plotly.

Reproducibilidad verificada en los dos escalones: misma semilla de demanda → misma corrida exacta (recompensa total, atendidas y sin-atender-al-final idénticas); semilla distinta → resultado distinto.

---

## 8.1 Visualización

`src/visualizacion.py` + celdas finales de `notebooks/01_escalon_1_verificacion.ipynb` → `output/escalon1/animacion.gif` (90 frames, uno por paso de 2 min). Reusa el mapa base y `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson` (persistidas en este mismo trabajo) para que los barcos se muevan bordeando tierra, no en línea recta. Colas visibles como contador junto a cada nodo, barcos como triángulos (gris = vacío, rojo = con gente, número = ocupación).

**El escalón 2 NO tiene animación ni reproductor interactivo, a propósito.** Con 540 pasos (6x los del escalón 1), un GIF de esa duración pesa y tarda mucho más de generar, y la verificación visual paso a paso ya se hizo a fondo en el escalón 1 (ahí fue donde se encontró el bug #3, sección 6) — no había necesidad de repetirla a esta escala solo para tener una animación más larga. Si más adelante hace falta ver la película completa del día, `visualizacion.animar_corrida` funciona igual sobre `env.historial_estados` del escalón 2, solo hay que agregar la celda.

**Nota técnica:** el mapa de `bergen-boats/02_ruteo_navegable` usa CartoDB.Positron, pero al probarlo aquí devolvió una marca de agua "API KEY REQUIRED" (CartoDB cambió su política de acceso gratuito desde que se construyó ese paso). Se probó también OpenStreetMap, que bloqueó la solicitud por política de uso. Se usa **Esri.WorldGrayCanvas** en su lugar (sin llave, respondió limpio) — mismo principio (mapa real de fondo), proveedor distinto. El fondo se descarga **una sola vez** por corrida (no una vez por frame) para no depender de 90 llamadas de red seguidas.

## 8.2 Inspector de un minuto concreto, y reproductor tipo video

`visualizacion.inspeccionar(minuto, env, ...)` pausa en el paso más cercano al minuto pedido y muestra: posición/ocupación de cada barco, las 12 colas con personas esperando, qué decidió cada barco ese paso, y la recompensa con su desglose — en texto y dibujado sobre el mapa. Funciona siempre, en cualquier contexto (no requiere Jupyter con kernel vivo). Fue precisamente con esta función (parado en el minuto 400) que se encontró el bug #3 de la sección 6.

`visualizacion.reproductor_interactivo(env, ...)` es el reproductor completo: botón de reproducir/pausar (`ipywidgets.Play`), barra de tiempo para saltar a cualquier paso hacia adelante o atrás, el mapa animado, y un panel de texto al lado con el estado organizado de ese paso (barcos, colas, decisión de cada barco, recompensa). **Solo funciona con un kernel de Jupyter vivo** — hay que abrir `notebooks/01_escalon_1_verificacion.ipynb` en VS Code o Jupyter Lab y correr las celdas ahí mismo (con "Run All" o celda por celda); un notebook ya ejecutado y guardado, como el que queda en el repositorio después de `nbconvert`, no tiene kernel corriendo, así que los botones no responden ahí ni se puede mostrar funcionando en una captura de pantalla.

## 8.3 Dónde viven los logs (lo que alimenta las métricas)

Todo lo que usa `metricas.py` y `visualizacion.py` vive, mientras dura la corrida, como atributos del objeto `env` (en memoria, no en disco) -- se llenan solos, paso a paso, dentro de `env.step()`:

| Atributo | Qué guarda | Una fila por |
|---|---|---|
| `env.log_eventos` | Lista de diccionarios: tipo (`sube`/`baja`/`decision`/`movimiento`), minuto, y datos según el tipo | Cada evento discreto |
| `env.log_recompensa` | Desglose de la recompensa (incomodidad, movimiento, total) | Cada paso de 2 min |
| `env.historial_estados` | Una foto completa (`estado.to_dict()`) | Cada paso de 2 min |
| `env.atendidas_historico` | Objetos `Unidad` ya resueltos (no un log de eventos, la lista completa de personas atendidas) | Cada persona atendida |

No hay ningún log de "perdidas" — no existe ese evento (sección 4.5). Quien no llegó a ser atendido cuando termina la corrida simplemente sigue en `env.colas` (esperando) o en `barco.a_bordo` (a bordo, sin llegar todavía) — de ahí lee `metricas.sin_atender_al_final_por_par` el backlog al final, en vez de un log dedicado.

Nada de esto se guarda a disco automáticamente con solo correr la simulación -- si cierras Python, se pierde. Cada notebook (01 y 02) sí exporta a su carpeta (`output/escalon1/` o `output/escalon2/`): los CSV crudos (`log_eventos.csv`, `log_recompensa.csv`) y las tablas de métricas ya calculadas (`metricas_por_par.csv`, `metricas_por_barco.csv`), para poder auditar los números sin volver a correr nada.

## 8.4 Escalón 3 (semana completa)

`notebooks/03_escalon_3_semana.ipynb` corre 7 episodios independientes (uno por día, `SimuladorBarcosBergen` sin tocar) y los junta con `metricas.combinar_corridas` (nueva función en `metricas.py` -- ver su docstring) antes de llamar a `metricas.reporte_completo`, exactamente igual que para 1 y 2. Un día es su propio episodio (su propio reloj 6:00-24:00) -- no hay un timeline único de 10 080 minutos; ver la justificación completa en la introducción del notebook.

**Conservación:** 5429 generadas = 5406 atendidas + 0 esperando al final + **23 a bordo al final**. A diferencia de los escalones 1 y 2 (100% atendidas en ambos), aquí sí aparece gente a bordo al cierre de un día -- con 7 ventanas de cierre en vez de una, hay 7 veces más oportunidades de que alguien suba a un barco justo antes de que se acabe la hora operativa (24:00) y el barco no alcance a llegar antes del corte. Es exactamente el caso que las categorías "esperando/a bordo al final" existen para capturar correctamente (sección 4.5) -- no es un bug, es información real de la corrida.

| Métrica | Escalón 3 (semana) |
|---|---|
| % atendidas | 99.58% |
| Espera media | 10.0 min |
| Tiempo en sistema medio / máximo | 19.8 / 57.1 min |
| Espera p50 / p90 / p95 | 9.1 / 21.4 / 25.1 min |

Demanda por día: lunes-viernes entre 756 y 1121 personas; sábado y domingo, 324-326 (factor `fin_de_semana=0.4` de `demand/config/instance.yaml`, aplicado automáticamente por `generar_llegadas_semana`). El backlog de las 23 personas a bordo al cierre se concentra en 3 pares (`bryggen->laksevag`: 10, `laksevag->bryggen`: 7, `kleppesto->bryggen`: 6) -- las rutas más transitadas de la semana, donde es más probable que un barco parta justo antes del corte.

Reproducibilidad verificada igual que 1 y 2 (misma semilla de demanda semanal → misma corrida completa en los 7 días; semilla distinta → resultado distinto).

## 9. Supuestos y limitaciones

- **El entorno en sí no tiene aleatoriedad propia.** Toda la aleatoriedad del pipeline vive en la generación de demanda (`demand/`, ya con su semilla). Una vez que la tabla de grupos está fija, el simulador es 100% determinista (misma entrada → mismo resultado siempre) — es una propiedad deseable para verificar, no un defecto.
- **Escala de demanda de los escalones (0.8% y 1.2% de la población):** valores propios de este paso, elegidos solo para que el tamaño de la corrida sea manejable de verificar — no tienen relación con el 10% "oficial" de `demand/`.
- **Sin paciencia, sin pérdidas — decisión de diseño explícita, no un accidente** (sección 4.5). Toda persona/grupo generado espera hasta ser atendido o hasta que termine la corrida. Esto habilita percentiles de espera reales, sin censurar, pero también significa que la recompensa (sección 4.4) ya no distingue conexión fuerte de conexión normal (el normalizador `sobrante_normalizador_min=18` es fijo para todos), y que la presión contra dejar a alguien esperando mucho depende enteramente del término de incomodidad con techo, no de un cargo explícito de pérdida. No se verificó todavía si esto cambia el comportamiento de un agente entrenado sobre esta recompensa — es una pregunta abierta para la fase siguiente.
- **`asignar_flota` prioriza siempre la demanda local sobre la remota** (sección 4.3), sin comparar urgencia global. Como ya no hay paciencia que ponga un límite implícito, esto puede dejar esperando **indefinidamente** a alguien muy urgente en otro nodo mientras el barco atiende demanda local menos urgente, si el patrón de demanda nunca le da prioridad a ese par. En las corridas de ejemplo (sección 8) esto no llegó a pasar (100% atendidas en ambas), pero con una flota más chica o una demanda más desbalanceada sí podría — declarado explícitamente, no es la única forma razonable de diseñar la política.
- **Tiempo de espera reportado en métricas** = tiempo hasta que el destino de la persona quedó resuelto (subió a un barco) — no el tiempo de viaje a bordo después de subir. El tiempo en sistema sí suma ambos (espera + viaje) para las atendidas. Solo se calcula sobre atendidas: quien queda esperando al final de una corrida truncada no tiene un tiempo de espera "cerrado" todavía.
- **Los percentiles de espera (`metricas_por_usuario`) son la base para definir garantías de tiempo de servicio** — ese es justamente el motivo por el que se quitó la paciencia (sección 4.5). Con datos sin censurar, un percentil como "p95 de espera es 26 min" es una medición real de la corrida, no un supuesto — pero todavía no se ha usado para proponer una garantía formal; es el siguiente paso natural antes o junto con conectar el agente de RL.
- **`asignar_flota` es miope, no óptima.** Coordina los barcos libres de UN mismo paso para que no se dupliquen entre sí (bug #3), pero sigue decidiendo de a uno, en el orden en que aparecen en `env.barcos` — no evalúa todas las combinaciones posibles para encontrar la asignación conjunta óptima. Es una mejora real sobre la versión sin coordinar, pero sigue siendo una regla simple, apropiada como línea base.
- **Reposicionamiento vacío (sección 4.3):** decisión de diseño propia, no está detallada explícitamente en la especificación — se dedujo como la forma más simple y consistente de manejar demanda fuera del nodo actual del barco bajo el modelo de viajes directos punto a punto.

---

## 10. Cómo correr

```bash
cd simulacion/notebooks
jupyter nbconvert --to notebook --execute --inplace 00_preparar_demanda_escalones.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_escalon_1_verificacion.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_escalon_2_metricas.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_escalon_3_semana.ipynb

# RL (necesita `pip install stable-baselines3`, instala torch como dependencia)
jupyter nbconvert --to notebook --execute --inplace 04_entrenamiento_rl.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_comparacion_agente_vs_base.ipynb
```

Necesita que `demand/output/` y `bergen-boats/02_ruteo_navegable/output/` ya existan (pasos previos, cerrados). El notebook 00 tiene que correr antes que el 01, 02 y 03 (genera `output/escalon1/grupos.csv`, `output/escalon2/grupos.csv` y `output/escalon3/grupos_semana.csv`, que los otros leen). El 05 necesita que el 04 ya haya guardado `output/rl_ppo/modelo_ppo.zip`.

---

## 11. Demanda fresca en cada reset (`entrenamiento.py`) -- necesario para RL

Los escalones 1-3 corren la política base sobre una demanda **fija**: se genera UNA VEZ (notebook 00) y se guarda en un CSV que los notebooks 01-03 solo leen. Eso es correcto para verificar el simulador (la política no tiene nada que ajustar), pero **no sirve para entrenar un agente**: si todos los episodios de entrenamiento fueran siempre la misma tabla de grupos, el agente podría memorizar esa realización particular (qué grupo exacto aparece en qué minuto exacto) en vez de aprender una política que generaliza sobre el patrón de demanda real.

`src/entrenamiento.py`, clase `EntornoDemandaAleatoria(SimuladorBarcosBergen)`: la misma subclase sirve para las dos cosas que hacen falta, según cómo se llame `reset()`:

- **`reset(seed=None)`** (el caso normal en un loop de entrenamiento -- SB3 no manda una semilla en cada episodio): genera una demanda NUEVA cada vez, con una semilla sacada de una secuencia propia (reproducible si se dio `semilla_entrenamiento` al construir el entorno). Cada episodio ve una realización distinta del mismo patrón de fondo -- muestreo Montecarlo de episodios.
- **`reset(seed=42)`** (caso explícito -- evaluación, comparación contra la política base): genera la demanda con ESA semilla exacta, reproducible igual que `SimuladorBarcosBergen` hoy. Dos instancias distintas de `EntornoDemandaAleatoria` con `reset(seed=misma_semilla)` producen la MISMA demanda (verificado en `04_entrenamiento_rl.ipynb`), porque la generación depende solo de la semilla, no de ningún estado interno del entorno -- es lo que hace posible comparar el agente y la política base en igualdad de condiciones (sección 13).

**Por qué es un wrapper y no un cambio a `env.py`:** `env.py` recibe un `grupos_df` ya construido y nunca sabe nada de `demand/` (rutas, `intensidad_od`, `conexiones_fuertes`, etc. -- ver su docstring, "nada se recalcula aquí"). Meter la generación de demanda ahí complicaría el entorno base que política base y los escalones 1-3 siguen usando tal cual. `EntornoDemandaAleatoria` solo cambia una cosa (qué `grupos_df` usa cada episodio); todo lo demás (acciones, transición, recompensa) es exactamente `SimuladorBarcosBergen` sin tocar. Tampoco importa `demand/src/llegadas.py` directamente -- recibe la función `generar_llegadas_dia` ya importada como parámetro (`generar_llegadas_dia_fn`), igual patrón de inyección de dependencia que ya usaba `00_preparar_demanda_escalones.ipynb` (nunca un import cruzado hardcodeado entre carpetas).

---

## 12. Agente PPO (Stable-Baselines3)

**PPO, no DQN.** El `action_space` del entorno es `MultiDiscrete` (una acción por barco, sección 4.3) -- el DQN de Stable-Baselines3 no soporta `MultiDiscrete` de forma nativa, PPO sí.

**Instancia de entrenamiento:** los parámetros de `escalon_1` (franja mañana, 2 barcos, pocos grupos) -- la instancia más chica, para verificar rápido si el agente aprende algo antes de escalar. Hiperparámetros en `config/instance.yaml` → `agente` (valores actuales, tras el diagnóstico de la sección 14.2):
```yaml
agente:
  gamma: 0.99
  entrenamiento:
    escalon_base: "escalon_1"
    semilla_entrenamiento: 123     # raiz de la secuencia de demandas de entrenamiento
    total_timesteps: 150000        # ~1666 episodios de 90 pasos c/u (calibrado: 300000 hubiera tomado ~68 min)
    usar_vecnormalize: true        # VecNormalize (obs y reward) -- ver seccion 14.2
    ent_coef: 0.01                 # mas exploracion (default SB3 = 0.0)
    learning_rate: 0.0003
    n_steps: 512                   # actualiza cada ~5-6 episodios, no ~22 (default 2048)
    recompensa_overrides:          # SOLO para el entorno RL, no toca la seccion `recompensa:` de arriba
      premio_por_persona_entregada: 0.5
      peso_movimiento: 0.0         # temporal para esta fase de diagnostico
  evaluacion:
    semillas: [1001, 1002, 1003, 1004, 1005]   # fijas, fuera del rango de entrenamiento
```

**`notebooks/04_entrenamiento_rl.ipynb`, en orden:** (1) confirma demanda fresca por reset (sección 11); (2) `stable_baselines3.common.env_checker.check_env` -- pasó sin advertencias, los espacios ya declarados (`MultiDiscrete`, `Box` float32) no necesitaron ningún ajuste; (3) envuelve el entorno con `Monitor` + (si `usar_vecnormalize`) `DummyVecEnv`/`VecNormalize` (guarda `output/rl_ppo/monitor.monitor.csv` y `vecnormalize.pkl`); (4) entrena `PPO("MlpPolicy", ...)` con los hiperparámetros del config; (5) guarda el modelo (`output/rl_ppo/modelo_ppo.zip`) y grafica la curva de recompensa por episodio (interactiva, Plotly -- `output/rl_ppo/training_curve.html`).

**Resultado de la corrida actual (150 000 timesteps, 1666 episodios, combo validado):** la recompensa media de los primeros 20 episodios fue -2439.28; la de los últimos 20, -691.15 -- una mejora mucho más marcada que el primer intento (100k timesteps sin ajustes: de -2650.65 a solo -2161.18). La curva de entrenamiento (`output/rl_ppo/training_curve.html`) todavía no se había aplanado cuando se detuvo -- el agente seguía mejorando activamente. Ver sección 13 para el resultado de esta corrida comparado contra la política base, y sección 14.2 para el diagnóstico completo (por qué la primera corrida fallaba y qué se cambió).

---

## 13. Comparación: agente PPO vs. política base

`notebooks/05_comparacion_agente_vs_base.ipynb` corre las dos políticas sobre las 5 semillas de `agente.evaluacion.semillas` (misma semilla → misma demanda para ambas, vía `EntornoDemandaAleatoria.reset(seed=...)`, sección 11), y junta los 5 episodios de cada política con `metricas.combinar_corridas` -- mismo mecanismo que el escalón 3, reusando el 100% de las funciones de `metricas.py` ya existentes.

**Versión original (100k timesteps, sin ajustes) -- política degenerada:** % atendidas 27.6% vs. 88.0% de la base, ocupación 1.15 vs. 3.50, apenas 79 movimientos vs. 155. Se armó un ciclo de diagnóstico completo (sección 14.2) que encontró la causa principal (falta de normalización de observaciones/recompensa) y la corrigió.

**Después del diagnóstico (150k timesteps, combo validado -- sección 14.2, `agente.entrenamiento.recompensa_overrides` activo): mejora enorme, todavía por debajo de la política base.**

| Métrica (agregada sobre las 5 semillas de evaluación) | Agente PPO (tras el diagnóstico) | Agente PPO (original, sin diagnóstico) | Política base |
|---|---|---|---|
| % atendidas | **77.2%** | 27.6% | 88.0% |
| Espera media | 20.5 min | 10.5 min | 16.4 min |
| Tiempo en sistema medio | 30.4 min | 21.2 min | 26.1 min |
| Tiempo en sistema máximo | 117.1 min | 36.3 min | 69.7 min |
| Movimientos totales (5 episodios) | 177 | 79 | 155 |
| Ocupación media | 2.97 | 1.15 | 3.50 |
| Recompensa total (5 episodios) | -4807.00 | -17 913.76 | -2009.26 |

**Lectura:** de 27.6% a 77.2% atendidas es un salto enorme -- pasó de una política que casi no se mueve a una que atiende a 3 de cada 4 personas, con ocupación de flota (2.97) ya cerca de la de la base (3.50). Sigue por debajo de la base en % atendidas y tiene una cola de espera más larga (máximo 117 min vs. 69.7 min -- al menos un par, `laksevag->sandviken`, queda especialmente desatendido). La curva de entrenamiento (sección 12) NO se había aplanado cuando se detuvo (recompensa media: -2439 en los primeros 20 episodios de esta corrida, -691 en los últimos 20, de 1666 episodios entrenados) -- el agente seguía mejorando activamente. **La conclusión del ciclo de diagnóstico (sección 14.2) es clara: la palanca que falta es más presupuesto de entrenamiento**, no otro fix estructural -- con el combo ya validado (VecNormalize, premio de entrega, sin penalización de movimiento temporal, más entropía/n_steps ajustados), subir `total_timesteps` más allá de 150k es el paso directo antes de evaluar si PPO puede igualar o superar a "nearest-available" en esta instancia.

**Nota sobre las recompensas reportadas:** desde el diagnóstico, la comparación usa `agente.entrenamiento.recompensa_overrides` (premio de entrega activo, sin penalización de movimiento) para AMBAS políticas -- así el reward es comparable 1:1. No es la misma fórmula que reportan las secciones 4.4/8 para escalones 1-3 (esas usan la recompensa de producción, sin overrides) -- ver sección 14.2 para el porqué.

**¿El agente simplemente no mueve la flota?** No -- es justo lo contrario. `metricas.decisiones_por_barco` (nueva, cuenta cuántas veces cada barco decidió "esperar" vs. moverse, cruzando `log_eventos` con el estado en ese momento) muestra que el agente espera en solo **13.0%** de sus decisiones, contra **47.5%** de la política base -- un barco del agente (`barco_1`) no espera NUNCA en las 5 semillas de evaluación. El agente mueve la flota más (177 movimientos vs. 155, 1334 min navegados vs. 1224) pero sirve MENOS gente (77.2% vs. 88.0%) -- el problema no es pasividad, es que parte de ese movimiento extra no está bien dirigido (reposicionamientos que no terminan sirviendo a nadie, o llegan tarde) -- visible en la comparación por par (arriba) y en la visualización paso a paso de abajo. Tabla completa en `output/comparacion/decisiones_por_barco_comparado.csv`.

**Visualización paso a paso (una semilla de ejemplo, las dos políticas):** `notebooks/05_comparacion_agente_vs_base.ipynb`, al final -- mapa real + animación GIF de los 90 pasos completos para el agente y para la base por separado (`output/comparacion/animacion_agente_semilla1001.gif` / `animacion_base_semilla1001.gif`), inspector en un par de minutos concretos para las dos, y el reproductor completo con botones de paso a paso (mismo patrón que el escalón 1, sección 8.2) para poder seguir exactamente qué decide cada barco, minuto a minuto, en las dos políticas sobre la misma demanda.

Tablas completas (por par origen-destino comparado) y gráficas interactivas (heatmaps lado a lado, recompensa por semilla) en `output/comparacion/` (`tabla_comparativa.csv`, `por_par_comparado.csv`, `heatmaps_comparados.html`, `reward_por_semilla.html`).

---

## 14. Diagnóstico: gráficas confusas y agente PPO degenerado (y cómo se corrigieron)

### 14.1 Gráficas -- por qué parecían "varias líneas del mismo barco"

La gráfica de ocupación de flota del escalón 3 (semana) mostraba algo que parecía varias líneas del mismo barco superpuestas. **Causa raíz confirmada leyendo el código, no un supuesto:** `graficar_ocupacion_flota`/`graficar_perfil_espera`/`graficar_desglose_recompensa` usaban `minuto_del_dia` (0-1439) como eje X. En una corrida combinada de varios episodios (escalón 3 = 7 días, o la comparación agente-vs-base = 5 semillas de evaluación, ver `metricas.combinar_corridas`) ese valor se REINICIA en cada episodio -- la misma línea de un barco se dibujaba repetida 7 veces, superpuesta en el mismo rango 0-1439.

**Arreglo:** las 5 gráficas de métricas (`src/visualizacion.py`) se reescribieron en **Plotly** (interactivo -- ya estaba instalado) en vez de matplotlib, y el eje de tiempo pasó a ser el **índice de paso** (0..N-1, siempre monótono, nunca se superpone) en vez de `minuto_del_dia` crudo. Para no perder la hora real: en una corrida de un solo episodio, los ticks del eje muestran `HH:MM`; en una combinada, los ticks marcan el inicio de cada episodio ("Ep 1", "Ep 2", ...) y la hora exacta de cada punto aparece al pasar el mouse. La leyenda de `fleet_occupancy.html` ya permite aislar un barco (click) o verlos todos (doble click) -- resuelve "elegir qué barco ver" sin controles aparte. Todo en inglés desde esta reescritura (títulos, ejes, leyendas).

**Bug encontrado y corregido durante la propia reescritura:** la primera versión detectaba los límites de episodio mirando si `minuto_del_dia` bajaba entre un paso y el siguiente. Para un escalón cuyo `hora_fin_min` es un múltiplo exacto de 1440 (medianoche -- escalón 2/3, que terminan a las 24:00), el ÚLTIMO frame de CADA episodio cae justo en `t_actual_min == 1440`, y `minuto_del_dia = t_actual_min % 1440` da `0` -- indistinguible de un cambio real de episodio. Esto hacía que el escalón 3 (7 días reales) mostrara 8 "episodios" en vez de 7. Se corrigió de raíz: `metricas.combinar_corridas` ahora guarda `limites_episodio` (cuántos frames aportó cada corrida fuente, un número exacto, conocido de antemano) en el objeto combinado, y `visualizacion._eje_tiempo` lo usa directo en vez de adivinar mirando los datos. Verificado con la corrida real de escalón 3: exactamente 7 episodios, límites en 0/541/1082/.../3246 (541 = 540 pasos + el frame inicial de `reset()`).

**Se dejaron en matplotlib** (no son series de tiempo, y no eran la parte confusa): el mapa real con barcos moviéndose (`dibujar_frame`, `animar_corrida`, `inspeccionar`, `reproductor_interactivo`) -- siguen igual que antes, sección 8.1-8.2.

**Nota técnica:** se probó también exportar una imagen estática (`.png`) de cada gráfica con `kaleido` (motor de renderizado de Plotly, headless-Chrome) además del `.html` interactivo -- causaba que notebooks con muchas gráficas + celdas de cómputo posteriores se colgaran bajo `nbconvert` (mismo síntoma que un bug ya visto antes en este proyecto con acumulación de figuras de matplotlib, pero esta vez con el manejo interno de procesos de `kaleido`). Se sacaron esas llamadas -- el entregable real (el `.html` interactivo) no las necesitaba, y una imagen estática se puede generar aparte si hace falta para el informe, fuera del camino crítico de estos notebooks.

### 14.2 Agente PPO -- diagnóstico y arreglo, paso a paso

**Punto de partida:** el primer entrenamiento (100k timesteps sobre `escalon_1`, sin ajustes) aprendió una política degenerada -- 27.6% atendidas vs. 88.0% de la política base, ocupación 1.15 vs. 3.50, apenas 79 movimientos vs. 155 (sección 13, versión original). Se armó un ciclo de diagnóstico ordenado, verificando cada cambio antes del siguiente -- detalle completo, con cada corrida y sus números, en `output/rl_ppo/bitacora_experimentos.md`; el setup completo (qué entra, qué semillas, qué significa cada parámetro) en `output/rl_ppo/README.md`.

**B.1 -- Instancia de juguete, receta original.** Antes de tocar nada, se verificó si PPO podía aprender algo en el problema más simple posible (1 barco, 2 nodos, `escalones.escalon_toy`, nuevo en el config). Con la receta original (sin normalización, sin premio positivo, `ent_coef=0.0` por defecto de SB3): el agente entrenado sirvió **0% de la demanda** (vs. 67.1% de la política base en la misma instancia), pese a que `check_env` y la mecánica de recompensa ya estaban verificadas correctas. Esto descarta un bug estructural (el entorno funciona bien) y apunta a un problema de escala/forma de la recompensa/exploración -- exactamente lo que se sospechaba.

**B.2 -- `VecNormalize`.** Se envolvió el entorno en `DummyVecEnv` + `VecNormalize` (normaliza observaciones y recompensa con estadísticas corridas). Resultado en la instancia de juguete: **0% → 67.1%, empatando exactamente con la política base**, y con mejor reward. Esta fue la causa principal.

**B.3 -- Premio positivo por entrega.** Se agregó un término nuevo a la recompensa (`recompensa.py`, `premio_por_persona_entregada`, activado solo para las corridas de RL vía `agente.entrenamiento.recompensa_overrides` -- no afecta escalones 1-3 ni la política base, que siguen con el valor default 0.0). En la instancia de juguete el % atendidas se mantuvo en 67.1% (parece ser un techo real de esa instancia con 1 solo barco), pero el reward crudo mejoró de forma consistente.

**B.4 -- Penalización de movimiento en 0 (temporal).** `peso_movimiento` bajado a 0 solo para el entrenamiento RL (mismo mecanismo de override), para que el agente aprenda a servir antes de optimizar eficiencia -- explícitamente temporal para esta fase, no el valor de producción de escalones 1-3.

**B.5 -- Exploración e hiperparámetros de PPO.** `ent_coef=0.01` (más exploración que el default 0.0), `n_steps=512` (actualiza cada ~5-6 episodios de 90 pasos en vez de ~22).

**B.6 -- Consistencia train/eval.** Se verificó (y se dejó un `assert` explícito en `05_comparacion_agente_vs_base.ipynb`, que compara la FORMA del vector de observación del modelo cargado contra la de la instancia de evaluación) que entrenamiento y evaluación usan el mismo escalón -- **no era un problema real en este proyecto** (los dos notebooks ya leían `cfg_entren["escalon_base"]` de la misma clave), pero queda la verificación estructural para que un desajuste futuro no pase desapercibido.

**B.7 -- Entrenamiento final sobre `escalon_1` (la instancia real), con el combo validado, más timesteps (150k).** Resultado: **27.6% -> 77.2% atendidas** -- mejora enorme, todavía por debajo del 88.0% de la política base (detalle completo en la sección 13, más arriba, y en `output/rl_ppo/bitacora_experimentos.md`). La curva de entrenamiento no se había aplanado -- la palanca directa para la siguiente iteración es más `total_timesteps`, no otro fix estructural.

**Bug real encontrado y corregido durante esta corrida:** la primera versión de `correr_agente` (en `05_comparacion_agente_vs_base.ipynb`) steppeaba el entorno A TRAVÉS del `DummyVecEnv` (`venv.step(...)`) para poder usar las estadísticas de `VecNormalize` -- pero `DummyVecEnv.step_wait()` auto-resetea el entorno interno en el mismo `step()` en que `done=True` (comportamiento estándar de estos wrappers, para no perder un paso al encadenar episodios), lo que borraba `atendidas_historico`/`log_eventos` ANTES de poder leerlos -- `metricas.verificar_conservacion` lo detectó de inmediato (todo en 0 salvo las generadas). Se corrigió steppeando el entorno CRUDO directamente y usando `VecNormalize` solo para normalizar la observación antes de `model.predict()` -- mismo patrón ya usado en el diagnóstico sobre `escalon_toy`.

**Meta de este ciclo (como se pidió):** que el agente al menos iguale a la política base -- superarla queda para un siguiente escalón del proyecto.

---

## Siguiente paso

Con el diagnóstico de la sección 14 aplicado: revisar los resultados finales de B.7 en la sección 13 y decidir si hace falta seguir subiendo el presupuesto de entrenamiento, o si el combo ya es competitivo. Después: usar los percentiles de `metricas_por_usuario` para proponer una garantía de tiempo de servicio con sustento en datos (sección 9), y actualizar el informe LaTeX del proyecto (`docs/informe/`) con estos resultados.
