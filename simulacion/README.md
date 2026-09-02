# Simulador de despacho de barcos (entorno Gymnasium, sin aprendizaje todavía)

**Qué es esto, en una frase:** un programa que modela minuto a minuto (en pasos de 2 min) qué hacen los barcos y los pasajeros en Bergen, controlado por una regla fija ("nearest-available"), para poder verificar que el mundo funciona bien ANTES de conectarle un agente que aprenda a controlarlo.

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
│   ├── metricas.py           # verificación de conservación + métricas detalladas
│   └── visualizacion.py      # mapa animado, gráficas, e inspector de un minuto concreto
├── notebooks/
│   ├── 00_preparar_demanda_escalones.ipynb   # genera la demanda de prueba (2 tamaños)
│   ├── 01_escalon_1_verificacion.ipynb        # franja mañana, verificación a ojo + animación
│   └── 02_escalon_2_metricas.ipynb            # día completo, métricas agregadas (sin animación)
└── output/
    ├── escalon1/              # todo lo que produce el notebook 01
    └── escalon2/              # todo lo que produce el notebook 02, misma estructura de archivos
```

Cada escalón tiene su propia carpeta en `output/` con los mismos nombres de archivo adentro (`grupos.csv`, `metricas_por_par.csv`, `heatmap_cumplimiento.png`, etc.) — así no hay que andar leyendo prefijos largos, y es fácil comparar un escalón contra el otro abriendo la carpeta correspondiente. La única diferencia de contenido: `escalon1/` tiene `animacion.gif` e `inspeccion_minuto400.png` (la corrida es chica, 90 pasos, se puede animar); `escalon2/` no (540 pasos — un GIF de esa duración pesa y tarda mucho más, y la verificación visual paso a paso ya se hizo a fondo en el escalón 1, no hacía falta repetirla).

**Cómo leer esto si nunca programaste en Python:** cada archivo de `src/` es una "pieza" con una responsabilidad. Los notebooks son los que las juntan y las corren, mostrando resultados. Nada se ejecuta solo con crear los archivos — hay que abrir un notebook y correrlo (ya está corrido y guardado, puedes abrirlo y ver los resultados sin volver a correrlo).

**Qué es esa primera línea `from __future__ import annotations` que aparece en casi todos los archivos:** es una instrucción del lenguaje Python, no algo específico de este proyecto. Sin ella, cuando escribimos una "anotación de tipo" como `nodos: list[str]` o `politica: str | None` (para decir "esto debería ser una lista de strings", "esto puede ser un string o nada"), Python las evalúa de inmediato al cargar el archivo, y en algunas situaciones eso falla (por ejemplo, si una clase se refiere a sí misma en su propia anotación, antes de terminar de definirse). Esa línea le dice a Python "trata las anotaciones de tipo como texto, no las evalúes ahora" — evita ese problema y es una práctica estándar recomendada en Python moderno, no algo que inventamos para este proyecto.

---

## 2. Qué es Gymnasium (y qué NO es)

**Gymnasium no trae simuladores prearmados para elegir.** Es una librería que define un **molde estándar**: cualquier "entorno" (environment) debe ser una clase de Python con exactamente dos métodos:

- `reset()` → arranca el mundo desde cero, devuelve el estado inicial.
- `step(accion)` → recibe una acción, avanza el mundo un paso, devuelve `(nuevo_estado, recompensa, terminó, se_truncó, info_extra)`.

Ese molde es TODO lo que Gymnasium impone. Nosotros escribimos el 100% de la lógica real (qué hace un barco, cómo suben y bajan los pasajeros) en `src/env.py`, clase `SimuladorBarcosBergen`. Gymnasium no sabe nada de barcos ni de Bergen — solo garantiza que nuestra clase "hable el mismo idioma" que espera cualquier librería de RL después (Stable-Baselines3, mencionada en la especificación §11, fase siguiente). Es como un enchufe: Gymnasium estandariza la forma del enchufe, nosotros construimos el aparato.

**El "tipo" de simulación** (una pregunta distinta, sobre la mecánica interna, no sobre Gymnasium): es una **simulación de tiempo discreto en pasos fijos de 2 minutos** — cada `step()` siempre avanza exactamente 2 minutos, nunca menos ni más (a diferencia de una simulación "por eventos", que salta directo al próximo momento interesante). La especificación (§3) pide explícitamente pasos fijos, por simplicidad y porque encaja natural con el `step()` de Gymnasium.

**¿Sí estamos usando Gymnasium de verdad, o solo el nombre?** Sí, de verdad: `src/env.py`, línea 27, `class SimuladorBarcosBergen(gym.Env):` — hereda literalmente de la clase base de la librería (`import gymnasium as gym`). Eso obliga (y verifica en tiempo de ejecución) a que la clase tenga `action_space` y `observation_space` bien declarados (líneas 90-92: `MultiDiscrete` para las acciones, `Box` para el vector aplanado) y los métodos `reset()`/`step()` con la firma exacta que la librería espera. No es una simulación casera a la que le pusimos "Gymnasium" de nombre — si `SimuladorBarcosBergen` no cumpliera el contrato, Stable-Baselines3 (fase siguiente) no podría usarla.

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

| | Escalón 1 | Escalón 2 |
|---|---|---|
| Cuándo | Franja mañana (6-9h) | Día completo (6-24h) |
| Barcos | 2 | 3 |
| Grupos / personas | 23 / 202 | 133 / 1106 |
| Pasos de 2 min | 90 | 540 |
| Para qué | Verificar a ojo, log de texto, animación | Métricas agregadas, sin animación (ver sección 8.1) |
| Notebook | `01_escalon_1_verificacion.ipynb` | `02_escalon_2_metricas.ipynb` |

La demanda de cada escalón se generó con `demand/src/llegadas.py` **sin tocarlo** — solo se le pasó un `porcentaje_poblacion_dia` más bajo que el oficial (10%), calibrado por prueba y error hasta acercarse a "~20-30 grupos" (escalón 1) que pedía la especificación. `demand/config/instance.yaml` sigue intacto en 10%.

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

**Gráficas** (`src/visualizacion.py`, `output/escalon1/*.png` y `output/escalon2/*.png`): perfil temporal de personas esperando, ocupación de la flota en el tiempo, heatmap de % atendidas por par (4×4), desglose de recompensa en el tiempo, barras de backlog (sin atender al final) por par — mismas gráficas, misma función, para los dos escalones.

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
```

Necesita que `demand/output/` y `bergen-boats/02_ruteo_navegable/output/` ya existan (pasos previos, cerrados). El notebook 00 tiene que correr antes que el 01 y el 02 (genera `output/escalon1/grupos.csv` y `output/escalon2/grupos.csv`, que los otros dos leen).

## Siguiente paso

Los dos escalones ya corren, sin paciencia, y dan métricas consistentes con datos de espera sin censurar. Sigue: usar los percentiles de `metricas_por_usuario` para proponer una garantía de tiempo de servicio con sustento en datos (sección 9), y conectar el agente de RL (DQN vía Stable-Baselines3, según la especificación §11) para compararlo contra la línea base medida aquí. Después, actualizar el informe LaTeX del proyecto (`docs/informe/`) con estos resultados.
