# Política base ("nearest-available") -- referencia + verificación del simulador

**Qué es esto, en una frase:** la política de referencia (una regla fija, no aprendida) y los notebooks que, corriéndola sobre demanda de tamaño creciente, verifican a fondo que el motor (`simulador/`) hace lo que debe -- antes de conectar cualquier agente de RL (`modelo_rl/`).

---

## 1. Cómo está organizado

```
politica_base/
├── README.md                      # este archivo
├── src/
│   └── politica_base.py           # la regla "nearest-available" + coordinación de flota
├── notebooks/
│   ├── 00_preparar_demanda_escalones.ipynb   # genera la demanda de prueba (escalones 1-3)
│   ├── 01_escalon_1_verificacion.ipynb       # franja mañana, verificación a ojo + animación
│   ├── 02_escalon_2_metricas.ipynb           # día completo, métricas agregadas (sin animación)
│   └── 03_escalon_3_semana.ipynb             # semana completa (7 días), métricas agregadas
└── output/
    ├── escalon1/                  # todo lo que produce el notebook 01
    ├── escalon2/                  # todo lo que produce el notebook 02, misma estructura de archivos
    └── escalon3/                  # todo lo que produce el notebook 03 (semana)
```

Los notebooks importan `simulador/src` (motor: `env.py`, `estado.py`, `recompensa.py`, `metricas.py`, `visualizacion.py`) además de `src/politica_base.py` propio, y leen `simulador/config/instance.yaml` (única fuente de verdad de parámetros compartidos). Cada escalón tiene su propia carpeta en `output/` con los mismos nombres de archivo adentro (`grupos.csv`, `metricas_por_par.csv`, `wait_profile.html`, etc.) -- fácil comparar un escalón contra el otro abriendo la carpeta correspondiente. La única diferencia de contenido: `escalon1/` tiene `animacion.gif` (la corrida es chica, 90 pasos, se puede animar); `escalon2/` y `escalon3/` no (corridas largas -- un GIF de esa duración pesa y tarda mucho más, y la verificación visual paso a paso ya se hizo a fondo en el escalón 1). `escalon3/` usa `grupos_semana.csv` en vez de `grupos.csv` (7 días concatenados, columna `dia`).

---

## 2. La política de referencia ("nearest-available")

`src/politica_base.py` tiene dos funciones:

- **`politica_base(barco, estado, matriz_tiempos, cfg)`** -- decide UN barco: prioriza la demanda que sale de su nodo actual (servible de inmediato); si no hay nada ahí, considera reposicionarse vacío hacia el nodo con la demanda más urgente del resto del sistema.
- **`asignar_flota(barcos_libres, estado, matriz_tiempos, capacidad_barco, cfg)`** -- aplica lo anterior a **varios barcos libres a la vez, uno por uno**, descontando localmente lo que cada barco ya "se llevaría" antes de decidir el siguiente. Es la forma correcta de usar la política con más de un barco (ver "coordinación de flota" más abajo) -- la que realmente se usa en los notebooks.

Es una heurística **competente, no trivial**: no es solo "vas al vecino más cercano" -- combina prioridad estricta a demanda local, reposicionamiento a la demanda remota más urgente cuando no hay nada local, y coordinación entre varios barcos libres en el mismo paso para no duplicar esfuerzo. Sirve como línea base seria contra la que comparar el agente de RL (`comparacion/`) -- está escrita como funciones independientes, separadas del motor, precisamente para poder compararla contra otra política sin tocar el simulador.

### 2.1 Quién sube a cada barco, y a dónde va

**La política NO elige a quién recoge, elige el NODO destino.** Quién sube es una regla fija del simulador (`simulador/`): las colas están separadas por par origen-destino (12 colas, una por cada combinación). Cuando un barco libre en A recibe la orden "ir a B", solo puede embarcar de la cola exacta A→B, en orden de llegada, hasta llenar el barco -- nunca lleva gente con destinos mixtos, y al llegar a B baja a todos. Es un viaje directo punto a punto, no una ruta con paradas intermedias.

**¿Y si la gente que más necesita un barco está en OTRO nodo?** Regla importante, no la única forma razonable de hacerlo:

> **La política SIEMPRE prioriza la demanda del nodo donde el barco ya está, sobre cualquier demanda de otro nodo -- sin importar cuánto tiempo lleve esperando la gente de otros nodos.**

Ejemplo concreto: un barco queda libre en Bryggen. Ahí mismo hay 5 personas que acaban de llegar (esperan 1 minuto) queriendo ir a Laksevåg. En Kleppestø hay 1 persona que lleva 20 minutos esperando ir a Sandviken. **La política manda el barco a buscar a las 5 de Laksevåg, no a la persona de Kleppestø** -- aunque esa persona lleve muchísimo más tiempo esperando y esté mucho más cerca de perderse. La regla no compara "qué tan urgente es cada uno" de forma global; primero agota TODA la demanda local (por más pequeña o reciente que sea) y solo mira otros nodos cuando en el propio no queda absolutamente nadie esperando.

¿Por qué se diseñó así? Para no dejar "abandonada" gente que el barco ya podría atender de inmediato, a cambio de perseguir a alguien que todavía requiere viajar. Es una decisión de diseño razonable pero **no es la única posible** -- una alternativa sería comparar la urgencia de TODOS los candidatos (locales y remotos) en una sola lista, y que el barco a veces se vaya a buscar a alguien lejano si es mucho más urgente que la demanda local. Esa alternativa no está implementada; la actual es más simple de explicar y de verificar, pero puede llevar a que alguien muy urgente en otro nodo espere mucho más tiempo de lo razonable. Como el simulador no purga a nadie por paciencia (`simulador/README.md`, sección 4.5), esa espera ya no tiene un límite implícito -- puede crecer indefinidamente si el patrón de demanda no le da nunca prioridad a ese par. Queda declarado como límite conocido, no escondido (sección 5).

Mecánicamente, en `politica_base`: primero arma la lista `candidatos_directos` (solo pares que SALEN del nodo actual del barco); si esa lista no está vacía, elige de ahí el más urgente y punto -- nunca mira `candidatos_reposicion` (demanda de otros nodos) a menos que `candidatos_directos` esté completamente vacío. Cuando sí se reposiciona, el barco viaja **vacío** hasta el nodo con la demanda remota más urgente y ahí, en su siguiente momento libre, decide de nuevo con información fresca -- es una decisión miope (no planea las dos etapas de una vez).

### 2.2 Coordinación de flota -- por qué `asignar_flota` existe

Con la política aplicada de forma INDEPENDIENTE por barco (cada uno mirando la misma foto del mundo, antes de que cualquiera de los dos suba gente), dos barcos libres en el mismo nodo y momento pueden decidir ambos "ir a Bryggen" pensando que hay, por ejemplo, 11 personas esperando ahí -- el primero en procesarse se las lleva todas, el segundo viaja **vacío** hasta Bryggen, pagando la penalización de movimiento sin servir a nadie. Se detectó durante la verificación original inspeccionando un minuto concreto de una corrida de ejemplo (`visualizacion.inspeccionar`) y se confirmó en el log de texto.

`asignar_flota` corrige esto asignando los barcos libres de a uno, con una copia LOCAL de las colas que se va descontando conforme cada barco decide -- así el siguiente barco de la lista ve la cola ya reducida por lo que el anterior se llevaría. Efecto medido en su momento (bajo el sistema de paciencia, antes de eliminarla del todo -- `simulador/README.md` sección 4.5): el % de cumplimiento subió de 53% a 90% -- no fue un detalle menor. Es exactamente la clase de coordinación que convierte "una regla simple aplicada varias veces" en "una heurística competente" -- sin ella, más barcos en la flota no necesariamente sirven más gente.

---

## 3. Los escalones (instancias de prueba, demanda creciente)

| | Escalón 1 | Escalón 2 | Escalón 3 |
|---|---|---|---|
| Cuándo | Franja mañana (6-9h) | Día completo (6-24h) | Semana completa (7 días, 6-24h c/u) |
| Barcos | 2 | 3 | 3 |
| Grupos / personas | 23 / 202 | 133 / 1106 | 620 / 5429 |
| Pasos de 2 min | 90 | 540 | 7 × hasta 540 (7 episodios independientes) |
| Para qué | Verificar a ojo, log de texto, animación | Métricas agregadas, sin animación | Métricas agregadas sobre entre-semana + fin de semana |
| Notebook | `01_escalon_1_verificacion.ipynb` | `02_escalon_2_metricas.ipynb` | `03_escalon_3_semana.ipynb` |

La demanda de cada escalón se generó con `demand/src/llegadas.py` **sin tocarlo** -- solo se le pasó un `porcentaje_poblacion_dia` más bajo que el oficial (10%, en `demand/config/instance.yaml`, intacto), calibrado por prueba y error hasta acercarse a "~20-30 grupos" en el escalón 1. El escalón 3 reusa la densidad y flota del escalón 2 sobre `generar_llegadas_semana` (`demand/src/llegadas.py`) en vez de `generar_llegadas_dia` -- 7 días (lunes=0..domingo=6), con el factor entre-semana/fin-de-semana ya resuelto adentro.

---

## 4. Resultados de verificación

**Conservación** (`metricas.verificar_conservacion`): toda persona generada debe terminar contabilizada.

| | Escalón 1 | Escalón 2 | Escalón 3 |
|---|---|---|---|
| Generadas | 202 | 1106 | 5429 |
| = Atendidas + | 202 | 1106 | 5406 |
| Esperando al final + | 0 | 0 | 0 |
| A bordo al final | 0 | 0 | 23 |

Escalones 1 y 2: 100% atendidas (la ventana de tiempo alcanza para vaciar el backlog). Escalón 3: **99.58%** -- con 7 ventanas de cierre en vez de una, hay 7 veces más oportunidades de que alguien suba a un barco justo antes de que se acabe la hora operativa (24:00) y el barco no alcance a llegar antes del corte. No es un bug: es exactamente el caso que las categorías "esperando/a bordo al final" existen para capturar correctamente (el simulador nunca "pierde" a nadie, `simulador/README.md` sección 4.5).

**Globales** (`metricas.metricas_globales`):

| Métrica | Escalón 1 | Escalón 2 | Escalón 3 |
|---|---|---|---|
| % atendidas | 100.0% | 100.0% | 99.58% |
| Espera media | 17.4 min | 10.9 min | 10.0 min |
| Tiempo en sistema medio / máximo | 27.2 / 54.0 min | 20.5 / 53.6 min | 19.8 / 57.1 min |
| Espera p50 / p90 / p95 | 15.6 / 33.4 / 33.5 min | 9.4 / 21.6 / 26.3 min | 9.1 / 21.4 / 25.1 min |

El escalón 2, pese a tener 5.5× más demanda y 9× más pasos que el escalón 1 (con solo 1 barco más), termina con espera media y p90/p95 **más bajos** -- tiene sentido: el escalón 1 concentra toda su demanda en 3h de franja pico, sin margen para que el backlog se drene entre picos, mientras que el escalón 2 cubre 18h con franjas de alta y baja intensidad alternándose, dando más oportunidades de que la flota se ponga al día en los valles. Demanda del escalón 3 por día: lunes-viernes entre 756 y 1121 personas; sábado y domingo, 324-326 (factor `fin_de_semana=0.4` de `demand/config/instance.yaml`). El backlog de las 23 personas a bordo al cierre se concentra en 3 pares (`bryggen->laksevag`: 10, `laksevag->bryggen`: 7, `kleppesto->bryggen`: 6) -- las rutas más transitadas de la semana.

Reproducibilidad verificada en los tres escalones: misma semilla de demanda → misma corrida exacta; semilla distinta → resultado distinto.

**Por par origen-destino, por barco, por usuario (percentiles), y backlog al final** -- tablas completas en cada notebook (`metricas.metricas_por_par`, `metricas_por_barco`, `metricas_por_usuario`, `metricas.sin_atender_al_final_por_par`), guardadas también en `output/escalonN/metricas_por_par.csv` / `metricas_por_barco.csv`. En escalones 1 y 2, `sin_atender_al_final_por_par` da una tabla vacía (nadie quedó sin atender), consistente con el 100%.

**Gráficas** (`simulador/src/visualizacion.py`, interactivas -- Plotly; `output/escalonN/*.html`): perfil temporal de personas esperando (`wait_profile.html`), ocupación de la flota en el tiempo (`fleet_occupancy.html`), heatmap de % atendidas por par (`pct_served_heatmap.html`), desglose de recompensa en el tiempo (`reward_breakdown.html`), barras de backlog por par (`backlog_by_pair.html`) -- misma función, para los tres escalones. Zoom/pan nativo, valores exactos al pasar el mouse; la leyenda de `fleet_occupancy.html` permite aislar un barco (click) o volver a mostrarlos todos (doble click).

`escalon1/` además tiene `animacion.gif` (90 frames, uno por paso), `visualizacion.inspeccionar` (un minuto concreto -- funciona siempre, no requiere kernel vivo) y `visualizacion.reproductor_interactivo` (botones de paso a paso, **solo funciona con un kernel de Jupyter vivo**: hay que abrir el notebook en VS Code/Jupyter Lab y correr las celdas ahí, no alcanza con verlo ya ejecutado). El mapa de fondo usa Esri.WorldGrayCanvas (sin llave de API) -- CartoDB.Positron (el que usa `bergen-boats/`) empezó a exigir API key, y OpenStreetMap bloqueó la solicitud por política de uso; se descarga una sola vez por corrida, no una vez por frame.

---

## 5. Supuestos y limitaciones (de la política de referencia)

- **`asignar_flota` prioriza siempre la demanda local sobre la remota** (sección 2.1), sin comparar urgencia global. Como el simulador no tiene paciencia, esto puede dejar esperando **indefinidamente** a alguien muy urgente en otro nodo mientras el barco atiende demanda local menos urgente, si el patrón de demanda nunca le da prioridad a ese par. En las corridas de los escalones esto no llegó a pasar (99.6-100% atendidas), pero con una flota más chica o una demanda más desbalanceada sí podría -- declarado explícitamente, no es la única forma razonable de diseñar la política.
- **`asignar_flota` es miope, no óptima.** Coordina los barcos libres de UN mismo paso para que no se dupliquen entre sí (sección 2.2), pero sigue decidiendo de a uno, en el orden en que aparecen en la lista de barcos -- no evalúa todas las combinaciones posibles para encontrar la asignación conjunta óptima. Es una mejora real sobre la versión sin coordinar, pero sigue siendo una regla simple, apropiada como línea base.
- **Reposicionamiento vacío:** decisión de diseño propia, no está detallada explícitamente en la especificación original -- se dedujo como la forma más simple y consistente de manejar demanda fuera del nodo actual del barco bajo el modelo de viajes directos punto a punto.
- **Escala de demanda de los escalones (0.8%/1.2% de la población):** valores propios de esta verificación, elegidos solo para que el tamaño de la corrida sea manejable de verificar a ojo -- no tienen relación con el 10% "oficial" de `demand/`.

---

## 6. Cómo correr

```bash
cd politica_base/notebooks
jupyter nbconvert --to notebook --execute --inplace 00_preparar_demanda_escalones.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_escalon_1_verificacion.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_escalon_2_metricas.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_escalon_3_semana.ipynb
```

Necesita que `demand/output/` y `bergen-boats/02_ruteo_navegable/output/` ya existan (pasos previos, cerrados). El notebook 00 tiene que correr antes que 01, 02 y 03 (genera `output/escalon1/grupos.csv`, `output/escalon2/grupos.csv` y `output/escalon3/grupos_semana.csv`, que los otros leen).
