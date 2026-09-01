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
│   └── 01_escalon_1_verificacion.ipynb        # corre el simulador y muestra qué pasó
└── output/                   # lo que producen los notebooks (CSV, PNG, GIF)
```

**Cómo leer esto si nunca programaste en Python:** cada archivo de `src/` es una "pieza" con una responsabilidad. Los notebooks son los que las juntan y las corren, mostrando resultados. Nada se ejecuta solo con crear los archivos — hay que abrir un notebook y correrlo (ya está corrido y guardado, puedes abrirlo y ver los resultados sin volver a correrlo).

---

## 2. Qué es Gymnasium (y qué NO es)

**Gymnasium no trae simuladores prearmados para elegir.** Es una librería que define un **molde estándar**: cualquier "entorno" (environment) debe ser una clase de Python con exactamente dos métodos:

- `reset()` → arranca el mundo desde cero, devuelve el estado inicial.
- `step(accion)` → recibe una acción, avanza el mundo un paso, devuelve `(nuevo_estado, recompensa, terminó, se_truncó, info_extra)`.

Ese molde es TODO lo que Gymnasium impone. Nosotros escribimos el 100% de la lógica real (qué hace un barco, cómo suben los pasajeros, cuándo se pierden) en `src/env.py`, clase `SimuladorBarcosBergen`. Gymnasium no sabe nada de barcos ni de Bergen — solo garantiza que nuestra clase "hable el mismo idioma" que espera cualquier librería de RL después (Stable-Baselines3, mencionada en la especificación §11, fase siguiente). Es como un enchufe: Gymnasium estandariza la forma del enchufe, nosotros construimos el aparato.

**El "tipo" de simulación** (una pregunta distinta, sobre la mecánica interna, no sobre Gymnasium): es una **simulación de tiempo discreto en pasos fijos de 2 minutos** — cada `step()` siempre avanza exactamente 2 minutos, nunca menos ni más (a diferencia de una simulación "por eventos", que salta directo al próximo momento interesante). La especificación (§3) pide explícitamente pasos fijos, por simplicidad y porque encaja natural con el `step()` de Gymnasium.

---

## 3. De dónde sale cada parámetro (nada se inventa ni se duplica)

Regla del proyecto (ya usada en `demand/`): cada carpeta es dueña de sus propios parámetros; las demás los **leen** de la fuente, nunca los copian.

| Parámetro | Vive en | Por qué ahí |
|---|---|---|
| Capacidad del barco (20), nodo inicial (Bryggen), conexiones fuertes | `bergen-boats/config/instance.yaml` → `flota`, `garantia` | Son propiedades de la flota/física, ya definidas en el paso de ruteo |
| Tiempos de viaje entre nodos | `bergen-boats/02_ruteo_navegable/output/matriz_tiempos_min.csv` | Ya calculados (Dijkstra sobre agua real), nunca se recalculan aquí |
| Geometría de las rutas (para animar) | `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson` | Igual — ya calculada, se persistió en este mismo trabajo para poder reusarla |
| Patrón de quién viaja a dónde y cuándo | `demand/src/llegadas.py` + `demand/output/matriz_intensidad_od.csv` | El generador de demanda real (gravedad + SSB + Poisson), no se toca |
| Paciencia (15/30 min ± jitter) | Ya viene calculada en la tabla de grupos que genera `demand/` | Columna `espera_maxima_min` |
| **Tolerancia (12 min), peso de pérdida (1.3), peso de movimiento (0.1)** | `simulacion/config/instance.yaml` → `recompensa` | Son propios de ESTE paso (la fórmula de recompensa) |
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

`demand/` genera **grupos** (ej. "3 personas que salen juntas de Kleppestø a las 6:23"). Este paso, por defecto, **explota** cada grupo en personas individuales (`src/unidades.py`, función `grupos_a_unidades`) — cada persona hereda el mismo origen/destino/hora de llegada/paciencia de su grupo, pero se trata como una solicitud independiente. Es un interruptor en el config (`unidad_demanda: "personas" | "grupos"`), no dos simuladores distintos.

### 4.3 Acciones (§5) — viajes directos punto a punto

Cada barco libre recibe una de 5 órdenes: ir a Kleppestø / Laksevåg / Bryggen / Sandviken, o esperar. Un barco en ruta ignora cualquier orden y sigue hasta llegar (no se redirige a media ruta, tal como pide la especificación).

**La política NO elige a quién recoge, elige el NODO destino.** Quién sube es una regla fija del simulador — pero, a diferencia de una primera versión, **las colas están separadas por par origen-destino** (12 colas, una por cada combinación, no 4 colas por nodo). Cuando un barco libre en A recibe la orden "ir a B", solo puede embarcar de la cola exacta A→B, en orden de llegada, hasta llenar el barco — nunca lleva gente con destinos mixtos, y al llegar a B baja a todos. Es un viaje directo punto a punto, no una ruta con paradas intermedias (ese modelo no existe todavía en este proyecto).

**¿Y si la gente que más necesita un barco está en OTRO nodo?** El barco se reposiciona vacío hacia ahí primero (la acción de este paso es solo "ir a ese nodo", sin embarcar nada), y recién en su siguiente momento libre — ya estando ahí — decide de nuevo con información fresca si los recoge. Es una decisión miope (no planea las dos etapas de una vez), consistente con que "nearest-available" es una regla simple, no un optimizador.

### 4.4 Recompensa (§7) — la fórmula exacta, con números reales

```
sobrante      = max(0, tiempo_en_el_sistema - 12)
sobrante_max  = max(1, paciencia - 12)      # el "1" es una corrección nuestra, ver sección 6
penalizacion  = (sobrante / sobrante_max) ** 2      # por cada persona activa (esperando O a bordo)

r = -[ Σ penalizacion + 1.3 × (personas perdidas este paso) + 0.1 × (barcos navegando) ]
```

Ejemplo real del log: un paso con 17 personas a bordo de un barco camino a Bryggen y 11 personas perdidas ese mismo paso da `r ≈ -37.8`. No es un número raro: son ~15-20 puntos de gente incómoda (cada una hasta 1.0) más `11 × 1.3 ≈ 14.3` de las perdidas, más el 0.1 por barco moviéndose. Todo suma.

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
2. **La fórmula de `sobrante_max` puede ser negativa.** La paciencia en conexiones fuertes es 15 min, pero con el ruido (`jitter`) que ya tenía `demand/` puede bajar a 10 min — por debajo de los 12 min de tolerancia de la fórmula. `sobrante_max = paciencia - 12` daría un número negativo (división sin sentido). Se le puso un piso (`sobrante_max_minimo: 1.0` en el config): para esos casos raros la penalización llega a su máximo casi de inmediato, que es razonable. No se tocó el generador de demanda.
3. **Doble despacho: dos barcos libres a la vez elegían el mismo destino.** Con la política aplicada de forma independiente por barco (cada uno mirando la misma foto del mundo, antes de que cualquiera de los dos suba gente), dos barcos libres en el mismo nodo y momento podían decidir ambos "ir a Bryggen" pensando que había 11 personas esperando ahí — el primero en procesarse se las llevaba todas, el segundo viajaba **vacío** hasta Bryggen, pagando la penalización de movimiento sin servir a nadie. Se detectó inspeccionando el minuto 400 de la corrida de ejemplo (`visualizacion.inspeccionar`) y se confirmó en el log de texto. Se corrigió con `asignar_flota` (sección 5): asigna los barcos libres de a uno, con una copia local de las colas que se va descontando. Efecto medido en la corrida de ejemplo: **el % de cumplimiento subió de 53% a 90%** (ver sección 8) — no era un detalle menor.

---

## 7. Los escalones (instancias de prueba)

| | Escalón 1 | Escalón 2 |
|---|---|---|
| Cuándo | Franja mañana (6-9h) | Día completo (6-24h) |
| Barcos | 2 | 3 |
| Grupos / personas | 23 / 202 | 133 / 1106 |
| Para qué | Verificar a ojo, log de texto | Métricas agregadas |

La demanda de cada escalón se generó con `demand/src/llegadas.py` **sin tocarlo** — solo se le pasó un `porcentaje_poblacion_dia` más bajo que el oficial (10%), calibrado por prueba y error hasta acercarse a "~20-30 grupos" (escalón 1) que pedía la especificación. `demand/config/instance.yaml` sigue intacto en 10%.

---

## 8. Corrida de ejemplo (escalón 1) y métricas

Ver `notebooks/01_escalon_1_verificacion.ipynb` para el log completo paso a paso, la verificación de conservación, y todas las tablas/gráficas.

**Verificación de conservación** (`metricas.verificar_conservacion`, sección 2 del plan de este paso): toda persona generada debe terminar contabilizada. Confirmado — 202 generadas = 181 atendidas + 21 perdidas + 0 esperando + 0 a bordo al final.

**Globales** (`metricas.metricas_globales`):

| Métrica | Valor |
|---|---|
| % cumplimiento global | 89.6% |
| Personas atendidas / perdidas | 181 / 21 |
| Espera media | 10.7 min |
| Tiempo en sistema medio / máximo | 19.5 / 34.0 min |

**Por par origen-destino, por barco, por usuario (percentiles), y dónde se pierde la gente** — tablas completas en el notebook (`metricas.metricas_por_par`, `metricas_por_barco`, `metricas_por_usuario`, `perdidas_por_nodo`), guardadas también en `output/metricas_por_par_escalon1.csv` y `output/metricas_por_barco_escalon1.csv`. Ejemplo notable: el par Bryggen→Kleppestø tuvo 0% de cumplimiento en esta corrida (11 generadas, 0 atendidas) — con solo 2 barcos, esa dirección quedó desatendida; es información real de la corrida, no un error.

**Gráficas** (`src/visualizacion.py`, `output/escalon1_*.png`): perfil temporal de personas esperando, ocupación de la flota en el tiempo, heatmap de % cumplimiento por par (4×4), desglose de recompensa en el tiempo, barras de pérdidas por par.

Reproducibilidad verificada: misma semilla de demanda → misma corrida exacta (recompensa total, atendidas y perdidas idénticas); semilla distinta → resultado distinto.

---

## 8.1 Visualización

`src/visualizacion.py` + celdas finales de `notebooks/01_escalon_1_verificacion.ipynb` → `output/escalon1_animacion.gif` (90 frames, uno por paso de 2 min). Reusa el mapa base y `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson` (persistidas en este mismo trabajo) para que los barcos se muevan bordeando tierra, no en línea recta. Colas visibles como contador junto a cada nodo, barcos como triángulos (gris = vacío, rojo = con gente, número = ocupación).

**Nota técnica:** el mapa de `bergen-boats/02_ruteo_navegable` usa CartoDB.Positron, pero al probarlo aquí devolvió una marca de agua "API KEY REQUIRED" (CartoDB cambió su política de acceso gratuito desde que se construyó ese paso). Se probó también OpenStreetMap, que bloqueó la solicitud por política de uso. Se usa **Esri.WorldGrayCanvas** en su lugar (sin llave, respondió limpio) — mismo principio (mapa real de fondo), proveedor distinto. El fondo se descarga **una sola vez** por corrida (no una vez por frame) para no depender de 90 llamadas de red seguidas.

## 8.2 Inspector de un minuto concreto

`visualizacion.inspeccionar(minuto, env, ...)` pausa en el paso más cercano al minuto pedido y muestra: posición/ocupación de cada barco, las 12 colas con personas esperando, qué decidió cada barco ese paso, y la recompensa con su desglose — en texto y dibujado sobre el mapa. Funciona siempre, en cualquier contexto (no requiere Jupyter con kernel vivo). Fue precisamente con esta función (parado en el minuto 400) que se encontró el bug #3 de la sección 6.

`visualizacion.inspeccionar_interactivo(env, ...)` agrega un control deslizante real (`ipywidgets`), pero **solo funciona si el notebook se abre con un kernel de Jupyter corriendo** (VS Code / Jupyter Lab) — un notebook ya ejecutado y guardado (como los que se generan con `nbconvert`) no tiene kernel vivo, así que el slider no se puede mover ahí.

## 9. Supuestos y limitaciones

- **El entorno en sí no tiene aleatoriedad propia.** Toda la aleatoriedad del pipeline vive en la generación de demanda (`demand/`, ya con su semilla). Una vez que la tabla de grupos está fija, el simulador es 100% determinista (misma entrada → mismo resultado siempre) — es una propiedad deseable para verificar, no un defecto.
- **Escala de demanda de los escalones (0.8% y 1.2% de la población):** valores propios de este paso, elegidos solo para que el tamaño de la corrida sea manejable de verificar — no tienen relación con el 10% "oficial" de `demand/`.
- **`sobrante_max_minimo` (piso de 1.0 min):** corrección nuestra a un caso límite no contemplado en la especificación (ver sección 6), no una cifra medida.
- **La recompensa puede escalar muy rápido** cuando personas de conexión fuerte (paciencia corta, 15 min) se acercan a su límite, porque `sobrante_max` es de solo ~3 minutos para ellas. Es una propiedad de los parámetros de la especificación (tolerancia 12 / paciencia 15), no un error de implementación — queda anotado para que el comité lo tenga presente al interpretar los números de recompensa.
- **Tiempo de espera reportado en métricas** = tiempo hasta que el destino de la persona quedó resuelto (subió a un barco, o se perdió) — no el tiempo de viaje a bordo después de subir. El tiempo en sistema sí suma ambos (espera + viaje) para las atendidas.
- **`asignar_flota` es miope, no óptima.** Coordina los barcos libres de UN mismo paso para que no se dupliquen entre sí (bug #3), pero sigue decidiendo de a uno, en el orden en que aparecen en `env.barcos` — no evalúa todas las combinaciones posibles para encontrar la asignación conjunta óptima. Es una mejora real sobre la versión sin coordinar, pero sigue siendo una regla simple, apropiada como línea base.
- **Reposicionamiento vacío (sección 4.3):** decisión de diseño propia, no está detallada explícitamente en la especificación — se dedujo como la forma más simple y consistente de manejar demanda fuera del nodo actual del barco bajo el modelo de viajes directos punto a punto.

---

## 10. Cómo correr

```bash
cd simulacion/notebooks
jupyter nbconvert --to notebook --execute --inplace 00_preparar_demanda_escalones.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_escalon_1_verificacion.ipynb
```

Necesita que `demand/output/` y `bergen-boats/02_ruteo_navegable/output/` ya existan (pasos previos, cerrados).

## Siguiente paso

Escalón 2 (día completo, 3 barcos, métricas agregadas a mayor escala) — pendiente. Después, cuando las métricas ya estén corriendo de forma estable, un informe LaTeX del proyecto completo (`docs/informe/`) para el asesor.
