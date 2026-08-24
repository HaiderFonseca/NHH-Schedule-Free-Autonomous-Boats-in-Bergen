# Demanda sintética anclada en datos abiertos (Bergen)

Genera llegadas de grupos de pasajeros, minuto a minuto, para los 4 nodos de demanda (Kleppestø, Laksevåg, Bryggen, Sandviken), ancladas en población y empleo reales de SSB en vez de supuestos a ojo. Alimenta después la simulación y un agente de RL.

**Principio rector: trazabilidad total.** Cada número de este README se puede rastrear hasta su origen: un archivo de `output/`, una celda de un notebook, o una cita explícita. Donde se asumió algo, está declarado como supuesto (sección final), no escondido en el código.

## Estado: paso cerrado

| Tarea | Estado | Qué cambió en la última iteración |
|---|---|---|
| 0 — Inspección y recorte | ✅ | — |
| 1 — Masas por nodo | ✅ | Grunnkretser que tocan cada área dibujada a mano, criterio "toca→completa" |
| 2 — Empleo (destino) | ✅ | `emp_tot` de la grilla SSB, sin necesitar StatBank; `factor_universitario` fijado por el equipo |
| 3 — Patrón O-D por franja | ✅ | Recalculado con masas actuales + factor universitario (dirección mañana: 85.6/14.4) |
| 4 — Llegadas minuto a minuto | ✅ | Escala como % de población real (10%, no número inventado); tamaño de grupo Binomial (no Normal) |

Los 4 nodos y su población/empleo final (Tarea 1 v2 + Tarea 2, ver detalle abajo):

| Nodo | Población | Empleo |
|---|---|---|
| Kleppestø (Askøy) | 25 153 | 7 550 |
| Laksevåg (Gravdal) | 15 986 | 7 760 |
| Bryggen / Sentrum | 27 695 | 52 213 |
| Sandviken (BSI Padling) | 11 109 | 4 106 |
| **Total** | **79 943** | **71 629** |

## Cómo está organizado

```
demand/
├── README.md                     # este archivo — documento único de las Tareas 0-4
├── config/instance.yaml          # única fuente de verdad de los parámetros
├── data/
│   ├── *.geojson                 # grillas SSB crudas (gitignored, ver "Procedencia de los datos")
│   ├── Areas_estudio.gpkg        # 4 áreas dibujadas a mano (QGIS), una por nodo
│   └── processed/                 # recortes y grunnkretser cacheados (GeoParquet)
├── src/
│   ├── masas.py                  # Tarea 0: recorte geográfico
│   ├── grunnkretser.py           # Tarea 1: grunnkretser de Kartverket, selección por área
│   ├── patron_od.py              # Tarea 3: intensidad O-D por franja
│   ├── llegadas.py               # Tarea 4: proceso de Poisson + grupos
│   └── plotting.py               # mapas de verificación
├── notebooks/
│   ├── 00_recorte_y_masas.ipynb  # Tarea 0: recorte geográfico (exploración inicial, no usada para masas)
│   ├── 00b_grunnkretser.ipynb    # exploración: descarga y mapea las grunnkretser candidatas
│   ├── 00c_masas_por_grunnkretser.ipynb  # Tarea 1 (canónica): masas con áreas reales
│   ├── 01_empleo.ipynb           # Tarea 2
│   ├── 02_patron_od.ipynb        # Tarea 3
│   └── 03_llegadas.ipynb         # Tarea 4
└── output/                       # CSV, mapas y gráficas — la evidencia de este README
```

## Procedencia de los datos

| Archivo | Qué es | Procedencia |
|---|---|---|
| `2026-08-18-befolkning_250m_2026.geojson` | Población en grilla de 250 m, todo Noruega | SSB — **[PENDIENTE: confirmar URL/nombre exacto del dataset con el usuario]** |
| `2026-08-18-bedrifter_250m_2026.geojson` | Establecimientos y empleo en grilla de 250 m, todo Noruega | SSB — **[PENDIENTE: confirmar URL/nombre exacto del dataset con el usuario]** |
| Grunnkretser (`data/processed/grunnkretser_norge.parquet`, cacheado) | Límites de las unidades estadísticas pequeñas de Noruega | Kartverket, WFS "Statistiske enheter grunnkretser" — [kartkatalog.geonorge.no, uuid `cc7ded0b-7d34-4db6-8fdb-c5a7682b6836`](https://kartkatalog.geonorge.no/metadata/statistiske-enheter-grunnkretser/cc7ded0b-7d34-4db6-8fdb-c5a7682b6836), capa `Grunnkrets`, licencia CC BY 4.0 |
| `Areas_estudio.gpkg` | 4 polígonos dibujados a mano (QGIS) delimitando la zona de captación de cada nodo | Elaboración propia del equipo, sin atributos — el nodo de cada área se infiere por cercanía (ver Tarea 1) |

No se pudo verificar la cita exacta de los 2 GeoJSON de SSB más allá de lo que el propio archivo permite inspeccionar (ver Tarea 0). La validación de plausibilidad (suma nacional de población ≈ población real de Noruega, ver Tarea 0) da confianza en que son datos SSB genuinos, pero la cita formal queda pendiente. La fuente de grunnkretser sí quedó completamente identificada (WFS oficial de Kartverket, URL exacta arriba).

---

## Tarea 0 — Inspección y recorte geográfico

**Notebook:** `notebooks/00_recorte_y_masas.ipynb`

### Qué se encontró en los archivos crudos

| | Población (`befolkning_250m`) | Bedrifter (`bedrifter_250m`) |
|---|---|---|
| CRS | **EPSG:32633** (UTM 33N) — métrico, correcto para radios en metros | **EPSG:32633** |
| Columnas | `ssbid250m` (id), **`pop_tot`** (población de la celda) | `SSBID250M` (id), `est_tot` (nº establecimientos), **`emp_tot`** (empleo real) |
| Resolución de celda | 250 × 250 m (verificado en código, no solo en el nombre del archivo) | 250 × 250 m |
| Nº de features (todo Noruega) | 225 238 | 133 688 |
| Cobertura | Todo Noruega, incluye Askøy (verificado explícitamente) | Todo Noruega |
| Validación de plausibilidad | Suma nacional de `pop_tot` = **5 617 894** ≈ población real de Noruega | Máx. de una celda = 10 628 empleos (plausible: un empleador grande concentrado) |

**Hallazgo importante que cambió el plan (Tarea 2):** `bedrifter_250m` trae `emp_tot` (empleo real), no solo conteo de establecimientos como se asumía al planear. Ver Tarea 2.

### Bounding box de recorte

Calculado como el rectángulo que cubre los 4 nodos (leídos de `bergen-boats/config/instance.yaml`) más un margen de 8000 m (`recorte.margen_bbox_m`):

```
EPSG:32633: xmin=-45041.1, ymin=6726620.1, xmax=-24044.2, ymax=6745676.7
Ancho: 21.00 km, Alto: 19.06 km
```

### Celdas tras el recorte

| | Todo Noruega | Tras recorte a Bergen+Askøy |
|---|---|---|
| Población | 225 238 celdas | **2 557 celdas** (260 168 habitantes) |
| Bedrifter | 133 688 celdas | **2 127 celdas** (160 895 empleos) |

Recorte cacheado en `data/processed/poblacion_bergen_askoy.parquet` y `bedrifter_bergen_askoy.parquet` — no se repite en corridas futuras.

---

## Tarea 1 — Masas por nodo

**Notebooks:** `notebooks/00b_grunnkretser.ipynb` (exploración) → `notebooks/00c_masas_por_grunnkretser.ipynb` (cálculo final) · **Output:** `output/masas_por_nodo.csv`, `output/grunnkretser_seleccionadas_por_nodo.csv`, `output/mapas_zonas_grunnkretser_por_nodo.png`, `output/mapa_general_zonas_grunnkretser.png`

**Fuente de los límites administrativos:** grunnkretser (la unidad estadística pequeña real de Noruega, ~403 en Bergen+Askøy), descargadas del WFS oficial de Kartverket (`wfs.grunnkretser`, capa `Grunnkrets`, ver "Procedencia de los datos"). Confirmado por código: Bergen = kommunenummer 4601, Askøy = 4627 (verificado buscando "Kleppestø" en los nombres de grunnkrets).

**Proceso:**
1. El equipo dibujó a mano 4 áreas de estudio en QGIS (`data/Areas_estudio.gpkg`, sin traslapes entre sí — verificado, intersección = 0 km² en los 6 pares).
2. Cada área se asignó al nodo cuyo centro está más cerca — sin ambigüedad: la distancia al nodo correcto es 3.9× a 10.7× menor que al segundo más cercano (verificado explícitamente en el notebook, no solo asumido).
3. Se seleccionaron las grunnkretser que **tocan** cada área (criterio "intersecta", no "centro adentro": si el área agarra aunque sea un pedacito de una grunnkrets, esa grunnkrets se toma **completa**). Decisión explícita del usuario: dibujó las áreas pensando en esto, y con el criterio anterior ("centro adentro") se estaban quedando afuera zonas costeras de Sandviken que el área sí tocaba.
4. **Resolución de duplicados:** con "intersecta", una grunnkrets justo en el borde entre dos áreas puede tocar a las dos. Se encontró **1 caso** ("Sandviksfjellet", 46010637, tocaba tanto Bryggen como Sandviken) y se resolvió dejándola solo en el nodo con mayor superposición real (Sandviken, 18.4% de su área vs. 0.05% en Bryggen) — verificado y reportado explícitamente en el notebook, no solo corregido en silencio.
5. Población y empleo se recalcularon sobre la unión de esas grunnkretser, usando la misma grilla SSB de 250 m de siempre — cambia la forma de la zona, no la fuente de los números de población/empleo.

**Grunnkretser seleccionadas y masas resultantes** (`output/grunnkretser_seleccionadas_por_nodo.csv` tiene el detalle completo, número y nombre de cada una):

| Nodo | Grunnkretser | Área (km²) | Población total | Empleo total |
|---|---|---|---|---|
| Kleppestø (Askøy) | 28 | 80.59 | **25 153** | **7 550** |
| Laksevåg (Gravdal) | 18 | 14.85 | **15 986** | **7 760** |
| Bryggen / Sentrum | 67 | 20.21 | **27 695** | **52 213** |
| Sandviken (BSI Padling) | 21 | 14.61 | **11 109** | **4 106** |

El empleo de Bryggen es el más alto con claridad (~73% del total de los 4 nodos), consistente con ser el polo de empleo real de la ciudad. Kleppestø es el nodo con más población (25 153 hab.) al cubrir buena parte del municipio de Askøy.

**Mapas de verificación:** `output/mapa_general_zonas_grunnkretser.png` (las 4 zonas juntas, con el área dibujada en línea punteada y las grunnkretser seleccionadas rellenas) y `output/mapas_zonas_grunnkretser_por_nodo.png` (zoom por nodo) — confirman que ya no quedan huecos costeros dentro de las áreas dibujadas.

---

## Tarea 2 — Empleo como masa de destino

**Notebook:** `notebooks/01_empleo.ipynb` · **Resuelta sin necesitar la API de StatBank.**

El plan original asumía que había que traer el empleo por API porque `bedrifter_250m` sería "solo conteo de establecimientos". Al inspeccionar el archivo (Tarea 0) apareció la columna `emp_tot` con empleo real. Antes de usarla se verificó la alternativa planeada:

- Se exploró `https://data.ssb.no/api/v0/no/table/al/al06/regsys/SBMENU4262` → tabla **12850** ("Sysselsatte, etter bosted, arbeidssted, næring..."). Su variable `Region` solo baja a **fylke/kommune** (Vestland, Bergen, etc.), no a grunnkrets/delområde — SSB no publica empleo por lugar de trabajo a grano fino vía StatBank por reglas de privacidad de desagregación; en su lugar lo publica como grilla, que es justo lo que ya se tenía.
- **Conclusión:** `emp_tot` de la grilla de 250 m es la fuente correcta, y de hecho más fina que grunnkrets.

**Evidencia de que `emp_tot` es empleo real y no ruido de `est_tot`:** en la zona de estudio (2127 celdas), correlación `est_tot` vs. `emp_tot` = **0.581** (positiva pero lejos de 1 — consistente con empleo real: una oficina grande concentra mucho empleo en pocos establecimientos). Ver `output/establecimientos_vs_empleo.png`.

Números actualizados con las zonas de grunnkretser de la Tarea 1 (ver arriba):

| Nodo | Establecimientos | Empleo | Empleados/establecimiento |
|---|---|---|---|
| Kleppestø | 1 884 | 7 550 | 4.0 |
| Laksevåg (Gravdal) | 1 447 | 7 760 | 5.4 |
| Bryggen / Sentrum | 7 101 | 52 213 | 7.4 |
| Sandviken (BSI Padling) | 879 | 4 106 | 4.7 |

`est_tot` se guarda solo como referencia comparativa, no se usa como masa principal.

**Sobre la población universitaria (actualizado):** Bergen es ciudad universitaria (NHH, UiB, HVL) y `emp_tot` no captura estudiantes, que también generan viajes. No se consiguió una fuente abierta de matrícula/población estudiantil por zona geográfica fina, así que se dejó un gancho en el config: `masas.factor_universitario`, que pondera la masa de **destino** (empleo) de cada nodo en la Tarea 3. El equipo fijó los valores a mano: `kleppesto: 0.7`, `laksevag: 0.85`, `bryggen: 1.0`, `sandviken: 1.0` (Sandviken queda junto a NHH/Kronstad/HVL, y de hecho el propio Gu & Wallace (2021) — el paper que originó estos mismos 4 nodos — describe explícitamente "Sandviken & Hegreneset" como zona con "office buildings, apartments, houses **and a university**"). Es una decisión de diseño del equipo, no una fuente de datos — declarada explícitamente en "Supuestos y limitaciones".

---

## Tarea 3 — Patrón origen-destino por franja horaria

**Notebook:** `notebooks/02_patron_od.ipynb` · **Output:** `output/matriz_intensidad_od.csv`, `output/heatmaps_od_por_franja.png`, `output/inversion_direccion_bryggen.png`, `output/factor_distancia.png`

### Método: Braathen, Goez & Guajardo (2024) §4.1, adaptado

El paper (`papers/Autonomous ferries in light of labor regulations-A passenger.pdf`) define:

```
pop_pg := ln(1 + Π(i=1..4) p_i,pg)          p1..p4 = origen, destino, día, hora (puntajes de estación "a criterio")
tamaño_pg := ceil((pop_pg − min_j pop_j) / β)
```

con **media 8.6 pasajeros/grupo, desviación 2.6, moda 9** (su Fig. 1-2, pág. 8).

**Qué es préstamo y qué es aporte propio — explícito:**

| Elemento | Origen |
|---|---|
| Fórmula multiplicativa | Braathen et al. §4.1 |
| Los 4 factores (origen, destino, día, hora) | Braathen et al. §4.1 |
| Puntaje de estación = **el mismo** para origen y destino | Braathen et al. §4.1 — **no lo copiamos tal cual** |
| Puntajes de estación puestos "a criterio", sin tabla publicada | Braathen et al. — no reproducible por diseño del paper |
| Sustituir puntajes de estación por masas reales SSB (Tareas 0-2) | **Aporte propio** |
| Separar el puntaje por rol — población si origen, empleo si destino, con inversión AM/PM | **Aporte propio**, no está en el paper |
| Factor de distancia decreciente con el tiempo de viaje | **Aporte propio** — el paper no pondera así en §4.1 (usa distancia en otro problema, filtrar "rutas realistas") |
| Tamaño de grupo como distribución configurable (vs. fórmula determinística del paper) | **Aporte propio**, calibrado para igualar la media/desviación *reportada* |
| Factores de franja horaria y día de semana | Con criterio propio, igual que el paper (no reproducible del paper tampoco) |

### Normalización de masas

Población y empleo se normalizan dividiendo por el máximo entre los 4 nodos (quedan en `(0, 1]`) antes de multiplicarse — si no, la escala cruda (cientos vs. miles) dominaría el producto arbitrariamente. Ver tabla completa en el notebook.

### Franjas horarias configuradas

| Franja | Horas | Factor volumen | Rol origen | Rol destino |
|---|---|---|---|---|
| Mañana | 6-9 | 1.0 | población | empleo |
| Valle | 9-15 | 0.4 | mixta | mixta |
| Tarde | 15-18 | 1.0 | empleo | población |
| Noche | 18-24 | 0.15 | mixta | mixta |

`factor_distancia = exp(-tiempo_viaje_min / tau_min)`, con `tau_min = 15` — forma y parámetro elegidos por razonabilidad (ver "Supuestos").

### ¿Los roles cambian entre semana y fin de semana? No — solo el volumen

Pregunta que vale la pena responder explícitamente: **la tabla de roles de arriba (mañana=población→empleo, tarde=empleo→población, valle/noche=mixta) es la misma los 7 días de la semana.** `factor_dia_semana` (Tarea 4: `entre_semana=1.0`, `fin_de_semana=0.4`) es un multiplicador **global** sobre el total de pasajeros del día — no toca `matriz_intensidad_od.csv`, que se calcula una sola vez y no distingue día de semana. Es decir: sábado y domingo tienen exactamente el mismo patrón direccional que un día laboral, solo que a ~40% del volumen (más el ruido propio de Poisson).

Verificado con los datos generados (`grupos_semana.csv`, % de grupos por nodo de origen dentro de cada franja):

| Franja | Origen dominante entre semana | Origen dominante fin de semana |
|---|---|---|
| Mañana | Kleppestø 39% | Kleppestø 38% |
| Tarde | Bryggen 73% | Bryggen 73% |
| Noche | Bryggen 36% | Bryggen 44% |

Prácticamente idénticos entre semana/fin de semana (la franja "noche" varía un poco más por tener menos grupos totales, más sensible al ruido de Poisson). Es una simplificación declarada: en la realidad el patrón de un sábado (ocio, turismo) probablemente no es solo "la misma commute de siempre pero más floja" — pero el modelo actual no lo distingue. Ver "Supuestos y limitaciones".

### Verificación clave: ¿aparece la dirección dominante sola?

`docs/parametros_instancia_base_bergen.md` había supuesto **90/10** hacia Bryggen en la mañana, a ojo, sin datos. Aquí **no se fijó ese número** — se dejó que emergiera del modelo (población en barrios + empleo en Bryggen). Resultado real:

| Franja | % intensidad hacia Bryggen | % intensidad desde Bryggen |
|---|---|---|
| Mañana | **85.6%** | 14.4% |
| Valle | 50.0% | 50.0% (por diseño: rol "mixta" simétrico) |
| Tarde | 14.4% | **85.6%** (inversión perfecta respecto a la mañana) |
| Noche | 50.0% | 50.0% |

Con las masas por grunnkretser (Tarea 1) y el `factor_universitario` fijado por el equipo (que reduce a Kleppestø/Laksevåg como destino), salió **85.6/14.4** — cerca del 90/10 supuesto originalmente sin datos. Sigue siendo consistente (inversión mañana↔tarde exacta, por construcción). Ver `output/inversion_direccion_bryggen.png`.

---

## Tarea 4 — Llegadas minuto a minuto (día y semana)

**Notebook:** `notebooks/03_llegadas.ipynb` · **Output:** `output/grupos_dia.csv`, `output/grupos_semana.csv`

Por cada `(origen, destino, hora)`, la intensidad de la Tarea 3 se convierte en una tasa esperada de **grupos** por hora (no personas sueltas — agrupadas desde esta primera iteración), usando la escala de pasajeros/día (ver abajo) y el tamaño medio de grupo. Dentro de cada hora: `N ~ Poisson(λ_hora)` grupos, cada uno con minuto uniforme dentro de la hora.

### Escala de demanda: porcentaje de población real, no un número inventado

La primera versión usaba `demanda.escala_global_viajes_dia = 2000` (pasajeros/día), un número puesto a mano sin respaldo. Se reemplazó por `demanda.porcentaje_poblacion_dia`, aplicado sobre la población real de las 4 zonas (`poblacion_total` de `masas_por_nodo.csv`, Tarea 1 v2 = **79 943 habitantes**). Ninguno de los dos papers de referencia reporta un porcentaje de participación modal directamente, así que se dedujo cruzando sus escalas de demanda contra población:

| Referencia | Escala reportada | Traducida a % de población |
|---|---|---|
| **Gu & Wallace (2021)** — el paper original de water-taxis en Bergen, **usa estos mismos 4 nodos** (Kleppestø, Laksevåg, Bryggen, Sandviken & Hegreneset) | 300 "demandas"/día (su Tabla 2: 76+74+74+76) × 8.6 pax/grupo (Braathen) = 2 580 pax/día | **3.2%** de la población de las 4 zonas (79 943 hab.) |
| **Braathen, Goez & Guajardo (2024)** — instancia grande, red hipotética de 6 paradas citywide | 85 920 pax/semana = 12 274 pax/día | **4.3%** de la población total de Bergen (~285 900 hab.) |

Ambas referencias convergen en 3-4%. El equipo, sin embargo, fijó el parámetro en **`porcentaje_poblacion_dia: 0.10`** (10%) — deliberadamente por encima de ese rango, como un escenario de mayor adopción/demanda, no como una lectura directa de las dos referencias. Queda declarado explícitamente como decisión del equipo (ver "Supuestos y limitaciones"), no una cifra que salga de la literatura. Con esto, la escala entre semana sale **7 994 pasajeros/día** (`79 943 × 0.10`), calculada dinámicamente a partir de la población — si se rehace el trazado de las áreas de la Tarea 1, la escala se recalcula sola en la misma proporción, no queda un número suelto desincronizado.

### Tamaño de grupo: Binomial, no Normal

Braathen et al. (2024) reportan media 8.6 pax/grupo y desviación 2.6 (su Fig. 1-2) pero **no dicen qué distribución usar** — en el paper el tamaño es determinístico (función del puntaje de popularidad), no viene de muestrear una distribución. La primera versión de este paso usaba `Normal(8.6, 2.6)` redondeada, que no es apropiada: es continua y simétrica, pensada para cantidades continuas, no para conteos de personas (discretos, no negativos).

Se reemplazó por **Binomial(n=40, p=0.215)**, calibrada por método de momentos para igualar la media y desviación de Braathen:

```
media = n·p = 8.60          (objetivo: 8.6)
desviación = √(n·p·(1-p)) = 2.598   (objetivo: 2.6)
P(grupo de 0 personas) = 0.006%    (Normal truncada: ~0.09% -- 15x más frecuente)
```

Además de ser discreta por construcción (nunca hay que redondear ni truncar valores negativos), es más realista: puede interpretarse como que hay un grupo "candidato" de hasta 40 personas cercanas en tiempo/lugar, cada una decidiendo independientemente con probabilidad 21.5% sumarse a ese viaje — un proceso de formación de grupo más plausible que una campana continua.

- **Espera máxima:** 15 min en conexiones fuertes (reusa `bergen-boats/config/instance.yaml → garantia.conexiones_fuertes`: Bryggen↔Kleppestø/Laksevåg/Sandviken), 30 min en el resto, con jitter ±5 min.
- **Semana:** `numpy.random.SeedSequence(42).spawn(7)` → 7 semillas hijas, una por día — mismo patrón de fondo, llegadas distintas y reproducibles.

### Resultado, semana generada (semilla 42)

| Día | Grupos | Pasajeros | Fin de semana |
|---|---|---|---|
| 0 (lun) | 940 | 8 124 | No |
| 1 (mar) | 953 | 8 385 | No |
| 2 (mié) | 963 | 8 097 | No |
| 3 (jue) | 932 | 8 120 | No |
| 4 (vie) | 929 | 8 137 | No |
| 5 (sáb) | 372 | 3 206 | Sí |
| 6 (dom) | 369 | 3 108 | Sí |

Entre semana ronda los ~8 100 pasajeros/día derivados (10% de 79 943 hab.); fin de semana cae a ~40% (`factor_dia_semana.fin_de_semana = 0.4`, más la aleatoriedad de Poisson) — mismo patrón direccional, solo cambia el volumen (ver Tarea 3). Ver `output/pasajeros_por_dia_semana.png`.

**Tamaño de grupo observado (semana completa):** media 8.64, desviación 2.64 — contra el objetivo Binomial(40,0.215) de 8.60/2.598, y bien por debajo de la capacidad del barco (20 pasajeros, `bergen-boats/config/instance.yaml`). Ver `output/histograma_tamano_grupo.png`.

**Reproducibilidad verificada explícitamente:** misma semilla → tabla idéntica (`.equals()` en el notebook, `True`). Semilla distinta → resultado distinto. Ver notebook, sección de verificación.

**Perfil de demanda por hora** (`output/perfil_demanda_hora.png`): pico mañana (6-9h) y pico tarde (15-17h) claramente visibles, valle y noche más bajos — reproduce la forma esperada sin haberla forzado directamente en la generación de llegadas (viene de las franjas de la Tarea 3).

**Heatmap O-D de las llegadas generadas** (`output/heatmap_od_llegadas_generadas.png`): Kleppestø↔Bryggen es el par dominante (7 249 + 7 379 = 14 628 pasajeros/semana) — consistente con que Kleppestø tiene la mayor población de los 4 nodos (25 153 hab.) y Bryggen el mayor empleo (52 213), reforzado por el `factor_universitario` que concentra la atracción de destino en Bryggen/Sandviken. Bryggen↔Laksevåg (6 079+6 264 = 12 343) y Bryggen↔Sandviken (4 992+4 948 = 9 940) le siguen.

### Columnas de la tabla de grupos

`grupo_id, dia, franja, hora, minuto_dia, origen, destino, tamano_grupo, espera_maxima_min, conexion_fuerte, fin_de_semana`

---

## Supuestos y limitaciones

Lista honesta de lo que se asumió en todo el paso (Tareas 0-4), para que el comité pueda juzgar qué tan sólido es cada número. Se ordena de mayor a menor impacto en los resultados finales.

- **Trazado de las 4 áreas de estudio (`Areas_estudio.gpkg`):** es la decisión con más apalancamiento de todo el paso — define población, empleo, y en cascada la escala de demanda (ver abajo). Es un juicio del equipo mirando el mapa, no un dato medido; otro trazado habría dado otra población/empleo por nodo y, por lo tanto, otra escala de demanda total (al ser esta última un porcentaje de esa población). Recomendación para una siguiente iteración: sensibilizar los resultados a 2-3 trazados alternativos razonables.
- **Criterio de selección de grunnkretser ("toca → se toma completa", Tarea 1 v2):** más generoso que "centro adentro" — puede incorporar grunnkretser grandes y poco pobladas (montaña, bosque) que solo rozan el borde dibujado, como pasó con Bryggen (su área subió de 4.0 a 20.2 km² sin que la población/empleo subiera proporcionalmente). Infla el área, no distorsiona mucho la población/empleo, pero es una propiedad del método que hay que tener presente.
- **`factor_universitario` (kleppesto=0.7, laksevag=0.85, bryggen=sandviken=1.0):** decisión de diseño del equipo, no una fuente de datos de matrícula real (que sigue sin conseguirse). Tiene impacto directo en la dirección mañana/tarde (ver Tarea 3: 85.6/14.4). Gu & Wallace (2021) sí menciona una universidad junto a Sandviken, lo que da cierto respaldo cualitativo a por qué Sandviken no se penaliza, pero los valores numéricos exactos son del equipo, no de una fuente.
- **Escala de demanda (`porcentaje_poblacion_dia = 10%`):** ya no es un número inventado sin ancla — es un porcentaje de población real, y las dos referencias cruzadas (Gu & Wallace ≈3.2%, Braathen et al. ≈4.3%, ver Tarea 4) dan un rango de 3-4%. El equipo eligió **10%**, deliberadamente por encima de ese rango (un escenario de mayor adopción) — es una decisión explícita del equipo, no una lectura de la literatura, y **hereda cualquier error del trazado de las áreas** (si la población de las zonas cambia, la escala cambia en la misma proporción).
- **Tamaño de grupo (Binomial(40, 0.215)):** discreta y calibrada por método de momentos para igualar la media/desviación *reportada* por Braathen et al. (8.6/2.6) — no su fórmula determinística real (que depende de una escala `β` propia de sus datos de estación, no transferible directo a masas SSB). Es una mejora sobre la `Normal` truncada de la primera versión, pero sigue siendo una elección de distribución del equipo, no medida de datos reales de Bergen (que no existen porque el servicio no existe).
- **Normalización de masas (dividir por el máximo entre los 4 nodos):** decisión de escala propia, no viene del paper ni de SSB. Otra normalización (por la suma total, o log) daría una intensidad relativa distinta.
- **Roles "mixta" en franjas valle/noche:** promedio simple de población y empleo normalizados — simplificación; el propósito de viaje fuera de pico (mandados, ocio, salud) no necesariamente sigue esa mezcla.
- **Fin de semana = mismo patrón direccional, menos volumen:** `factor_dia_semana` solo escala el total de pasajeros (Tarea 4); no cambia los roles por franja de la Tarea 3. Un sábado tiene la misma commute mañana→Bryggen/tarde→casa que un lunes, solo que más floja — no un patrón propio de fin de semana (ocio, turismo). Ver Tarea 3, verificación explícita con los datos generados.
- **Factor de distancia (`exp(-tiempo/tau)`, tau=15 min):** forma y parámetro elegidos por razonabilidad dentro del rango de tiempos de la instancia (5-11 min navegables), no calibrados con datos de elasticidad real de demanda vs. tiempo de viaje.
- **Factores de franja y día de semana** (volumen relativo, entre semana vs. fin de semana): con criterio propio, igual que Braathen et al. — no vienen de un conteo real de pasajeros de Bergen.
- **Espera máxima por grupo:** valores de diseño (15/30 min + jitter), no medidos.
- **Procedencia exacta de los 2 GeoJSON de SSB:** pendiente de confirmar la URL/nombre exacto del dataset (ver "Procedencia de los datos" arriba) — la plausibilidad de los números (suma nacional de población ≈ población real de Noruega) da confianza, pero la cita formal falta. La fuente de grunnkretser sí quedó completamente identificada.

## Referencias

- **Gu, Y. & Wallace, S.W. (2021).** "Operational benefits of autonomous vessels in logistics — A case of autonomous water-taxis in Bergen." *Transportation Research Part E* 154, 102456. `papers/Operational benefits of autonomous vessels in logistics-A case of.pdf`. **Usa los mismos 4 nodos de este proyecto** (Kleppestø, Laksevåg, Bryggen, Sandviken & Hegreneset) — es la referencia más comparable para la escala de demanda (Tarea 4) y confirma la universidad junto a Sandviken (Tarea 2/3).
- **Braathen, C., Goez, J.C. & Guajardo, M. (2024).** "Autonomous ferries in light of labor regulations — A passenger perspective." *Maritime Transport Research* 7, 100115. `papers/Autonomous ferries in light of labor regulations-A passenger.pdf`. Fuente del método multiplicativo de generación de grupos (§4.1, Tarea 3) y de las estadísticas de tamaño de grupo (Tarea 4).

## Reproducir

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00b_grunnkretser.ipynb             # exploracion de zonas
jupyter nbconvert --to notebook --execute --inplace notebooks/00c_masas_por_grunnkretser.ipynb   # Tarea 1
jupyter nbconvert --to notebook --execute --inplace notebooks/01_empleo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_patron_od.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_llegadas.ipynb
```

En orden — `02` y `03` dependen del `output/masas_por_nodo.csv` que deja `00c`. En Windows, los notebooks que usan el WFS de grunnkretser (`00b`, `00c`) necesitan `PYTHONUTF8=1` en el entorno (nombres con å/æ/ø rompen GDAL si no). Todos los que descargan mapas necesitan conexión a internet.
