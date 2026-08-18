# Paso 2 — Ruteo navegable (corrige las líneas rectas que cruzan tierra)

El paso 1 (`../01_tiempos_distancias/`) usaba Haversine: distancia en línea recta. Al revisarla visualmente encontramos que las líneas que tocan **Bryggen** cruzan tierra — Bryggen está metida en la bahía de Vågen, junto a la **península de Nordnes**. Este paso lo corrige con un **módulo de ruteo sobre agua**: construye una malla navegable real y calcula el camino más corto que no pasa por tierra.

## Cómo funciona, paso a paso

1. **Descargar un mapa real** de la zona (CartoDB Positron sin etiquetas — sin etiquetas para que el texto de nombres no se clasifique por error como tierra).
2. **Clasificar agua/tierra por color de píxel.**
3. **Construir un grafo**: cada píxel de agua es un nodo conectado a sus 8 vecinos.
4. **Enganchar (snap)** cada puerto al píxel de agua más cercano.
5. **Dijkstra** desde cada puerto hacia los demás → distancia navegable real y el camino exacto.
6. **Aplicar la velocidad de diseño fija** (30 km/h, ver más abajo) para convertir distancia a tiempo.

Toda la lógica reutilizable está en [`../src/water_routing.py`](../src/water_routing.py), documentada para usarse en pasos futuros (p.ej. si se necesita rutear un barco real durante la simulación, no solo calcular una matriz).

---

## Cómo se construye la máscara agua/tierra, en detalle

**El mapa es un mosaico de tiles XYZ** (el mismo formato que usan Google Maps y OpenStreetMap): para cada nivel de zoom, el mundo se divide en una grilla de tiles de 256×256 píxeles, y cada nivel duplica la resolución del anterior. Con `ZOOM=14` eso da una resolución **fija en todo el planeta** de:

```
metros_mercator_por_píxel = 156543.03392 / 2^zoom = 156543.03392 / 16384 ≈ 9.55 m/píxel
```

Ese valor está en metros **proyectados** (Web Mercator, EPSG:3857), que no son metros reales: Mercator estira las distancias por un factor `1/cos(latitud)` para poder representar la Tierra esférica en un plano (por eso Groenlandia se ve gigante en un mapamundi). Bergen está a ≈60.4°N, donde `cos(60.4°) ≈ 0.494` — así que un metro proyectado ahí equivale a solo ~0.494 metros reales. La resolución real en el suelo es:

```
metros_reales_por_píxel = 9.55 × cos(60.4°) ≈ 4.7 m/píxel
```

Es decir: **cada "cuadrito" de la malla es un cuadrado de ~4.7 × 4.7 m de agua o tierra real.** El notebook calcula este número explícitamente (sección 2) en vez de dejarlo fijo, porque depende de la latitud exacta de la instancia.

**El grafo se construye directamente sobre estos píxeles — no hay una distancia de grilla elegida aparte.** Un nodo del grafo *es* un píxel de agua, así que el espaciado entre dos nodos vecinos es exactamente esa resolución: **~4.72 m** para un vecino ortogonal (arriba/abajo/izquierda/derecha) y **~6.67 m** (`4.72 × √2`) para un vecino diagonal, con `ZOOM=14`. El notebook lo imprime explícitamente en la sección 3, al construir el grafo. Subir el `ZOOM` da una malla más fina (más nodos, más precisión, más lento); bajarlo da una malla más gruesa.

**Clasificación por color:** se descargó un tile de prueba de Bergen y se contaron los colores de píxel más frecuentes. El agua en CartoDB Positron resultó tener un color muy consistente — `COLOR_AGUA_POSITRON = (212, 218, 220)` (gris-azulado claro) — que aparecía en ~40% de los píxeles del área de prueba, consistente con ser el color del fiordo. Cada píxel se clasifica como agua si su distancia euclidiana en espacio RGB a ese color de referencia es menor a un umbral (`umbral_color=15`); si no, es tierra.

**Por qué la máscara queda tan pegada a la costa real** (ver `output/mapa_mascara_agua.png`, donde se superpone en azul semitransparente sobre el mapa real): porque es *literalmente* el mismo raster que dibuja el mapa. La conversión píxel↔lon/lat (`wr.pixel_a_lonlat` / `wr.lonlat_a_pixel`) usa la misma fórmula de Web Mercator esférico, con el mismo radio (6378137 m), que usó el servidor de tiles para renderizar la imagen. Al superponer la máscara clasificada de vuelta sobre el mapa original, coincide píxel a píxel por construcción — la precisión de la costa depende de qué tan buena sea la geometría de OpenStreetMap (la fuente de datos de CartoDB), no de ninguna aproximación nuestra.

*(Nota técnica que costó un bug: Web Mercator usa el radio ecuatorial de WGS84, 6378137 m, no el radio medio terrestre 6371009 m que se usa en Haversine para distancias reales. Usar el radio equivocado en la fórmula de proyección desalinea todo el grid — nos pasó en la primera versión y quedaba a ~150 km de donde debía.)*

## Cómo funciona Dijkstra aquí, en detalle

1. **El grafo**: cada píxel de agua (~1 millón en el bbox de esta instancia) es un nodo, conectado a sus 8 vecinos (arriba, abajo, izquierda, derecha y las 4 diagonales) *solo si ese vecino también es agua*. Si un vecino es tierra, no existe esa arista — así que el grafo **no tiene forma de ofrecer un salto que cruce tierra**, ni por accidente. El peso de cada arista es la distancia real en km entre los centros de los dos píxeles (con la corrección `cos(lat)` aplicada fila por fila).
2. **Snap**: las coordenadas reales de un puerto casi nunca caen justo en un píxel de agua (pueden caer en el borde de un muelle). `wr.snap_a_grafo()` busca el píxel de agua más cercano en espiral creciente, **restringido a la componente conexa principal** del grafo (calculada con `scipy.sparse.csgraph.connected_components`). Esto evita enganchar un puerto a un charco o estanque aislado que el color clasificó como agua pero que no está conectado al mar abierto — nos pasó con Hegreneset en una primera versión, antes de agregar esta restricción.
3. **Dijkstra multi-fuente**: `scipy.sparse.csgraph.dijkstra(grafo, indices=[...], return_predecessors=True)` corre el algoritmo de Dijkstra una vez por cada nodo/puerto de origen, todo en una sola llamada. Como todos los pesos son distancias reales (siempre positivos), Dijkstra garantiza encontrar el camino de **menor distancia total** desde cada origen hacia todos los demás nodos del grafo. No hace falta A* ni ninguna heurística: con ~1 millón de nodos y ~8 millones de aristas dirigidas, Dijkstra puro (implementado en Cython dentro de scipy) corre en ~1 segundo para los 5 puertos a la vez.
4. **Reconstrucción de la ruta**: Dijkstra también devuelve `predecessors`, el nodo anterior en el camino más corto hacia cada destino. `wr.reconstruir_ruta_latlon()` camina esa cadena hacia atrás (destino → origen), junta los píxeles visitados y los convierte de vuelta a lon/lat — así se dibuja la ruta real en los mapas, no solo se reporta un número.

**Limitación conocida:** al permitir solo 8 direcciones (no 360°), el camino más corto puede hacer un ligero "zigzag" en vez de una diagonal perfecta cuando la dirección real no coincide con ninguna de las 8 permitidas (se nota como un pequeño quiebre en algunas rutas de los mapas). El sobrecosto de esto es pequeño (unos pocos % en el peor caso) y no afecta la conclusión de fondo: evita tierra con certeza.

---

## Nodos de esta versión

Laksevåg y Sandviken se corrigieron a ubicaciones confirmadas (las anteriores eran aproximaciones):

| Nodo | Antes (aprox.) | Ahora (confirmado) |
|---|---|---|
| Laksevåg | 60.3945, 5.2875 | **60.390886, 5.259586** (Gravdal) |
| Sandviken | 60.4075, 5.3214 | **60.421149, 5.300502** (BSI Padling) |

Kleppestø y Bryggen no cambiaron.

## Velocidad: decisión de diseño fija (30 km/h)

En vez de calibrar con el tramo real Kleppestø–Bryggen (como en una versión anterior de este paso, que daba ≈24.65 km/h), el equipo decidió usar una **velocidad de diseño fija para toda la flota: 30 km/h (~16.2 nudos)** — `config/instance.yaml` → `calibracion.velocidad_forzada_kmh`. No es la velocidad de un ferry existente; es una suposición de diseño para los barcos pequeños a demanda, que se puede volver a barrer más adelante como parámetro de sensibilidad.

## Resultado: las 6 rutas (todas las combinaciones de a pares entre 4 nodos)

`output/comparacion_recta_vs_navegable.csv` — línea recta (paso 1) vs. ruta navegable real (este paso), ambas a 30 km/h:

| Conexión | km recta | km navegable | diferencia | min recta | min navegable |
|---|---|---|---|---|---|
| Laksevåg ↔ Bryggen | 3.48 | 4.31 | **+23.8%** | 7.0 | 8.6 |
| Kleppestø ↔ Sandviken | 4.33 | 4.78 | +10.5% | 8.7 | 9.6 |
| Bryggen ↔ Sandviken | 3.13 | 3.41 | +8.9% | 6.3 | 6.8 |
| Kleppestø ↔ Laksevåg | 2.47 | 2.67 | +7.8% | 4.9 | 5.3 |
| Kleppestø ↔ Bryggen | 5.36 | 5.75 | +7.3% | 10.7 | 11.5 |
| Laksevåg ↔ Sandviken | 4.05 | 4.31 | +6.4% | 8.1 | 8.6 |

6 pares = C(4,2), todas las combinaciones posibles entre los 4 nodos de demanda — están todas. Con las coordenadas nuevas, Laksevåg (ahora en Gravdal, más adentro del Puddefjorden) es la que más se alarga al rutear sobre agua real, porque el rodeo alrededor de Nordnes pesa más relativo a su distancia total.

## ⚠️ A partir de aquí, usar esta matriz

`output/matriz_tiempos_min.csv` de este paso **reemplaza** al de `01_tiempos_distancias/` para todo lo que siga (`03_demanda/` en adelante).

## Outputs (`output/`)

| Archivo | Qué es |
|---|---|
| `matriz_distancias_km.csv` / `matriz_tiempos_min.csv` | Matrices navegables 4×4 a 30 km/h — **las canónicas de aquí en adelante** |
| `matriz_*_con_waypoints_REFERENCIA.csv` | Igual pero 5×5 incluyendo Hegreneset |
| `comparacion_recta_vs_navegable.csv` | Tabla de diferencias por par (las 6 combinaciones) |
| `velocidad_usada_kmh.txt` | 30.0 (fija, decisión de diseño) |
| `mapa_mascara_agua.png` | Validación: máscara agua/tierra superpuesta sobre el mapa real |
| `mapa_rutas_navegables.png` | Las 6 rutas reales entre los 4 nodos de demanda |
| `mapa_zoom_bryggen_comparacion.png` | Bryggen: recta (roja) vs. ruta real (verde) — la comparación clave |
| `heatmap_tiempos_navegables.png` | Heatmap de la matriz de tiempos a 30 km/h |

## Cómo correr

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Necesita conexión a internet (descarga tiles de mapa la primera vez). Tarda ~15-20 s en total.

## Siguiente paso

`../03_demanda/` — generar las solicitudes Poisson por franja horaria, usando `output/matriz_tiempos_min.csv` de este paso.
