# Paso 1 — Matriz de tiempos y distancias

Primer bloque de la instancia base: cuánto se demora un barco entre cada par de nodos.

## Qué hace

1. Carga los 4 nodos de demanda (Kleppestø, Laksevåg, Bryggen, Sandviken) desde [`../config/instance.yaml`](../config/instance.yaml).
2. Calcula la distancia en línea recta sobre el agua entre cada par, con la fórmula de **Haversine**.
3. Calibra una **velocidad efectiva** con el único dato real disponible: la ruta 490 (Askøybåten) entre Kleppestø y Bryggen mide 5.36 km y el ferry real tarda 14 min puerta a puerta → **≈ 23 km/h (12.4 nudos)**. Esa velocidad ya incluye maniobras y atraque, así que se usa tal cual para todos los pares.
4. Convierte la matriz de distancias a una matriz de **tiempos de viaje** (min) con esa velocidad.
5. Guarda las matrices como CSV y genera las gráficas.
6. **Verificación visual**: dibuja las 6 líneas rectas (una por par de nodos) sobre el mapa real, con zoom cerrado en Bryggen, para revisar si alguna cruza tierra.

## Por qué Hegreneset no aparece en la matriz

Hegreneset es un punto intermedio entre Sandviken y Bryggen, no una parada: nadie sube ni baja ahí. El notebook lo carga por separado (sección `waypoints` del config) solo para mostrarlo en el mapa como referencia geométrica, y produce una matriz de referencia aparte (`output/*_con_waypoints_REFERENCIA.csv`) que **no** se usa en el resto del proyecto — la matriz de demanda real es la de 4×4.

## Cómo correr

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

o abrir `notebook.ipynb` en Jupyter/VS Code y correr todas las celdas.

## Outputs (`output/`)

| Archivo | Qué es |
|---|---|
| `matriz_distancias_km.csv` | Distancias Haversine, 4×4, nodos de demanda |
| `matriz_tiempos_min.csv` | Tiempos de viaje, 4×4, nodos de demanda — **este es el que usan los pasos siguientes** |
| `velocidad_calibrada_kmh.txt` | Velocidad efectiva calibrada (≈ 22.97 km/h) |
| `mapa_nodos_bergen.png` | Mapa de Bergen con los 4 nodos de demanda + Hegreneset marcado como waypoint |
| `heatmap_tiempos.png` | Heatmap de la matriz de tiempos |
| `matriz_*_con_waypoints_REFERENCIA.csv` | Matriz 5×5 de referencia incluyendo Hegreneset — solo para consulta futura, no se usa en la demanda |
| `mapa_lineas_rectas.png` | Las 6 líneas rectas entre nodos, con distancia en km, sobre el mapa |
| `mapa_lineas_rectas_detalle.png` | Lo mismo con más contexto de calles/costa (zoom medio) |
| `mapa_zoom_bryggen.png` | Zoom cerrado sobre Bryggen, para ver el cruce con la península de Nordnes |

## ⚠️ Hallazgo: las líneas rectas que tocan Bryggen cruzan tierra

La inspección visual (sección 9 del notebook) muestra que Haversine subestima la distancia real en las conexiones de Bryggen:

- **Bryggen↔Sandviken**: la recta corta por el centro de Bergen en vez de salir por la boca de Vågen. Cruza tierra con claridad.
- **Bryggen↔Kleppestø** (el propio tramo de calibración) **y Bryggen↔Laksevåg**: pasan muy cerca —probablemente por encima— de la punta de la península de Nordnes.
- **Laksevåg↔Sandviken**: mismo problema con Nordnes.
- **Kleppestø↔Laksevåg y Kleppestø↔Sandviken**: sobre fiordo abierto, sin problema.

Como Bryggen concentra las 3 "conexiones fuertes" del modelo (garantía dura de 15 min), esto es relevante: los tiempos hacia/desde Bryggen podrían estar subestimados. Queda documentado en `../../docs/parametros_instancia_base_bergen.md` como pendiente de decisión — no se corrigió automáticamente porque hay varias formas razonables de hacerlo (factor de desvío manual, ruta por waypoints, esperar datos AIS reales) y es una decisión de modelado.

## Resultado

Matriz de tiempos (min):

| desde \ hacia | Kleppestø | Laksevåg | Bryggen | Sandviken |
|---|---|---|---|---|
| **Kleppestø** | — | 9.3 | 14.0 | 13.5 |
| **Laksevåg** | 9.3 | — | 5.0 | 6.2 |
| **Bryggen** | 14.0 | 5.0 | — | 3.6 |
| **Sandviken** | 13.5 | 6.2 | 3.6 | — |

Coincide con lo documentado en `../../docs/parametros_instancia_base_bergen.md` (test de sanidad incluido en el notebook: Kleppestø–Bryggen debe dar exactamente 14.0 min).

## Siguiente paso

`../02_ruteo_navegable/` — la inspección visual de arriba mostró que las líneas rectas que tocan Bryggen cruzan tierra; ese paso lo corrige con rutas reales sobre una malla de agua. **Los pasos posteriores a ese (`03_demanda/` en adelante) usan la matriz corregida, no la de aquí.**
