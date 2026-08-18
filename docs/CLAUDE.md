# Proyecto: Sistema de barcos a demanda en Bergen (tesis NHH)

> Este archivo es el contexto del proyecto. Claude Code lo lee automáticamente al abrir la carpeta. Resume todo lo decidido antes de empezar a programar. Si algo aquí choca con una instrucción nueva del usuario, gana la instrucción nueva.

## Qué estamos construyendo

Una **simulación + optimización** de un servicio de barcos pequeños **a demanda** para Bergen, Noruega. El servicio **no existe todavía**: no se trata de mejorar algo existente, sino de **diseñar la lógica de cómo operaría** para cuando alguien lo monte. Debe parecerse a un sistema de water-taxis.

**Tres principios que no se deben olvidar:**
1. **No se modela ganancia ni ingresos.** El transporte público no es rentable por sí solo; la justificación es ambiental y de descongestión. El objetivo es **dar buen servicio de forma eficiente**, no maximizar plata.
2. **La capa flexible NO tiene horarios.** El pasajero aprieta un botón y un barco viene, con una garantía de espera (ej. 15 min). No se crean horarios; se crea una **política de despacho** que se recalcula cada pocos minutos.
3. **La demanda se inventa** con patrones realistas. No hay datos reales porque el servicio no existe. La demanda cumple dos roles: (a) genera las solicitudes de la simulación, (b) sirve de anticipación para no dejar barcos mal posicionados (evitar el **efecto cola**).

## Contexto académico

- Maestría en NHH (Bergen). Asesor principal: **Julio Goez** (optimización cónica/entera-mixta). También **Stein W. Wallace** (programación estocástica; coautor del paper base de water-taxis).
- Paper base: Gu & Wallace (2021), *Operational benefits of autonomous vessels in logistics — A case of autonomous water-taxis in Bergen*, TR-E 154:102456. Es un modelo **estático** de localización + flota + ruteo. **Nuestra diferencia:** operación en **tiempo real** (despacho minuto a minuto, garantías, anticipación de demanda).
- El ángulo de ML/IA (fase posterior): aprender el patrón de demanda; y/o resolver la política de despacho con aprendizaje por refuerzo.

## Los datos de la instancia base

### Nodos de demanda (4 paradas reales, coordenadas reales)
| # | Nodo | Lat | Lon |
|---|------|-----|-----|
| 1 | Kleppestø (Askøy) | 60.4065 | 5.2275 |
| 2 | Laksevåg | 60.3945 | 5.2875 |
| 3 | Bryggen / Sentrum | 60.3951 | 5.3223 |
| 4 | Sandviken | 60.4075 | 5.3214 |

(Laksevåg es coordenada aproximada, a confirmar con Julio.)

**Nota importante — Hegreneset NO es una parada.** Es un punto intermedio entre Sandviken y Bryggen, no un nodo de demanda: nadie sube ni baja ahí. Solo se conserva como punto de referencia / waypoint opcional para el ruteo (útil si más adelante se quiere modelar la geometría de la costa o una escala física en una ruta con conexión). Sus coordenadas (60.4185, 5.3125) se mantienen en el histórico por si se necesita como waypoint, pero **no entra en la matriz de demanda ni en la lista de nodos donde los barcos recogen/dejan pasajeros**.

### Cómo se construye la matriz de tiempos (reproducir en código)
1. **Distancia:** fórmula de **Haversine** entre coordenadas (distancia en línea recta sobre el agua, en km).
2. **Velocidad efectiva:** se **calibra** con un dato real. La conexión Kleppestø–Bryggen mide 5.36 km en línea recta, y el ferry real (línea 490, Askøybåten) tarda **14 min**. Eso fija una velocidad efectiva de **≈ 23 km/h (12.4 nudos)**, que ya incluye maniobras y atraque.
3. **Tiempo de viaje** = distancia / velocidad. Fórmula: `t_min = haversine_km / 23.0 * 60`.
4. La velocidad debe ser un **parámetro configurable** (para después volverla variable en la optimización de velocidad).

Matriz de tiempos resultante (min) entre las 4 paradas reales, como referencia para tests:
```
              Kleppestø  Laksevåg  Bryggen  Sandviken
Kleppestø         —        9.3      14.0      13.5
Laksevåg         9.3        —        5.0       6.2
Bryggen         14.0       5.0        —        3.6
Sandviken       13.5       6.2       3.6        —
```
> Nota: algunas rectas podrían cruzar tierra (revisar Laksevåg↔Sandviken). Para v1 se acepta la aproximación; después se refina con AIS (Kystverket/BarentsWatch) o rodeando la costa.

### Modelo de demanda (inventado)
Proceso de llegadas tipo **Poisson** por par origen-destino, con tasa λ que depende de la franja horaria:
| Franja | Volumen | Dirección dominante |
|---|---|---|
| 06:00–09:00 | Alto | hacia Bryggen (~90/10) |
| 09:00–15:00 | Medio-bajo | balanceado (~50/50) |
| 15:00–18:00 | Alto | desde Bryggen (~10/90) |
| 18:00–24:00 | Bajo | disperso |
Todos los parámetros de demanda deben vivir en un archivo de config para cambiarlos rápido.

### Flota y decisión (v1)
- 5 barcos, **todos iguales**, capacidad 20 (parámetros configurables; barrer 3–8 barcos).
- Época de decisión: cada **3 min**. Horizonte de anticipación: 15–30 min.
- Garantía de espera: **15 min** en conexiones fuertes (Bryggen↔Sandviken, Bryggen↔Laksevåg, Bryggen↔Kleppestø).
- **Objetivo:** minimizar flota/costo operativo sujeto a cumplir la garantía **o** (variante) dada la flota, medir la garantía alcanzable. NO ingresos.

## Stack técnico

- **Python** (todo el proyecto). Entorno con `venv` o conda.
- Optimización MILP: **Gurobi** vía `gurobipy` (licencia académica gratis; el usuario debe confirmar que NHH la tiene). Parte cónica/SOCP (Julio): Gurobi o **MOSEK**.
- Simulación: bucle propio por pasos de tiempo (rolling-horizon), re-optimizando el estado cada 3 min. `SimPy` opcional.
- ML/RL (fase posterior): Gymnasium + PyTorch.
- Geo/plots: `numpy`, `pandas`, `matplotlib`; `geopandas` + `contextily` para el mapa; `shapely` para geometrías.
- Control de versiones: **git + GitHub**.

## Estructura de carpetas propuesta
```
bergen-boats/
├── CLAUDE.md                  # este archivo
├── README.md
├── requirements.txt
├── config/
│   └── instance.yaml          # nodos, velocidad, flota, demanda, garantía (todo configurable)
├── data/
│   ├── nodes.csv              # las 4 paradas (nodos de demanda) con coordenadas
│   ├── route_490.geojson      # (opcional) ruta existente Askøybåten como capa
│   └── bergen_basemap.geojson # (opcional) costa/agua de Bergen
├── src/
│   ├── geo.py                 # haversine, matriz de distancias/tiempos, calibración
│   ├── demand.py              # generador de solicitudes (Poisson por franja)
│   ├── simulation.py          # bucle rolling-horizon
│   ├── dispatch.py            # política/optimización de despacho (Gurobi)
│   └── plotting.py            # mapa de nodos + capas geojson + animación simple
├── notebooks/
│   └── explore.ipynb
└── docs/
    ├── parametros_instancia_base_bergen.md
    └── modelo_barcos_a_demanda_bergen.md
```

## Convenciones
- **Todo parámetro va en `config/instance.yaml`**, nunca hardcodeado, para cambiar cosas rápido.
- Cada módulo de `src/` con un docstring en español explicando qué hace y por qué.
- Documentar decisiones en `docs/` en markdown.
- Escribir tests mínimos (ej. que la matriz reproduzca los 14 min de Kleppestø–Bryggen).

## Primeras tareas (en orden)
1. Crear la estructura de carpetas, `requirements.txt`, `venv`, y `config/instance.yaml` con todos los parámetros de arriba.
2. `src/geo.py`: cargar los 4 nodos de demanda, calcular matriz de distancias (Haversine) y de tiempos (con velocidad calibrada). Test: Kleppestø–Bryggen ≈ 14 min.
3. `src/demand.py`: generar las solicitudes de un día según las franjas (solo entre los 4 nodos de demanda). Graficar el perfil de demanda para verlo.
4. `src/plotting.py`: dibujar los 4 nodos de demanda sobre un mapa de Bergen (con `contextily`, sin necesidad de descargar nada). Si existe `data/route_490.geojson`, dibujarlo como capa.
5. `src/simulation.py` + `src/dispatch.py`: primera política de despacho simple (ej. asignar el barco libre más cercano) y luego la versión optimizada con Gurobi. Comparar contra la garantía de 15 min.
