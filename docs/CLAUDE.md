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
- **Despacho: no se usó Gurobi/MILP en la práctica.** La idea original de esta sección (abajo queda como registro histórico) era optimizar el despacho con un solver entero-mixto; el proyecto evolucionó hacia **simulación + política de despacho** (una heurística de referencia, después un agente de RL) -- ver "Estado actual" más abajo. No se descarta retomar una capa de optimización más adelante, pero no es lo que hoy corre.
- Simulación: **Gymnasium** (`gymnasium.Env`, pasos fijos de 2 min) -- no un bucle rolling-horizon con re-optimización cada 3 min como se planteó originalmente; ver `simulador/README.md` sección 2.
- RL: **Stable-Baselines3** (PPO) + PyTorch, sobre el entorno Gymnasium -- ver `modelo_rl/README.md`.
- Geo/plots: `numpy`, `pandas`; `geopandas` + `contextily` para el mapa; `shapely` para geometrías; `plotly` para las gráficas interactivas de métricas (`matplotlib` solo para el mapa animado).
- Control de versiones: **git + GitHub**.

## Estado actual de la estructura (reemplaza la propuesta original de abajo)

```
.
├── bergen-boats/          # Instancia base: nodos, matriz de tiempos, ruteo navegable
├── demand/                 # Generación de demanda sintética (gravedad + SSB + Poisson)
├── simulador/               # El MOTOR: entorno Gymnasium, estado, recompensa, métricas, visualización
├── politica_base/            # Política de referencia ("nearest-available") + verificación (escalones 1-3)
├── modelo_rl/                 # Agente PPO: entrenamiento, experimentos de recompensa, corrida final
├── comparacion/                 # Agente PPO vs. política base, mismas semillas de evaluación
├── docs/                         # Este archivo + especificación + informe LaTeX (docs/informe/)
└── papers/                       # Papers de referencia
```

`simulador/`, `politica_base/`, `modelo_rl/` y `comparacion/` son 4 bloques con responsabilidad única (ver `simulador/README.md` sección 1 para el detalle de cada uno y cómo se conectan) -- reemplazan una carpeta `simulacion/` anterior que mezclaba las cuatro cosas. `simulador/config/instance.yaml` es la única fuente de verdad de los parámetros que comparten los 4 bloques (recompensa, escalones, hiperparámetros de PPO); cada bloque lee ese archivo, ninguno lo copia -- misma regla que ya usaban `bergen-boats/` y `demand/` con sus propios `config/instance.yaml`.

Documento de diseño detallado del simulador/MDP: `docs/especificacion_simulador_rl.md`. Informe final (LaTeX, en inglés): `docs/informe/`.

## Convenciones

- **Todo parámetro va en un `config/instance.yaml`** propio de cada bloque, nunca hardcodeado -- las demás carpetas lo leen, nunca lo copian.
- Cada módulo de `src/` con un docstring explicando qué hace y por qué (español en el código de simulación/RL, inglés en las figuras/informe final).
- Documentar decisiones en el `README.md` de cada bloque, o en `docs/` para decisiones transversales.
- Reproducibilidad: semillas fijas y explícitas para generar demanda; semillas de evaluación (`simulador/config/instance.yaml` → `agente.evaluacion.semillas`) siempre fuera del rango usado para entrenar.

## Estructura de carpetas propuesta originalmente (registro histórico, ya no vigente)

La sección de abajo era el plan de arranque del proyecto (antes de que existiera `bergen-boats/02_ruteo_navegable`, `demand/`, o cualquiera de los 4 bloques de simulación) -- se conserva para trazabilidad de cómo cambió el enfoque (de MILP/Gurobi hacia simulación + RL), no como referencia de dónde está el código hoy (ver "Estado actual" arriba).

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
