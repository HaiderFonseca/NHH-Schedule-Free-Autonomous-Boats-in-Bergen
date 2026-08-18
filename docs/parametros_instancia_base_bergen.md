# Parámetros de la instancia base — barcos a demanda en Bergen

*Primer conjunto de parámetros para arrancar la simulación y la optimización. Todo lo marcado como **[estimado]** o **[a confirmar]** se refina después con datos AIS o con Julio/Stein.*

---

## 1. Puntos de referencia (nodos)

Las cuatro paradas reales del caso (nodos de demanda), con coordenadas reales:

| # | Nodo | Latitud | Longitud | Rol |
|---|------|---------|----------|-----|
| 1 | Kleppestø (Askøy) | 60.4065 | 5.2275 | Punto de isla. Terminal actual de Askøybåten. Conexión larga. |
| 2 | Laksevåg (Gravdal) | 60.390886 | 5.259586 | Oeste, en Gravdal, cruzando el Puddefjorden. **Confirmado.** |
| 3 | Bryggen / Sentrum | 60.3951 | 5.3223 | Hub central (Strandkaiterminalen). Máxima demanda laboral. |
| 4 | Sandviken (BSI Padling) | 60.421149 | 5.300502 | Norte, en BSI Padling (club de piragüismo), camino a NHH/Åsane. **Confirmado.** |

> Estas son las cuatro paradas donde la gente sube y baja. Laksevåg y Sandviken se corrigieron a coordenadas confirmadas (Gravdal y BSI Padling); las anteriores eran aproximaciones.

**Hegreneset (60.4185, 5.3125) no es una parada.** Es un punto intermedio entre Sandviken y Bryggen; nadie embarca ni desembarca ahí. Se conserva únicamente como waypoint de referencia para el ruteo (por ejemplo si se quiere refinar la geometría de una ruta que bordea la costa), pero no participa en la matriz de demanda ni en las decisiones de parada de los barcos.

---

## 2. Matriz de tiempos de viaje

**Cómo se calcula la distancia — dos métodos, dos carpetas:**
- `bergen-boats/01_tiempos_distancias/`: línea recta sobre el agua (fórmula de Haversine). Rápido, pero para las conexiones que tocan Bryggen cruza tierra (ver más abajo).
- `bergen-boats/02_ruteo_navegable/`: ruta real navegable, calculada con un grafo sobre una malla de píxeles de agua (~4.7 m/píxel en Bergen) y Dijkstra. **Esta es la que se usa de aquí en adelante.**

**Velocidad: decisión de diseño fija, 30 km/h (~16.2 nudos).** Se intentó calibrar con el
único dato real disponible (ruta 490 Askøybåten, Kleppestø–Bryggen: 5.36 km en línea recta /
5.75 km navegable, 14 min reales → 23–24.65 km/h según el método), pero el equipo decidió que
la flota nueva de barcos pequeños a demanda no tiene por qué heredar la velocidad de ese ferry
existente. 30 km/h es una suposición de diseño, no una calibración — se puede volver a barrer
como parámetro de sensibilidad más adelante.

**Tiempos en línea recta (minutos, `01_tiempos_distancias/`, a 30 km/h), con las coordenadas confirmadas de Laksevåg (Gravdal) y Sandviken (BSI Padling):**

| desde \ hacia | Kleppestø | Laksevåg | Bryggen | Sandviken |
|---|---|---|---|---|
| **Kleppestø** | — | 4.9 | 10.7 | 8.7 |
| **Laksevåg** | 4.9 | — | 7.0 | 8.1 |
| **Bryggen** | 10.7 | 7.0 | — | 6.3 |
| **Sandviken** | 8.7 | 8.1 | 6.3 | — |

**Tiempos navegables — la matriz a usar (minutos, `02_ruteo_navegable/`, a 30 km/h):**

| desde \ hacia | Kleppestø | Laksevåg | Bryggen | Sandviken |
|---|---|---|---|---|
| **Kleppestø** | — | 5.3 | 11.5 | 9.6 |
| **Laksevåg** | 5.3 | — | 8.6 | 8.6 |
| **Bryggen** | 11.5 | 8.6 | — | 6.8 |
| **Sandviken** | 9.6 | 8.6 | 6.8 | — |

> **Sobre el paper:** el artículo de Gu & Wallace está detrás de pago y no publica abiertamente su matriz exacta de tiempos. Por eso la estimamos desde coordenadas. Para afinarla de verdad se baja un tramo AIS (Kystverket/BarentsWatch) de la ruta 490 y se mide la velocidad real.
>
> **Resuelto (ver `bergen-boats/02_ruteo_navegable/`):** se confirmó visualmente que las líneas rectas que tocan **Bryggen** cruzan tierra (Bryggen↔Sandviken corta por el centro de Bergen; Bryggen↔Kleppestø y Bryggen↔Laksevåg pasan por encima de la península de Nordnes). Se corrigió con un módulo de ruteo sobre una malla de agua real (clasificación de píxeles agua/tierra + grafo + Dijkstra), que calcula el camino más corto que no cruza tierra — el método completo, con la explicación de la resolución del píxel y de cómo funciona Dijkstra, está documentado en `bergen-boats/02_ruteo_navegable/README.md`.
>
> **Hegreneset como waypoint (opcional, no nodo de demanda):** sigue sin ser una parada; se guarda solo como referencia geométrica. Con las coordenadas nuevas de Sandviken ya no queda tan "en medio" del tramo Bryggen-Sandviken como antes — si se necesita como waypoint real para alguna ruta, hay que revisar su utilidad de nuevo.

---

## 3. Parámetros de la flota

| Parámetro | Valor inicial | Nota |
|---|---|---|
| Número de barcos | **5** [a barrer: 3–8] | La pregunta central es cuántos hacen falta para la garantía. |
| Tamaño / capacidad | **todos iguales, 20 pasajeros** [estimado] | Empezar homogéneo: cualquier barco sirve para cualquier viaje. |
| Velocidad efectiva | **30 km/h (fija, decisión de diseño)** | Ver §2 — no es una velocidad calibrada de un ferry existente, es una suposición de diseño para la flota nueva. Después se puede volver variable (optimización de velocidad) o barrer como parámetro de sensibilidad. |
| Posición inicial | todos en Bryggen | Punto de partida neutro; se puede cambiar. |

---

## 4. Tiempo y decisión

| Parámetro | Valor inicial | Nota |
|---|---|---|
| Horizonte del día | 06:00–24:00 (18 h) | Ventana de operación. |
| Época de decisión | cada **3 min** | Cada 3 min se recalcula qué hace cada barco. |
| Horizonte de anticipación | **15–30 min hacia adelante** | Se mira este futuro (con el patrón de demanda) para evitar el efecto cola. |
| Tiempo de atraque | 1–2 min por parada [estimado] | Se puede incorporar o dejar dentro de la velocidad efectiva. |

---

## 5. Garantía de servicio (el corazón del objetivo)

| Parámetro | Valor inicial | Nota |
|---|---|---|
| Espera máxima garantizada | **15 min** | La promesa al pasajero en conexiones importantes. |
| Conexiones fuertes (directas) | Bryggen↔Sandviken, Bryggen↔Laksevåg, Bryggen↔Kleppestø | Mucha demanda → garantía dura. |
| Conexiones débiles (aceptan escala) | la de baja demanda entre las paradas restantes (ej. Kleppestø↔Sandviken) | Garantía más blanda; puede haber transbordo (posiblemente vía Bryggen). |

**Objetivo (elegir uno como principal):**
- **A** — fijar garantía (15 min) y **minimizar el número de barcos / costo de operar**.
- **B** — fijar la flota (5 barcos) y **medir qué garantía se puede cumplir** en cada conexión.

Recuerda: **no se modela ingreso ni ganancia.** El objetivo es servicio eficiente.

---

## 6. Patrón de demanda (inventado, con estructura realista)

La demanda no es un horario; es un patrón espacio-temporal que **genera las solicitudes** (quién aprieta el botón, cuándo, de dónde a dónde) y sirve de **anticipación** para posicionar barcos.

**Estructura por franja horaria:**

| Franja | Volumen | Dirección dominante |
|---|---|---|
| 06:00–09:00 (mañana) | Alto | **Hacia** Bryggen (desde Kleppestø, Laksevåg, Sandviken). ~90/10. |
| 09:00–15:00 (valle) | Medio-bajo | Balanceado ~50/50 entre todos. |
| 15:00–18:00 (tarde) | Alto | **Desde** Bryggen hacia los barrios. ~10/90. |
| 18:00–24:00 (noche) | Bajo | Disperso; los barcos podrían operar como taxis puros. |

**Cómo generar las solicitudes:** proceso de llegadas tipo Poisson por par origen-destino, con una tasa λ que sube en pico y baja en valle, y que respeta la dirección dominante de la franja. (Los números exactos de λ se fijan al construir el código; empezar con algo simple y visible.)

> **Fase posterior (ML):** el patrón se puede *aprender* y actualizar en ventanas de semanas/meses, no día a día, para no dejarse engañar por un evento puntual (una clase de colegio que se mueve entera).

---

## 7. Simplificaciones de la v1 (y qué dejar para después)

**En la v1:**
- Barcos idénticos, capacidad fija.
- Demanda inventada con la estructura de §6.
- Solo la capa flexible (a demanda) sobre estas 4 paradas.
- Velocidad constante (23 km/h).

**Refinamientos posteriores:**
- Barcos de distinto tamaño (hay que elegir cuál va a dónde) y capacidad.
- Velocidad como decisión → optimización de velocidad + combustible/emisiones (gancho cónico de Julio).
- Capa fija (ferries con horario) para el híbrido completo.
- Aprendizaje del patrón de demanda (ML/RL).
- Eventos especiales, operación nocturna, transbordos.

---

## 8. Plataforma y herramientas (stack propuesto)

| Componente | Herramienta | Por qué |
|---|---|---|
| Lenguaje | **Python** | Ecosistema único que junta optimización, simulación y ML/RL. |
| Editor | **VS Code** | Ya lo usas; ideal con la extensión de Python + Jupyter. |
| Optimización (MILP) | **Gurobi** vía `gurobipy` | Licencia académica gratis (NHH suele tenerla). El más rápido. |
| Optimización (cónica/SOCP) | Gurobi o **MOSEK** | Para la parte cónica de Julio; MOSEK es fuerte en conos, licencia académica gratis. |
| Modelado alternativo | Pyomo o PuLP | Si se quiere código de modelo independiente del solver. |
| Simulación | Bucle propio por pasos de tiempo (o **SimPy**) | Rolling-horizon: cada 3 min re-optimizas el estado. |
| RL (fase posterior) | **Gymnasium** + Stable-Baselines3 / PyTorch | Para la política de despacho aprendida. |
| Datos geográficos | GeoPandas + coordenadas | Distancias, y luego cruzar con AIS. |
| Control de versiones | **git + GitHub** | Respaldo y trazabilidad de la tesis. |
| Entorno | conda o venv | Reproducibilidad. |

---

## 9. Preguntas para confirmar con Julio / Stein

- Laksevåg (Gravdal) y Sandviken (BSI Padling) ya están confirmados con coordenadas concretas. ¿Kleppestø y Bryggen también quedan como están? ¿Hegreneset debería usarse como waypoint de ruteo en algún escenario, o se puede omitir del todo en v1?
- ¿Tienen la matriz de tiempos original del paper, para calibrar mejor?
- ¿La garantía entra como restricción dura o como penalización blanda en el objetivo?
- ¿La anticipación al futuro se modela estocástica (Stein) o como penalización de reposicionamiento?
- ¿NHH tiene licencia de Gurobi/MOSEK activa para el proyecto?
- ¿Capacidad de barco y tamaño de flota sugeridos, o los barremos nosotros?
