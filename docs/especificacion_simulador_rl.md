# Especificación del simulador — barcos autónomos on-demand en Bergen

*Diseño cerrado del entorno (environment) para el agente de aprendizaje por refuerzo. Este documento es la fuente única para construir el simulador y para presentar el diseño a Julio.*

---

## 1. Qué es esto

El simulador es el **entorno (environment)** donde vivirá el agente de RL. Modela la operación de una flota de barcos autónomos que recogen pasajeros bajo demanda en Bergen. En el lenguaje del MDP, el simulador **es la transición**: recibe una acción, avanza el mundo y devuelve el nuevo estado y la recompensa. El agente no ve las reglas de adentro; solo actúa y recibe resultados.

Primero se construye el simulador y se controla con una **política tonta de referencia** (sin aprendizaje). Solo cuando el mundo funciona bien se conecta el agente.

---

## 2. El mundo

- **Nodos (paradas):** 4 — Kleppestø, Laksevåg, Bryggen, Sandviken.
- **Tiempos de viaje:** matriz náutica real ya calculada (ruteo sobre agua, rodeando tierra), a 30 km/h. No se recalcula aquí; se carga del paso de ruteo.
- **Demanda:** grupos de pasajeros que van llegando en el tiempo, generados por el modelo de gravedad + SSB ya construido. Cada grupo tiene origen, destino, minuto de llegada, tamaño y paciencia.

---

## 3. El tiempo: paso fijo de 2 minutos

El simulador avanza en **pasos fijos de 2 minutos**. En cada paso:
1. Llegan los grupos nuevos de ese intervalo.
2. Los barcos avanzan 2 minutos por sus rutas.
3. Se calcula la recompensa del intervalo.
4. Se le pasa el estado al agente y este devuelve la acción.

Se eligió paso fijo de 2 min (en lugar de eventos puros) por simplicidad y porque permite reaccionar a lo que aparezca en cada intervalo sin lógica de eventos enredada.

---

## 4. El estado (lo que ve el agente)

El estado es una foto del mundo en el paso actual. Tiene tres partes:

**A. Los barcos** — un bloque por barco:
- posición: nodo de origen y nodo de destino (si está quieto, son el mismo)
- minutos que faltan para llegar (0 si está quieto)
- ocupación (personas a bordo)
- libre (sí/no)

**B. La demanda esperando** — agregada por par origen-destino (dimensión fija):
- para cada uno de los 12 pares: cuántas personas esperan y hace cuánto espera el grupo más antiguo

> El detalle por grupo (id, tamaño, tiempo) vive **dentro del simulador** para manejar capacidad y recogida; al agente se le pasa el resumen agregado, que es de tamaño fijo y es lo que la red necesita.

**C. El tiempo** — minuto del día (0–1439) y día de la semana (0–6), para que el agente pueda anticipar los patrones por su cuenta sin cortes de franja impuestos.

**Ejemplo de estado (2 barcos):**
```
{
  "tiempo": { "minuto_del_dia": 465, "dia_semana": 1 },
  "barcos": [
    { "origen": "B", "destino": "B", "min_para_llegar": 0, "ocupacion": 0,  "libre": true  },
    { "origen": "K", "destino": "B", "min_para_llegar": 6, "ocupacion": 12, "libre": false }
  ],
  "demanda": {
    "K->B": { "personas": 15, "espera_max": 8 },
    "K->S": { "personas": 3,  "espera_max": 2 },
    "L->B": { "personas": 0,  "espera_max": 0 }
    // ... los 12 pares
  }
}
```
Para la red, este diccionario se aplana a un vector de números de tamaño fijo.

---

## 5. Las acciones (lo que decide el agente)

Cada 2 minutos, el agente decide qué hace **cada barco**:

- **Barco libre:** 5 opciones → ir a Kleppestø, ir a Laksevåg, ir a Bryggen, ir a Sandviken, o **esperar** donde está.
- **Barco en ruta:** sigue hasta su destino (en esta versión **no se redirige** a media ruta). Solo recibe nueva orden cuando llega.

**A quién recoge (regla fija del simulador, no la decide el agente):** cuando un barco llega a un nodo, sube a los grupos en **orden de llegada** (fila: el que lleva más tiempo esperando primero) y va subiendo a todos los que quepan. Al llegar a un grupo que **no cabe entero**, el barco **se detiene ahí** (respeta la fila estricta) y no sigue buscando grupos más pequeños detrás. Ese grupo espera al siguiente barco. El agente no decide esto en la versión inicial (mantiene el espacio de acciones manejable).

**Ejemplo de acción (6 barcos):**
```
{
  "barco_0": "ir_a_Bryggen",
  "barco_1": "seguir",
  "barco_2": "esperar",
  "barco_3": "ir_a_Kleppesto",
  "barco_4": "seguir",
  "barco_5": "ir_a_Sandviken"
}
```

---

## 6. La transición (qué pasa en cada paso)

Al ejecutar la acción, el simulador:
1. Actualiza destinos de los barcos según la acción (respetando que los en-ruta siguen).
2. Avanza 2 minutos: los barcos que llegan a su nodo recogen (regla de §5, respetando **capacidad** y **grupos indivisibles**: un grupo solo sube si cabe entero) y dejan pasajeros.
3. Incorpora los grupos nuevos que aparecieron en el intervalo.
4. Retira como **perdidos** los grupos que agotaron su paciencia.
5. Calcula la recompensa (§7) y construye el nuevo estado.

**Reglas físicas (todas viven en el simulador):**
- Capacidad del barco (ej. 20). Grupos **indivisibles**: si no cabe entero, espera al siguiente barco.
- Un grupo se pierde si su espera supera su **paciencia** (15 min en conexiones fuertes, 30 en el resto — dato de la demanda).

---

## 7. La recompensa (calculada cada 2 minutos)

El simulador calcula la recompensa; el agente solo la recibe. Es negativa (son penalizaciones) y suma tres términos en cada paso:

**a) Incomodidad por tiempo excesivo (escala fija 0–1).** Para cada pasajero que aún no llega a su destino, se define el sobrante sobre una **tolerancia de 12 minutos** (la ruta directa más larga), y se **normaliza por el sobrante máximo** (paciencia − tolerancia) para que quede entre 0 y 1:

```
sobrante      = máximo(0, tiempo_en_el_sistema − 12)
sobrante_máx  = paciencia − 12          # ej. 30 − 12 = 18
penalización_pasajero = (sobrante / sobrante_máx)²
```

Dentro de los 12 min → 0 (zona libre). Justo al borde de perderse → 1. Así toda penalización de pasajero vive en una **escala fija y comparable** (0 = perfecto, 1 = a punto de irse). Crece de forma acelerada (cuadrática). Se suma sobre todos los pasajeros en el sistema, en cada paso. Un pasajero sin atender aparece paso tras paso con sobrante mayor → cuesta cada vez más → presiona a atenderlo.

**b) Pérdida (comparable en la misma escala).** Por cada grupo que agotó su paciencia y se fue en este intervalo: penalización fija `P_perdido` por grupo, un poco mayor que 1 (ej. **1.3**). Como un pasajero al borde vale 1, esto significa que **perder penaliza ~30% más que tenerlo al límite** — comparable y expresable como porcentaje, y suficiente para que perder nunca sea "preferible" a atender.

**c) Movimiento.** `0.1` por cada barco que está navegando en este paso. Desalienta mover barcos sin necesidad → en horas de baja demanda el agente aprende a no usar toda la flota.

**Recompensa del paso:**
```
r = − [ Σ (sobrante/sobrante_máx)²  +  P_perdido × (grupos perdidos)  +  0.1 × (barcos en movimiento) ]
```
Todo en la misma escala: un pasajero va de 0 a 1, un perdido ≈ 1.3, un movimiento 0.1. Comparables entre sí.

---

## 8. La flota

- **Fija dentro de cada entrenamiento** (ej. 6 barcos), todos del mismo tamaño y capacidad.
- El número óptimo de barcos **no** es una acción del agente: se estudia con un **barrido** (entrenar con 4, 6, 8… barcos y comparar servicio vs. costo).
- "Apagar" barcos **emerge** de la acción *esperar* + la penalización de movimiento; no se modela explícitamente.

---

## 9. Política de referencia (sin aprendizaje)

Regla fija tipo **"nearest-available"**: cuando un barco queda libre, se le manda a atender el grupo que lleva más tiempo esperando y que alcanza a recoger a tiempo (o el más cercano). Es la **línea base** contra la que se comparará el agente. Debe quedar como función intercambiable, para sustituirla luego por el agente sin reescribir el simulador.

---

## 10. Parámetros configurables (y para barrer)

Todo en un archivo de config, nada fijo en el código:
- tolerancia de incomodidad (12 min) — define la zona libre y el sobrante
- peso de pérdida `P_perdido` (≈ 1.3, comparable a la escala 0–1 del pasajero)
- peso de movimiento (0.1)
- factor de descuento gamma (≈ 0.99, para el agente)
- tamaño de flota (eje de estudio)
- capacidad de barco
- paso de tiempo (2 min)
- semilla aleatoria (reproducibilidad)

---

## 11. Herramientas

- **Entorno:** escrito como un environment de **Gymnasium** (interfaz estándar: dar estado, ejecutar acción, devolver recompensa).
- **Agente (fase siguiente):** **Stable-Baselines3** sobre **PyTorch**, empezando con **DQN**.
- No se programan las redes ni el entrenamiento a mano; se usan estas librerías estándar.

---

## 12. Camino desde aquí

1. **Construir el simulador** (este documento) + política de referencia, sin aprendizaje. Verificar en instancia mínima (1 franja, 2–3 barcos, pocos grupos).
2. **Medir la línea base** (regla tonta): % atendidos, tiempo de espera, movimientos.
3. **Conectar el agente** (DQN) y entrenar en instancia pequeña.
4. **Comparar** agente vs. línea base vs. óptimo estático en casos chicos. Escalar flota y demanda.

El aporte de la tesis está en el paso 4: una política aprendida en tiempo real para un sistema schedule-free, que los modelos estáticos del grupo no abordaron.

---

## Ajuste posterior — demanda por personas, no por grupos

Cambia la unidad de demanda de grupos a personas individuales. Cada persona es una solicitud independiente con: origen, destino, minuto de llegada a la parada, y paciencia (15 min conexiones fuertes / 30 resto). Esto simplifica capacidad y recogida.

Concretamente:

- **Generación de demanda:** el generador existente produce personas individuales según el mismo patrón (gravedad + SSB + Poisson por franja). Deja el modo "grupos" disponible como opción en config (`unidad_demanda: "personas" | "grupos"`), pero por defecto usa "personas".
- **Recogida (regla del simulador):** cuando un barco llega a un nodo, sube a las personas en orden de llegada (fila) hasta llenar la capacidad. Ya no existe la restricción de "grupo indivisible que no cabe"; simplemente se llena hasta la capacidad y el resto espera al siguiente barco.
- **Estado agregado por par origen-destino:** ahora es la suma directa de personas que esperan en cada par (más el tiempo que lleva la más antigua). Más simple aún.
- **Recompensa:** el término de incomodidad se calcula por persona (igual fórmula normalizada). El término de pérdida `P_perdido` pasa a ser por persona que agota su paciencia (≈1.3 por persona), no por grupo.

Mantén el código estructurado para poder volver a "grupos" activando la opción, sin reescribir el simulador.
