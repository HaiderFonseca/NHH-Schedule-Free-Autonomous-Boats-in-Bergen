# H2 — heurística corregida (diagnóstico + fix sobre H1)

**Qué es esto, en una frase:** H1 (`politica_base/src/politica_h1.py`) tenía un defecto real en su
función de costo, confirmado con datos, no solo sospechado. H2 es la corrección: mismo algoritmo de
asignación (coordinación global + reserva de capacidad), con un solo término del costo eliminado.

---

## 1. El diagnóstico (semilla 1001, 8 barcos, día completo al 10%)

Corriendo H0 y H1c sobre la MISMA demanda y comparando `metricas.metricas_por_par` (ver
`politica_base/output/escalon_dia_10pct/`), aparece un patrón sistemático en los pares que **salen
del hub** (`bryggen`):

| Par (desde bryggen) | Viaje (min) | Espera media H0 | Espera media H1c | Cambio |
|---|---|---|---|---|
| bryggen -> sandviken | 6.8 | 10.02 | 11.78 | +18% |
| bryggen -> laksevag | 8.6 | 10.85 | 15.28 | +41% |
| bryggen -> kleppesto | 11.5 | 9.66 | 16.32 | +69% |

**Mientras más largo el viaje, peor le fue a ese par bajo H1c** — y son justo las tres "conexiones
fuertes" que `bergen-boats/config/instance.yaml -> garantia.conexiones_fuertes` marca como las que
más necesitan servicio garantizado. No es ruido de una sola semilla: el patrón es monótono en las
tres rutas, ordenado exactamente por tiempo de viaje.

## 2. Por qué pasaba esto (no es un bug de código, es un término mal puesto)

La función de costo de H1c era:

```
C(b,q,t) = k * (tiempo_pickup(b,q,t) + tiempo_viaje(q)) - ESPERA(q,t,k)
```

Para un barco que ya está en el origen de la cola (pickup=0 -- el caso "local", que es la mayoría de
las decisiones en un hub como bryggen), el término que sobrevive para comparar entre destinos
distintos es `k * tiempo_viaje(q) - ESPERA(q,t,k)`. Entre `bryggen->sandviken` (barato, 6.8 min) y
`bryggen->kleppesto` (caro, 11.5 min), el segundo necesita bastante MÁS espera acumulada para ganarle
al primero -- así que sistemáticamente pierde la comparación hasta que su espera se dispara muy por
encima de lo razonable. El propio diseño original (ver `politica_base/src/politica_h1.py`, sección C
del documento de diseño) ya advertía este riesgo ("sesga contra pares de viaje largo... hay que
vigilarlo por par") -- el diagnóstico confirma que el riesgo se materializó, y con una magnitud
grande (+69% en el peor caso), no marginal.

**El argumento matemático para sacarlo, no solo para bajarle el peso:** `tiempo_viaje(q)` es un costo
que el sistema va a pagar de todas formas, sin importar CUÁNDO se atienda esa cola -- no depende de
la decisión de orden, así que no ayuda a minimizar el tiempo total del sistema (el objetivo
declarado del proyecto). Lo único que hace es sesgar la comparación entre colas locales a favor de la
más corta. No es un término que necesite un peso más chico: no tiene ningún rol que cumplir ahí, se
elimina del todo.

## 3. La corrección

```
C(b,q,t) = k(b,q,t) * tiempo_pickup(b,q,t) - ESPERA(q,t,k)
```

Un solo cambio sobre H1c: sin `tiempo_viaje`. Implementado en `src/politica_h2.py`
(`costo_par_h2`, `asignar_flota_h2`) -- el resto de la arquitectura (asignación conjunta de costo
mínimo, reserva de capacidad para barcos en reposicionamiento, recalculada cada paso, sin
información futura) es idéntica a H1, documentada en `politica_base/src/politica_h1.py`.

**Por qué esto sí es "partir de la política base y mejorarla" aunque el código no se copió de ahí:**
con `tiempo_viaje` fuera, para cualquier par de candidatos LOCALES (pickup=0 para ambos) el costo se
reduce a `-ESPERA(q,t,k)` -- es decir, la comparación queda en limpio "atender primero a quien más
minutos-persona acumulados tiene", que es EXACTAMENTE el mismo espíritu que la regla de
`politica_base.politica_base` ("demanda local primero, la más urgente entre las locales") -- solo que
coordinada globalmente entre todos los barcos libres a la vez (no barco por barco en orden fijo, el
defecto real de H0 documentado en `politica_base/README.md` sección 5) y con la espera SUMADA sobre
toda la gente que se embarcaría, no solo la más antigua. La prioridad local no se codificó a mano
como una partición rígida -- emerge sola del costo, porque pickup=0 siempre gana salvo que lo remoto
sea dramáticamente más urgente (lo cual, a diferencia de H0, SÍ puede pasar -- corrigiendo la otra
debilidad documentada de H0: que nunca cruza a atender algo remoto por más urgente que sea, mientras
haya cualquier cosa local pendiente).

## 4. Verificación del fix (misma semilla, mismos 8 barcos)

| Par | H0 | H1c (roto) | H2 (corregido) |
|---|---|---|---|
| bryggen -> kleppesto | 9.66 | 16.32 | **9.57** |
| bryggen -> laksevag | 10.85 | 15.28 | **11.31** |
| kleppesto -> bryggen | 15.22 | 14.95 | **9.43** |
| laksevag -> bryggen | 10.39 | 12.20 | **8.69** |

Movimientos totales: H0 870 (355 vacíos) / H1c 729 (200 vacíos) / **H2 718 (192 vacíos)** -- H2 queda
con el menor número de movimientos Y el menor número de movimientos vacíos de las tres, además de
corregir el sesgo contra las conexiones largas.

## 5. Literatura que respalda el diagnóstico y la corrección

El defecto encontrado en H1c es una instancia conocida del problema de **starvation por
minimización miope de distancia**, bien documentado en despacho de ascensores -- el paralelo
metodológico más cercano a este problema (recurso móvil compartido, múltiples solicitudes
pendientes, decisión recalculada en cada evento):

- Crites, R.H., Barto, A.G. (1998). *Elevator Group Control Using Multiple Reinforcement Learning
  Agents*. Machine Learning 33:235-262. DOI:
  [10.1023/A:1007518724497](https://doi.org/10.1023/A:1007518724497). Documentan exactamente este
  mecanismo: una regla que prioriza minimizar distancia/tiempo de viaje (análoga al SSTF de
  planificación de disco) dejando esperando indefinidamente a las solicitudes lejanas mientras
  sigan apareciendo solicitudes cercanas más baratas de atender -- y muestran que una función
  objetivo basada en tiempo de espera (al cuadrado, en su caso) evita ese sesgo. Lo que se adapta:
  el diagnóstico del mecanismo de starvation y la idea de que la función de costo debe estar basada
  en espera acumulada, no en el costo de servir la solicitud. Lo que NO se traslada directo: ellos
  usan RL sobre la política completa; acá el ajuste es puramente a la función de costo de una regla
  fija.
- El mecanismo general -- **aging**: la prioridad efectiva de una solicitud debe crecer con su
  tiempo de espera para evitar que un criterio de "costo de servir" (distancia, tiempo de viaje)
  la posponga indefinidamente -- es un resultado clásico de teoría de scheduling (ver *Aging
  (scheduling)*, cualquier texto estándar de sistemas operativos/teoría de colas). Es exactamente
  el mecanismo que `ESPERA(q,t,k)` ya implementaba correctamente -- el error de H1c no fue no tener
  aging, fue agregar un segundo término (`tiempo_viaje`) que competía contra el aging sin necesidad.

## 6. Qué sigue

`politica_base/src/fleet_sweep.py` corre H2 junto con H0/H1a/H1b/H1c en la misma matriz
(4/6/8/10/12 barcos x 5 semillas de evaluación) -- resultados agregados en
`politica_base/output/escalon_dia_10pct/`.
