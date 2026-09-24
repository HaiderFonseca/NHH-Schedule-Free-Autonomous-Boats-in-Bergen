# Heurísticas de despacho - H0 → H1 → H2 → H3

**Qué es esto, en una frase:** la fase de heurísticas fuertes (sin RL) del proyecto de barcos a demanda en Bergen - cuatro políticas de despacho, cada una construida como una extensión estricta de la anterior, auditadas y validadas con casos controlados antes de compararlas en un experimento completo.

**Por qué existe una carpeta `heuristicas/` separada de `politica_base/`:** las primeras versiones de estas ideas (`politica_base/src/politica_h1.py`, `politica_h2/`, `politica_h3/`, `politica_h0c/`) se probaron, no superaron consistentemente a la política base, y una auditoría encontró por qué (una función de costo con un término mal puesto). Esas carpetas se conservan tal cual, como registro histórico del proceso - **no se borra nada** - pero la definición vigente, auditada y validada, es la de aquí.

---

## 1. El ciclo de control (igual para las 4 políticas - esto no cambia nunca)

El simulador (`simulador/src/env.py`) avanza en pasos fijos de 2 minutos. Ninguna política toca esto - todas reciben la misma foto del mundo y devuelven la misma clase de decisión (a qué nodo debe ir cada barco libre, o `None` para esperar). El bucle de control (en cada notebook) hace, en cada paso:

```text
1. Mirar el estado actual: qué colas tienen gente esperando, dónde está cada barco,
   cuáles están libres.
2. Para cada barco LIBRE, la política decide: ¿a qué nodo va, o espera?
   (un barco EN TRÁNSITO no recibe ninguna decisión nueva - el motor no permite
   redirigir a media ruta)
3. El motor aplica las decisiones: el barco libre que decide "ir a B" embarca de
   inmediato lo que quepa de la cola (A,B), FIFO hasta su capacidad, y parte.
4. El motor avanza el reloj 2 minutos. El barco que llega a destino baja a TODOS
   los que lleva (viaje directo, nunca mezcla destinos) y queda libre.
5. El motor incorpora a las colas la gente que llegó durante esos 2 minutos
   - recién visible para la PRÓXIMA decisión, nunca para la que se acaba de tomar.
6. Volver a 1.
```

Ninguna política ve nunca demanda que todavía no llegó (paso 5) ( es lo que garantiza que ninguna de las 4 use información del futuro. La única diferencia entre H0/H1/H2/H3 es **qué calcula el paso 2** ) el resto del ciclo es idéntico, y ninguna lo modifica.

---

## 2. H0 - política base (`politica_base/src/politica_base.py`)

Regla, para un barco libre en el nodo A:


1. **Demanda local primero.** Si hay colas que salen de $A$, el barco va al destino cuya persona más antigua lleva más esperando:

2. **Reposicionamiento.** Solo si no hay ninguna demanda local, el barco va a buscar la demanda remota más urgente (empate: el nodo más cercano en tiempo de viaje):


Coordinación entre varios barcos libres en el mismo paso (`asignar_flota`): decide los barcos en el orden en que aparecen en la lista, descontando de una copia local de las colas lo que cada uno se llevaría, antes de pasar al siguiente.



## 3. H1 = H0 + reserva persistente (`h1_reserva/src/politica_h1.py`)

Toda la regla de decisión de arriba se conserva exactamente igual. Lo único que cambia es cómo se coordinan varios barcos libres, y que ahora existe una reserva que sobrevive entre pasos.

### 3.1 El diccionario de reservas

```python
reservas: dict[str, tuple[str, str]]   # barco_id -> (origen, destino)
```

Vive fuera de la política, mantenido por el bucle de control (igual que ya mantiene `obs, info` de `env.step()`):

```python
reservas = {}
while True:
    decisiones, reservas = asignar_flota_h1(libres, estado, matriz_tiempos, capacidad, reservas)
    ...
```

### 3.2 Cuándo se crea

Solo cuando un barco decide *reposicionarse* (rama 2 de la regla - demanda remota). Una decisión *local* (rama 1) embarca de inmediato - no queda nada pendiente que proteger, así que no genera reserva.

Ejemplo paso a paso: t=100 min, un barco B1 libre en Bryggen, sin demanda local. En Laksevåg→Kleppestø hay 3 personas esperando. B1 decide reposicionarse a Laksevåg → se crea `reservas["B1"] = ("laksevag", "kleppesto")`.

### 3.3 Cómo protege del duplicado

En cada paso, antes de evaluar candidatos, se calcula la cola efectiva de cada par: `len(cola) − Σ(capacidad_barco de cada reserva que apunta a ese par)`. Con capacidad 30 y una sola reserva activa sobre `(laksevag,kleppesto)`, esa cola queda con disponibilidad efectiva `max(0, N−30)` para cualquier OTRO barco libre - si tiene menos de 30 personas, queda en 0: ningún otro barco la ve como candidata.

Siguen llegando personas a la misma cola mientras B1 viaja (2 min después llegan 4 más, total 7) - la reserva sigue cubriendo hasta 30, así que las 7 siguen protegidas sin que nadie tenga que recalcular nada (verificado en `01_metodologia_heuristicas.ipynb`, Caso 2, con dos llamadas separadas: las reservas que entran al paso 2 son *idénticas*, bit a bit, a las que salieron del paso 1).

### 3.4 Cuándo se libera

Un solo evento posible: el barco llega a destino y vuelve a estar libre. El motor nunca redirige un barco a medio camino (`env.py`) y no existe ningún mecanismo de cancelación, así que "la reserva deja de ser viable" no puede pasar antes de llegar - la liberación ocurre automáticamente, al principio de la siguiente llamada a la política, para cualquier barco cuyo `.libre` ya sea `True`.

### 3.5 Coordinación entre varios barcos libres

En vez del orden de lista arbitrario de H0, cada barco libre propone su mejor opción (local si tiene, si no remota); se compromete la propuesta de MAYOR espera entre TODAS las propuestas pendientes primero, se descuenta, se repite.



## 4. H2 = H1 + costo, con partición local/remoto preservada (`h2_costo_local/src/politica_h2.py`)

Función de costo (en minutos):

$$C(b,q) = T_{pickup}(b,q) - W_{max}(q)$$

- $T_{pickup}(b,q)$: minutos que el barco $b$ tarda en llegar al origen de la cola $q$ (0 si ya está ahí).
- $W_{max}(q)$: minutos que lleva esperando la persona más antigua de la cola $q$.

Regla por barco:

```text
1. Si hay demanda local: comparar SOLO las colas locales con C, elegir la de menor C.
2. Si no hay demanda local: comparar SOLO las colas remotas con C, elegir la de menor C.
3. Si no hay nada: esperar.
```

**Por qué la rama local no cambia nada respecto a H0:** para cualquier candidato local, $T_{pickup}=0$, así que $C=-W_{max}$ - minimizar $C$ es exactamente maximizar la espera, la MISMA regla de H0. El costo solo aporta algo nuevo en la rama remota: en vez de "mayor espera, empate por menor viaje" (regla de H0, que compara por umbrales), hace un trade-off continuo - una cola remota algo menos urgente pero mucho más cercana puede ganarle a una más urgente pero muy lejana.

**Una cola remota NUNCA compite con una local**, sin importar la magnitud - verificado numéricamente: con 5 personas esperando 3 min en una cola local y 5 personas esperando **200 min** en una remota, H2 sigue eligiendo la local (`01_metodologia_heuristicas.ipynb`).



## 5. H3 = H1 + costo global, sin partición (`h3_costo_global/src/politica_h3.py`)

Misma función de costo exacta que H2. La única diferencia:

```text
candidatos = TODAS las colas con gente esperando (locales Y remotas a la vez)
-> elegir la de menor C, sin ninguna restricción de nivel
```

**El umbral exacto de cuándo lo remoto le gana a lo local** (para un barco con demanda local de espera $W_{local}$ y una cola remota con pickup $T_{pickup}$):

$$W_{remoto} > T_{pickup} + W_{local}$$

Es decir: lo remoto solo gana cuando su ventaja de espera supera lo que cuesta ir a buscarlo - no por diferencias triviales. **Verificado numéricamente:** con pickup Bryggen→Kleppestø = 11.5 min y espera local = 3 min, el umbral teórico es 14.5 min; H3 efectivamente cruza a la cola remota exactamente entre 14 y 15 min de espera remota, ni un minuto antes (`01_metodologia_heuristicas.ipynb`, Caso 4).

Esta es la única de las 4 que permite que una demanda remota, si es suficientemente urgente, le gane a la local - corrigiendo la debilidad documentada de H0 (puede dejar esperando indefinidamente a alguien muy urgente en otro nodo mientras el barco atiende cualquier cosa local, sin importar cuán poco urgente).


## 6. Comparación de las 4 reglas

| | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| Regla de decisión por barco | local estricto > remoto | igual que H0 | local estricto > remoto, costo dentro de cada nivel | costo global, sin partición |
| Coordinación entre barcos libres en un paso | orden de lista (bug de reposicionamiento) | por urgencia (bug corregido) | por urgencia | por urgencia (costo) |
| Reserva persistente entre pasos | no | sí | sí | sí |
| Función de costo | ninguna | ninguna | $C=T_{pickup}-W_{max}$ (solo remoto) | $C=T_{pickup}-W_{max}$ (local y remoto) |
| ¿Demanda remota puede ganarle a la local? | nunca | nunca | nunca | si $W_{remoto} > T_{pickup}+W_{local}$ |
| Forecasting / demanda futura | no | no | no | no |
| Multi-stop | no | no | no | no |
| Pesos que calibrar | ninguno | ninguno | ninguno | ninguno |



## 7. Resultado central (5 semillas de evaluación, capacidad 30, día completo al 10% de población)

Espera media (min), barrido completo 4-16 barcos:

| Barcos | H0 | H1 | H2 | **H3** |
|---|---|---|---|---|
| 4 | 73.9 | 71.5 (−3%) | 72.2 (−2%) | 72.9 (−1%) |
| 6 | 26.6 | 25.4 (−4%) | 25.7 (−3%) | **24.3 (−8%)** |
| 8 | 12.0 | 12.0 (0%) | 12.0 (0%) | **10.6 (−12%)** |
| 10 | 7.6 | 7.8 (+3%, peor) | 7.8 (+3%, peor) | **6.3 (−17%)** |
| 12 | 6.2 | 7.3 (+18%, peor) | 7.0 (+13%, peor) | **4.8 (−22%)** |
| 14 | 5.6 | 7.1 (+27%, peor) | 6.8 (+22%, peor) | **4.4 (−22%)** |
| 16 | 5.5 | 7.2 (+31%, peor) | 6.7 (+23%, peor) | **4.1 (−25%)** |

H1 y H2 no superan a H0 de forma consistente - la brecha en contra empeora con la flota (H1 llega a +31% peor en 16 barcos): son más conservadoras sobre cuándo mover un barco, sin una forma de valorar si vale la pena, y la flota queda ociosa de más. H3 es la única que gana a H0 en las 7 flotas probadas, y la ventaja se sostiene incluso creciendo (−25% en 16 barcos) - con menos movimientos totales que cualquiera de las otras tres.

La curva de H3 se aplana pero no se estanca del todo: de 12→14 barcos mejora −8.8%, de 14→16 solo −6.9% - rendimientos decrecientes claros, pero H3 sigue extrayendo valor de barcos adicionales más allá de donde H0 ya está prácticamente plano (H0 de 14→16 solo mejora −2.3%). Ver `03_comparacion_flota.ipynb` para la tabla y curvas completas (espera, tiempo en sistema, backlog, movimientos, ocupación, y la serie de barcos en movimiento vs. ociosos por hora del día, incluida como evidencia para evaluar una flota de tamaño variable a lo largo del día).


## 8. Estructura

```
heuristicas/
├── README.md                          -- este archivo
├── comun/src/reservas.py              -- reserva persistente compartida (crear/liberar/descontar)
├── h1_reserva/src/politica_h1.py
├── h2_costo_local/src/politica_h2.py
├── h3_costo_global/src/politica_h3.py
├── experimento_fleet_sweep.py         -- barrido de flota standalone (reusado por el notebook 03)
├── notebooks/
│   ├── 01_metodologia_heuristicas.ipynb   -- reglas de cada política + los 5 casos controlados, ejecutados en vivo
│   ├── 02_experimentos_heuristicas.ipynb  -- H0/H1/H2/H3 bajo condiciones idénticas (1 flota, 1 semilla), gráficas interactivas
│   ├── 03_comparacion_flota.ipynb         -- barrido 4-16 barcos x 5 semillas, todo interactivo (Plotly)
│   └── 04_visualizacion_heuristicas.ipynb -- animación 2h, mapa real, reservas explícitas, inspector de decisión paso a paso
└── outputs/
    ├── resultados/  -- CSVs (por corrida, agregados, y serie de movimiento por minuto)
    └── figuras/     -- PNGs y GIFs
```

## 9. Qué NO hace ninguna de las 4 (restricciones respetadas en todas)

- No usa información de demanda que todavía no llegó (sección 1, paso 5).
- No hace reposicionamiento anticipatorio ni forecasting.
- No permite rutas multi-parada - todo viaje es directo origen→destino.
- No tiene ningún peso o coeficiente que calibrar - todo lo que no es una regla booleana está en minutos.
- No implementa una segunda lógica de embarque/capacidad - eso sigue siendo, en las 4, responsabilidad exclusiva de `simulador/src/env.py`.

## 10. Cómo correr

```bash
cd heuristicas/notebooks
jupyter nbconvert --to notebook --execute --inplace 01_metodologia_heuristicas.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_experimentos_heuristicas.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_comparacion_flota.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_visualizacion_heuristicas.ipynb
```

Necesita `politica_base/output/escalon_dia_10pct/grupos_seed*.csv` ya generado (fase anterior) y `bergen-boats/02_ruteo_navegable/output/` (rutas navegables + matriz de tiempos).
