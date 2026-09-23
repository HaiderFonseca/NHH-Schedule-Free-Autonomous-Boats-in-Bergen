"""Heurística coordinada H1 -- despacho por costo mínimo en minutos-persona,
con asignación global y reserva de capacidad por reposicionamiento.

**SUPERADA por H2 (`politica_h2/src/politica_h2.py`) -- se conserva tal cual
como registro del ablation, no se recomienda para producción.** Diagnóstico
por par O-D (semilla 1001, 8 barcos, `politica_base/output/escalon_dia_10pct/`)
encontró que el término `tiempo_viaje(q)` de `costo_par` sesga sistemáticamente
contra las colas locales de viaje largo -- justo las "conexiones fuertes" de
`bergen-boats/config/instance.yaml` -- empeorando `bryggen->kleppesto` en
+69% de espera media frente a H0. H2 es la corrección (mismo algoritmo,
`tiempo_viaje` eliminado del costo) y supera a H0 en TODAS las flotas
probadas (4-12 barcos), mientras que H1c (este módulo) solo empata o pierde
levemente contra H0 en flotas poco congestionadas. Ver `politica_h2/README.md`
para el diagnóstico completo y la literatura que lo respalda.

Reemplaza la regla lexicográfica de `politica_base.py` ("demanda local
SIEMPRE gana sobre remota", decidido barco por barco en el orden fijo de
`estado.barcos`) por una asignación conjunta de mínimo costo evaluada en cada
época de decisión, sobre TODOS los pares (barco libre, cola) factibles a la
vez. Misma cadencia (`paso_tiempo_min`, 2 min), mismas restricciones del
motor (viaje directo punto a punto, sin redirigir a media ruta, embarque FIFO
hasta capacidad, ver `simulador/src/env.py`) -- lo único que cambia es CÓMO
se decide, no las reglas físicas del simulador.

**Todo en minutos-persona, sin pesos que calibrar** -- ver `costo_par`.

**Reserva de capacidad, recalculada cada paso, sin estado externo.** Un
barco en reposicionamiento vacío (en tránsito, sin nadie a bordo) representa
un compromiso YA TOMADO con la cola que sale de su nodo destino -- pero ese
compromiso no se guarda entre pasos en ningún diccionario persistente: se
RECONSTRUYE desde cero en cada llamada a `asignar_flota_h1`, mirando
`estado.barcos` (ya trae quién está en tránsito, hacia dónde, y con qué
ocupación). No hace falta que el bucle de control mantenga nada extra de un
paso al siguiente -- ver `_reservas_por_reposicionamiento`.

**Por qué NO hay una `politica_h1` de un solo barco (a diferencia de
`politica_base.politica_base` + `asignar_flota`):** el punto entero de H1 es
evaluar TODOS los barcos libres contra TODAS las colas a la vez para elegir
la asignación conjunta de menor costo -- decidir "un barco a la vez" no tiene
una versión con sentido aquí (sería, en el mejor caso, el mismo greedy con un
solo barco en la lista). Por eso el único punto de entrada es
`asignar_flota_h1`, con la misma firma que `politica_base.asignar_flota` para
poder sustituir una por otra en el bucle de control sin tocar nada más.
"""
from __future__ import annotations

import pandas as pd

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def _cola_efectiva(
    par: tuple[str, str],
    estado: EstadoSimulacion,
    reservado: dict[tuple[str, str], int],
) -> list[Unidad]:
    """Unidades de `estado.colas[par]` que quedan disponibles después de
    descontar lo ya reservado (por barcos en reposicionamiento, o por otro
    barco ya comprometido en esta misma época -- ver `asignar_flota_h1`).

    Se descuentan las unidades MÁS ANTIGUAS primero (mismo orden FIFO que usa
    el motor para embarcar, `env._embarcar_para_viaje`): son las que un barco
    ya comprometido con este par embarcaría primero, así que son las que
    dejan de estar disponibles para el cálculo de costo de cualquier OTRO
    barco que evalúe este mismo par en la misma pasada.
    """
    cola = estado.colas.get(par, [])
    n_reservado = reservado.get(par, 0)
    return cola[n_reservado:] if n_reservado < len(cola) else []


def costo_par(
    tiempo_pickup: float,
    tiempo_viaje: float,
    cola_disponible: list[Unidad],
    t_actual_min: float,
    capacidad_barco: int,
    ponderar_minutos: bool = True,
) -> tuple[float, int] | None:
    """Costo C(b,q,t) en minutos-persona (documento de diseño de la fase H1,
    sección C, versión final tras revisión):

        k = min(capacidad_barco, len(cola_disponible))
        C = k * (tiempo_pickup + tiempo_viaje) - Σ_{i=1..k} espera_i

    donde `espera_i` es cuánto lleva esperando cada una de las k personas más
    antiguas de `cola_disponible` (las que efectivamente se embarcarían).

    **Por qué `tiempo_pickup` y `tiempo_viaje` se multiplican por k, en vez
    de dejarlos como escalares sueltos (versión inicial del diseño):** la
    espera acumulada (Σ espera_i) crece con k -- una cola de 30 personas
    esperando 15 min cada una aporta -450, mientras que el pickup de un
    barco a 8 minutos de distancia aporta apenas +8. Sin escalar, la espera
    acumulada domina cualquier cola con más de un puñado de personas y el
    pickup/viaje dejan de influir en la decisión -- se pierde la dimensión
    que sí tienen que tener frente a la espera. Escalados por k, los tres
    términos quedan en la MISMA unidad real: minutos-persona.
    - `k * tiempo_pickup`: mientras el barco tarda `tiempo_pickup` minutos en
      llegar, esas k personas acumulan esos mismos minutos DE MÁS de espera
      antes de poder subir -- es la misma "moneda" que la espera ya
      acumulada, no un ajuste artificial de escala.
    - `k * tiempo_viaje`: los minutos-persona que esas mismas k personas van
      a pasar viajando una vez suban. Con esto, `tiempo_viaje` SÍ discrimina
      entre colas distintas para un mismo barco (antes, sin escalar, era una
      constante idéntica para cualquier barco frente a una cola dada, así
      que no aportaba nada a la decisión de qué barco mandar -- solo influía,
      débilmente, al comparar colas entre sí).

    Devuelve `None` si `cola_disponible` está vacía (par no factible: nadie
    esperando ahí, o todo ya reservado/comprometido por otro barco) -- así
    el llamador no repite ese chequeo. Si no es `None`, devuelve `(costo, k)`
    -- `k` se reutiliza para descontar la cola tras comprometer el par.

    `ponderar_minutos=False` (solo para el ablation, ver `asignar_flota_h1`):
    usa en cambio `C = -espera_max` (la espera del más antiguo de la cola,
    SIN pickup ni viaje). **OJO -- esto NO reproduce la regla de H0**
    (`politica_base.politica_base`): H0 tiene una partición ESTRICTA
    ("¿hay algo local? -> ignora todo lo remoto por completo"), que equivale
    a tratar el pickup=0 como una prioridad infinita, no a ignorarlo. Esta
    variante simple, en cambio, compara la espera más vieja de CUALQUIER
    cola del sistema sin ningún término de pickup -- es una línea base de
    ablation deliberadamente más simple y más débil que H0 (mide el efecto
    de tener coordinación global SIN ningún costo de reposicionamiento), no
    una versión "global" de H0. Ver el resultado de esa comparación en el
    documento de diseño, sección de ablation: H1a (esta variante, sin
    reservas) sale PEOR que H0 -- perseguir la espera más vieja del sistema
    entero sin penalizar el viaje vacío para llegar genera reposicionamientos
    caros que H0 nunca haría, precisamente porque H0 nunca abandona la
    demanda local por la remota. Es el hallazgo que motiva el término de
    pickup en `H1c`, no un resultado indeseado.
    """
    k = min(capacidad_barco, len(cola_disponible))
    if k == 0:
        return None
    embarcarian = cola_disponible[:k]
    if ponderar_minutos:
        espera_total = sum(t_actual_min - u.minuto_llegada for u in embarcarian)
        costo = k * (tiempo_pickup + tiempo_viaje) - espera_total
    else:
        espera_max = t_actual_min - embarcarian[0].minuto_llegada
        costo = -espera_max
    return costo, k


def _reservas_por_reposicionamiento(
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    ponderar_minutos: bool = True,
) -> dict[tuple[str, str], int]:
    """Cuántas unidades de cada par (o,d) están efectivamente reservadas por
    barcos que YA están en camino, vacíos, hacia el nodo o (documento de
    diseño, sección E -- el matiz del "barco en camino"). Se recalcula desde
    cero en cada llamada -- no hay estado persistente entre pasos, y por eso
    tampoco hace falta ninguna condición explícita de "la reserva dejó de ser
    factible": si algo cambió, la próxima llamada ya lo ve.

    Un barco "en reposicionamiento" es uno EN TRÁNSITO (`not b.libre`) y sin
    nadie a bordo (`not b.a_bordo`). Por construcción del motor (viajes
    directos punto a punto, ver `simulador/src/env.py`/`estado.py`), un barco
    con gente a bordo va a un destino ya resuelto por un embarque real que
    YA ocurrió -- al llegar deja a todos y vuelve a decidir desde cero, sin
    que ese destino le "deba" nada a ninguna cola. Por eso solo los barcos
    vacíos en tránsito participan de esta reserva.

    Para cada barco en reposicionamiento hacia el nodo X, se elige la cola
    (X, d) más barata evaluada COMO SI el barco ya estuviera libre en X
    (`tiempo_pickup=0`, porque ya va para allá -- nunca se redirige a medio
    camino, así que X es un hecho, no una opción) y se le reserva su
    capacidad completa contra esa cola.

    Se procesan los barcos en reposicionamiento en orden de llegada más
    próxima primero (`min_para_llegar` ascendente): si dos barcos van al
    mismo nodo X, el segundo ve la cola ya descontada por el primero y se
    reparte naturalmente hacia la siguiente más urgente en X -- salvo que la
    cola sea grande de verdad y sobre gente después de descontar al primero,
    en cuyo caso SÍ se apilan los dos sobre la misma (exactamente el matiz
    pedido: la reserva es de capacidad, no de la cola entera).
    """
    reservado: dict[tuple[str, str], int] = {}
    en_reposicionamiento = [b for b in estado.barcos if not b.libre and not b.a_bordo]
    en_reposicionamiento.sort(key=lambda b: b.min_para_llegar)

    for barco in en_reposicionamiento:
        X = barco.nodo_destino
        mejor: tuple[float, tuple[str, str]] | None = None
        for par in estado.colas:
            o, d = par
            if o != X:
                continue
            disponible = _cola_efectiva(par, estado, reservado)
            resultado = costo_par(
                tiempo_pickup=0.0,
                tiempo_viaje=float(matriz_tiempos.loc[o, d]),
                cola_disponible=disponible,
                t_actual_min=estado.t_actual_min,
                capacidad_barco=capacidad_barco,
                ponderar_minutos=ponderar_minutos,
            )
            if resultado is None:
                continue
            costo, _ = resultado
            if mejor is None or costo < mejor[0]:
                mejor = (costo, par)
        if mejor is not None:
            par_reservado = mejor[1]
            reservado[par_reservado] = reservado.get(par_reservado, 0) + capacidad_barco

    return reservado


def asignar_flota_h1(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    cfg: dict | None = None,
    usar_reservas: bool = True,
    ponderar_minutos: bool = True,
) -> dict[str, str | None]:
    """Asignación global de mínimo costo para los barcos libres de esta
    época de decisión (documento de diseño, secciones B-E). Misma firma que
    `politica_base.asignar_flota` (mismo lugar en el bucle de control, mismo
    contrato de entrada/salida: `{barco_id: nodo_destino | None}`) -- pensada
    para sustituir una por otra sin tocar nada más del notebook.

    Algoritmo (greedy global por costo ascendente, documento de diseño
    sección D -- preferido sobre matching húngaro porque, con capacidad
    limitada por barco, el greedy maneja EXACTO el caso "una cola grande
    justifica más de un barco" mediante descuento secuencial, mientras que
    Hungarian necesitaría aproximar ese descuento replicando columnas):

    1. Calcular qué parte de cada cola ya está reservada por barcos en
       reposicionamiento (`_reservas_por_reposicionamiento`).
    2. Mientras queden barcos libres sin decisión: evaluar `costo_par` para
       TODOS los pares (barco libre pendiente, cola con disponibilidad) a la
       vez, tomar el de menor costo, comprometerlo (la acción es el DESTINO
       de la cola si el barco ya está en el origen -- embarca de inmediato;
       o el ORIGEN de la cola si no -- reposicionamiento vacío, se evalúa de
       nuevo al llegar), descontar esa cantidad de la cola disponible (para
       que otro barco libre pueda seguir sirviendo la MISMA cola si sobra
       gente) y sacar ese barco del conjunto de pendientes.
    3. Los barcos que terminan sin ningún par factible: esperan (`None`),
       igual que en H0 cuando no hay nada que hacer en ningún lado.

    `cfg` no se usa -- la función de costo no tiene ningún parámetro que
    calibrar (documento de diseño, principio "todo en minutos, sin pesos").
    Se deja en la firma solo por intercambiabilidad con `asignar_flota`.

    **`usar_reservas` y `ponderar_minutos` -- para el ablation (documento de
    diseño, sección I), no para uso normal:**
    - `H1a` = `asignar_flota_h1(..., usar_reservas=False, ponderar_minutos=False)`:
      asignación conjunta global (el cambio de "decidir barco por barco en
      orden fijo" a "evaluar todos los pares a la vez"), pero con un costo
      que solo mira la espera del más antiguo de cada cola -- SIN ningún
      término de pickup, y sin reservas. **No es "la regla de H0 evaluada
      globalmente"** (ver la nota en `costo_par`): H0 nunca compara espera
      local contra remota, siempre agota lo local primero -- eso equivale a
      tratar el pickup=0 como infinitamente mejor, no a no considerarlo.
      H1a mide, a propósito, qué tan bien funciona la coordinación conjunta
      POR SÍ SOLA, sin ninguna noción de costo de reposicionamiento --
      resultado observado: peor que H0 (ver documento de diseño), porque sin
      pickup persigue la espera más vieja del sistema aunque eso implique un
      viaje vacío caro. Es el hallazgo que justifica por qué H1c necesita el
      término de pickup, no un resultado indeseado.
    - `H1b` = `asignar_flota_h1(..., usar_reservas=True, ponderar_minutos=False)`:
      H1a + reservas de capacidad entre pasos (sección E) -- mejora sobre
      H1a (evita reposicionamientos redundantes) pero sigue sin pickup, así
      que sigue sin alcanzar a H0 en las corridas de verificación.
    - `H1c` = `asignar_flota_h1(...)` (los defaults) = H1b + la función de
      costo en minutos-persona completa (sección C, con pickup y viaje
      ponderados por k). **Ya NO es la versión recomendada** -- el sweep de
      flota (4-12 barcos, `politica_base/output/escalon_dia_10pct/`) mostró
      que solo empata o pierde levemente contra H0 en flotas poco
      congestionadas (8+ barcos), y el diagnóstico por par encontró que
      `tiempo_viaje` sesga contra colas locales largas. La versión corregida
      es H2 (`politica_h2/src/politica_h2.py`), que sí supera a H0 en todas
      las flotas probadas -- ver `politica_h2/README.md`.
    """
    reservado = (
        _reservas_por_reposicionamiento(estado, matriz_tiempos, capacidad_barco, ponderar_minutos)
        if usar_reservas else {}
    )
    disponible: dict[tuple[str, str], list[Unidad]] = {
        par: _cola_efectiva(par, estado, reservado) for par in estado.colas
    }

    pendientes = list(barcos_libres)
    decisiones: dict[str, str | None] = {}

    while pendientes:
        mejor: tuple[float, Barco, tuple[str, str]] | None = None
        for barco in pendientes:
            A = barco.nodo_origen
            for par, cola in disponible.items():
                o, d = par
                resultado = costo_par(
                    tiempo_pickup=float(matriz_tiempos.loc[A, o]),
                    tiempo_viaje=float(matriz_tiempos.loc[o, d]),
                    cola_disponible=cola,
                    t_actual_min=estado.t_actual_min,
                    ponderar_minutos=ponderar_minutos,
                    capacidad_barco=capacidad_barco,
                )
                if resultado is None:
                    continue
                costo, _ = resultado
                if mejor is None or costo < mejor[0]:
                    mejor = (costo, barco, par)

        if mejor is None:
            break  # a nadie le queda ya ningun par factible -- el resto espera

        _, barco_elegido, par_elegido = mejor
        o_par, d_par = par_elegido
        A = barco_elegido.nodo_origen
        # Si el barco ya esta en el origen del par, la accion es ir al
        # DESTINO (embarca de inmediato, viaje directo). Si no, la accion es
        # ir al ORIGEN del par (reposicionamiento vacio -- al llegar, decide
        # de nuevo con informacion fresca, igual que en H0).
        decisiones[barco_elegido.id] = d_par if o_par == A else o_par
        pendientes.remove(barco_elegido)
        disponible[par_elegido] = disponible[par_elegido][capacidad_barco:]

    for barco in pendientes:
        decisiones[barco.id] = None  # sin ningun par factible -- esperar

    return decisiones
