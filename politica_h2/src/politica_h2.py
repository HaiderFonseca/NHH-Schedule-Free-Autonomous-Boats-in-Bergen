"""Heurística H2 -- corrección de H1 tras diagnóstico con datos reales.

**Qué pasó con H1 (`politica_base/src/politica_h1.py`), diagnosticado
corriendo H0 vs H1c sobre la misma demanda (semilla 1001, 8 barcos,
`politica_base/output/escalon_dia_10pct/`) y comparando `metricas_por_par`:**
los pares que salen del hub (`bryggen`) hacia destinos LARGOS empeoraron
mucho bajo H1c frente a H0 -- `bryggen->kleppesto` (12 min de viaje): 9.66
min de espera media en H0 vs. **16.32 min en H1c** (+69%); `bryggen->laksevag`
(10 min): 10.85 vs. 15.28 (+41%). El par corto `bryggen->sandviken` (8 min)
en cambio mejoró levemente. El patrón es sistemático: mientras más largo el
viaje, peor le fue a ese par bajo H1c -- exactamente lo que predice el
término `k * tiempo_viaje(q)` que H1c sumaba al costo.

**Por qué ese término estaba mal, no solo mal calibrado:** para un candidato
LOCAL (el barco ya está en el origen de la cola, pickup=0), `tiempo_viaje(q)`
es un costo que hay que pagar de todas formas, sin importar CUÁNDO se
atienda esa cola -- no depende de la decisión, así que no ayuda a minimizar
el tiempo total del sistema. Lo único que hace es sesgar la comparación
entre colas locales a favor de la más corta, penalizando a quien vive o
trabaja en el nodo más lejano -- justo las 3 "conexiones fuertes"
(`bergen-boats/config/instance.yaml -> garantia.conexiones_fuertes`), que el
propio proyecto marca como las que más necesitan servicio garantizado. Se
saca del todo, no se reescala ni se le baja el peso -- no tiene un rol que
cumplir en la comparación local.

**Función de costo final (documento de diseño, sección C, revisión 2):**

    C(b,q,t) = k(b,q,t) * tiempo_pickup(b,q,t) - ESPERA(q,t,k)

Un solo cambio respecto a `politica_h1.costo_par`: sin `tiempo_viaje`. Con
esto, para cualquier par de candidatos LOCALES (pickup=0 para ambos), la
comparación queda en limpio "atender a quien más minutos-persona acumulados
tiene" -- sin ninguna distorsión -- que es, en espíritu, la MISMA prioridad
que ya usaba `politica_base.politica_base` (demanda local primero), solo que
ahora coordinada globalmente entre todos los barcos libres a la vez (no
barco por barco en orden fijo) y con la espera SUMADA sobre toda la gente
que se embarcaría, no solo la persona más antigua.

**Y para candidatos remotos, H2 corrige una debilidad real y documentada de
H0** (`politica_base/README.md`, sección 5): H0 SIEMPRE prefiere cualquier
demanda local, sin importar cuán chica, sobre cualquier demanda remota, sin
importar cuán urgente -- puede dejar esperando indefinidamente a alguien muy
urgente en otro nodo. H2 no tiene esa regla absoluta: el pickup solo suma un
costo proporcional (`k * tiempo_pickup`), así que en el caso normal lo local
sigue ganando (pickup=0 vs. pickup>0, a igual espera acumulada), pero si una
cola remota acumula muchísima más espera que cualquier cosa local, SÍ puede
ganar la comparación. No es una partición rígida "local siempre gana" --
emerge del costo, sin necesidad de codificar dos niveles de prioridad a mano.

El resto de la arquitectura (asignación conjunta de costo mínimo, reserva de
capacidad para barcos en reposicionamiento, todo recalculado cada paso, sin
información futura) es EXACTAMENTE la de H1 -- ver
`politica_base/src/politica_h1.py` para la documentación completa de esas
piezas, que no cambiaron. Este módulo solo corrige `costo_par`.
"""
from __future__ import annotations

import pandas as pd

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def costo_par_h2(
    tiempo_pickup: float,
    cola_disponible: list[Unidad],
    t_actual_min: float,
    capacidad_barco: int,
) -> tuple[float, int] | None:
    """C(b,q,t) = k * tiempo_pickup - ESPERA(q,t,k) -- ver docstring del
    módulo para por qué NO lleva `tiempo_viaje` (a diferencia de
    `politica_h1.costo_par`). `k * tiempo_pickup` sigue en minutos-persona
    (mientras el barco tarda `tiempo_pickup` minutos en llegar, las k
    personas que se embarcarían acumulan esos mismos minutos de más de
    espera) -- misma unidad que `ESPERA`, misma lógica de escalado que en H1.
    """
    k = min(capacidad_barco, len(cola_disponible))
    if k == 0:
        return None
    embarcarian = cola_disponible[:k]
    espera_total = sum(t_actual_min - u.minuto_llegada for u in embarcarian)
    costo = k * tiempo_pickup - espera_total
    return costo, k


def _cola_efectiva(
    par: tuple[str, str],
    estado: EstadoSimulacion,
    reservado: dict[tuple[str, str], int],
) -> list[Unidad]:
    """Idéntico a `politica_h1._cola_efectiva` -- ver esa docstring."""
    cola = estado.colas.get(par, [])
    n_reservado = reservado.get(par, 0)
    return cola[n_reservado:] if n_reservado < len(cola) else []


def _reservas_por_reposicionamiento(
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
) -> dict[tuple[str, str], int]:
    """Idéntico en espíritu a `politica_h1._reservas_por_reposicionamiento`
    (mismo mecanismo: recalculado cada paso, sin estado persistente, barcos
    en reposicionamiento vacío procesados por ETA ascendente) -- solo usa
    `costo_par_h2` (sin viaje) para elegir el destino virtual de cada barco
    en tránsito, en vez de `politica_h1.costo_par`. `tiempo_viaje` no
    aparece aquí tampoco: el pickup es 0 (el barco ya va para el nodo X), así
    que ni siquiera hubiera tenido con qué compararse.
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
            resultado = costo_par_h2(
                tiempo_pickup=0.0,
                cola_disponible=disponible,
                t_actual_min=estado.t_actual_min,
                capacidad_barco=capacidad_barco,
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


def asignar_flota_h2(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    cfg: dict | None = None,
) -> dict[str, str | None]:
    """Misma firma y mismo algoritmo de asignación conjunta que
    `politica_h1.asignar_flota_h1` (greedy global por costo ascendente, con
    reserva de capacidad para reposicionamiento) -- la única diferencia es
    `costo_par_h2` en vez de `costo_par`. Ver `politica_h1.asignar_flota_h1`
    para la documentación completa del algoritmo de asignación; este
    docstring solo cubre lo que cambia.
    """
    reservado = _reservas_por_reposicionamiento(estado, matriz_tiempos, capacidad_barco)
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
                resultado = costo_par_h2(
                    tiempo_pickup=float(matriz_tiempos.loc[A, o]),
                    cola_disponible=cola,
                    t_actual_min=estado.t_actual_min,
                    capacidad_barco=capacidad_barco,
                )
                if resultado is None:
                    continue
                costo, _ = resultado
                if mejor is None or costo < mejor[0]:
                    mejor = (costo, barco, par)

        if mejor is None:
            break

        _, barco_elegido, par_elegido = mejor
        o_par, d_par = par_elegido
        A = barco_elegido.nodo_origen
        decisiones[barco_elegido.id] = d_par if o_par == A else o_par
        pendientes.remove(barco_elegido)
        disponible[par_elegido] = disponible[par_elegido][capacidad_barco:]

    for barco in pendientes:
        decisiones[barco.id] = None

    return decisiones
