"""Heurística H3 -- costo en minutos SIN ninguna cantidad de personas (ni
suma de esperas, ni ponderación por k), solo el máximo (el más antiguo de la
cola) contra el pickup. Variante pedida explícitamente tras revisar H2: la
suma de esperas (`ESPERA(q,t,k)` de H1/H2) mejora el tiempo medio y P95
frente a H0 en las 5 flotas probadas, pero empeora la espera MÁXIMA absoluta
en las 5 flotas también (11%-35% peor) -- un caso chico y aislado
(`kleppesto->laksevag`, el par de menor volumen del sistema) puede quedar
postergado porque atender una cola grande "rinde más" en la suma, aunque la
persona más vieja de la cola chica lleve más esperando. H3 prueba si volver
al máximo (sin sumar personas) resuelve eso sin perder las otras ganancias.

**Función de costo:**

    C(b,q,t) = tiempo_pickup(b,q,t) - espera_max(q,t)

Sin ningún `k *` -- ni en el pickup ni en la espera. Con `ESPERA` reducida a
un solo escalar (la espera del más antiguo, no una suma), pickup y espera ya
quedan en la MISMA magnitud de forma natural (ambos son un solo número de
minutos, no minutos-persona) -- no hace falta la ponderación por k que sí
hacía falta en H1/H2 para evitar que la suma (que crece con el tamaño de la
cola) aplastara al pickup. Es una forma distinta, más simple, de resolver el
mismo problema de "las dos magnitudes deben ser comparables".

**Qué se espera que pase (a verificar con el sweep, no se asume):** al no
premiar el TAMAÑO de la cola, H3 debería tratar a `kleppesto->laksevag` (una
sola persona muy vieja) igual de en serio que a un par con mucha gente
esperando lo mismo -- debería mejorar la espera máxima. El costo esperado:
menos eficiente en agregado que H2 (el tiempo medio/P95 podría empeorar un
poco, porque ya no prioriza "atender a más gente primero").

Arquitectura de asignación (global, por costo ascendente, con reserva de
capacidad para reposicionamiento, recalculada cada paso) idéntica a H1/H2 --
ver `politica_base/src/politica_h1.py` para la documentación completa de esas
piezas, no se repiten aquí.
"""
from __future__ import annotations

import pandas as pd

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def costo_par_h3(
    tiempo_pickup: float,
    cola_disponible: list[Unidad],
    t_actual_min: float,
    capacidad_barco: int,
) -> tuple[float, int] | None:
    """C(b,q,t) = tiempo_pickup - espera_max -- ver docstring del módulo.
    `espera_max` es la espera de la unidad MÁS ANTIGUA de la cola disponible
    (cabeza de la FIFO), no una suma sobre las k que se embarcarían. `k` se
    sigue devolviendo (se necesita para descontar la cola tras comprometer
    el par), pero no entra en el costo.
    """
    k = min(capacidad_barco, len(cola_disponible))
    if k == 0:
        return None
    espera_max = t_actual_min - cola_disponible[0].minuto_llegada
    costo = tiempo_pickup - espera_max
    return costo, k


def _cola_efectiva(
    par: tuple[str, str],
    estado: EstadoSimulacion,
    reservado: dict[tuple[str, str], int],
) -> list[Unidad]:
    cola = estado.colas.get(par, [])
    n_reservado = reservado.get(par, 0)
    return cola[n_reservado:] if n_reservado < len(cola) else []


def _reservas_por_reposicionamiento(
    estado: EstadoSimulacion,
    capacidad_barco: int,
) -> dict[tuple[str, str], int]:
    """Igual que en H1/H2 (recalculada cada paso, sin estado externo, barcos
    en reposicionamiento vacío procesados por ETA ascendente) -- usa
    `costo_par_h3` (pickup=0, ya está en X) para elegir el destino virtual.
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
            resultado = costo_par_h3(
                tiempo_pickup=0.0, cola_disponible=disponible,
                t_actual_min=estado.t_actual_min, capacidad_barco=capacidad_barco,
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


def asignar_flota_h3(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    cfg: dict | None = None,
) -> dict[str, str | None]:
    """Mismo algoritmo de asignación conjunta que H1/H2 (greedy global por
    costo ascendente, con reserva de capacidad) -- solo cambia `costo_par_h3`.
    Ver `politica_h1.asignar_flota_h1` para la documentación del algoritmo.
    """
    reservado = _reservas_por_reposicionamiento(estado, capacidad_barco)
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
                resultado = costo_par_h3(
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
