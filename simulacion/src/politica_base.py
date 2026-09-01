"""Política de referencia -- nearest-available (docs/especificacion_simulador_rl.md §9).

Sin aprendizaje: regla fija. Firma intercambiable a propósito
(`politica_base(barco, estado, matriz_tiempos, cfg) -> nodo | None`) para
que la fase siguiente sustituya esto por el agente sin tocar `env.py` -- el
bucle de control (no el `Env`) es quien llama a esta función por cada barco
libre en cada paso.

**Viajes directos punto a punto (ver estado.py/env.py):** un barco libre en
A que recibe la orden "ir a B" solo puede embarcar gente de la cola exacta
`(A,B)` -- no hay pasajeros a bordo con destinos mixtos, así que un barco
libre siempre está vacío (`a_bordo == []`). Esto simplifica la política
respecto a una versión anterior que trataba a los pasajeros a bordo como
candidatos: ya no hace falta, porque nunca hay nadie a bordo de un barco
libre.

**Dos tipos de candidato:**
1. Demanda que sale de A (servible de inmediato: el barco embarca y viaja).
2. Demanda que sale de otro nodo X != A (no servible de inmediato -- el
   barco tendría que reposicionarse vacío hasta X primero, y solo en su
   PRÓXIMO momento libre, ya en X, podría embarcarla). Es una decisión
   miope (greedy): no planea las dos etapas de una vez, vuelve a evaluar
   con información fresca cuando el barco llegue a X.

La demanda directa (tipo 1) tiene prioridad estricta sobre el
reposicionamiento (tipo 2): solo se considera reposicionarse si no hay
absolutamente nadie esperando para salir de A.
"""
from __future__ import annotations

import pandas as pd

from estado import Barco, EstadoSimulacion


def politica_base(
    barco: Barco,
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    cfg: dict | None = None,
) -> str | None:
    """Decide el próximo nodo de un barco libre. `None` = esperar donde está
    (nadie esperando en ningún par, o nada alcanzable a tiempo).
    """
    A = barco.nodo_origen  # == nodo_destino: está libre, se le puede pedir una nueva acción

    # 1) Demanda que sale de A: servible de inmediato. Se elige el destino
    #    B tal que la unidad más antigua en cola[(A,B)] es la más urgente
    #    entre todos los B posibles.
    candidatos_directos = []
    for (o, d), cola in estado.colas.items():
        if o != A or not cola:
            continue
        mas_antigua = cola[0]  # las colas se mantienen en orden de llegada (FIFO)
        tiempo_esperando = estado.t_actual_min - mas_antigua.minuto_llegada
        candidatos_directos.append((tiempo_esperando, d))

    if candidatos_directos:
        candidatos_directos.sort(key=lambda c: -c[0])
        return candidatos_directos[0][1]

    # 2) Nada que hacer en A: considerar reposicionarse vacío hacia el nodo
    #    X con la demanda más urgente en cualquier otro par (X, *), siempre
    #    que el barco alcance a llegar antes de que esa unidad se pierda.
    candidatos_reposicion = []
    for (o, d), cola in estado.colas.items():
        if o == A or not cola:
            continue
        mas_antigua = cola[0]
        tiempo_esperando = estado.t_actual_min - mas_antigua.minuto_llegada
        paciencia_restante = mas_antigua.paciencia_min - tiempo_esperando
        tiempo_viaje = float(matriz_tiempos.loc[A, o])
        if tiempo_viaje <= paciencia_restante:
            candidatos_reposicion.append((tiempo_esperando, tiempo_viaje, o))

    if not candidatos_reposicion:
        return None

    # más tiempo esperando primero; empate -> menor tiempo de viaje (más cercano)
    candidatos_reposicion.sort(key=lambda c: (-c[0], c[1]))
    return candidatos_reposicion[0][2]


def asignar_flota(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    cfg: dict | None = None,
) -> dict[str, str | None]:
    """Aplica `politica_base` a varios barcos libres EN EL MISMO PASO, uno a
    la vez, con una copia local de las colas que se va descontando -- para
    que dos barcos libres a la vez no elijan ambos "ir a B" pensando que
    hay gente esperando ahí, y el segundo llegue vacío porque el primero ya
    se la llevó toda.

    Sin esto (llamar a `politica_base` por separado para cada barco, todos
    mirando la MISMA foto del mundo) se detectó exactamente ese problema
    durante la verificación: con 2 barcos libres a la vez en el mismo nodo,
    ambos decidían ir al mismo destino, y el segundo viajaba vacío
    (`ocupacion=0`) -- un movimiento desperdiciado, con su propia
    penalización, sin servir a nadie. `asignar_flota` es la forma
    correcta/recomendada de usar la política con más de un barco; se deja
    también `politica_base` expuesta y usable sola (p.ej. para un solo
    barco, o como referencia de la regla base sin la capa de coordinación).
    """
    colas_restantes = {par: list(cola) for par, cola in estado.colas.items()}

    class _EstadoLocal:
        def __init__(self, colas):
            self.t_actual_min = estado.t_actual_min
            self.colas = colas

    decisiones: dict[str, str | None] = {}
    for barco in barcos_libres:
        estado_local = _EstadoLocal(colas_restantes)
        decision = politica_base(barco, estado_local, matriz_tiempos, cfg)
        decisiones[barco.id] = decision
        if decision is None or decision == barco.nodo_origen:
            continue
        # Descuenta localmente lo que ESTE barco se llevaría, hasta su
        # capacidad, para que el siguiente barco de la lista vea la cola ya
        # reducida -- misma regla FIFO/capacidad que `env._embarcar_para_viaje`.
        par = (barco.nodo_origen, decision)
        cola = colas_restantes[par]
        espacio = capacidad_barco - barco.ocupacion
        restante, tomado = [], 0
        for u in cola:
            if tomado + u.tamano <= espacio:
                tomado += u.tamano
            else:
                restante.append(u)
        colas_restantes[par] = restante

    return decisiones
