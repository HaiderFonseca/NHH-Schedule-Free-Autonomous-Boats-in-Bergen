"""Reserva persistente O-D por barco -- pieza compartida por H1, H2 y H3
(definición cerrada tras la auditoría y los 5 casos controlados).

    reservas: dict[barco_id, (origen, destino)]

Es un diccionario LITERAL, creado por el barco cuando se compromete con una
cola remota (se reposiciona vacío hacia ella), y mantenido **por el bucle de
control** de un paso al siguiente -- no se recalcula desde cero en cada
llamada (a diferencia de la versión anterior de este trabajo,
`politica_h0c/src/politica_h0c.py`, que sí recalculaba). El bucle de control
debe guardar el diccionario que devuelve `asignar_flota_hX` y pasárselo de
vuelta en la siguiente llamada, exactamente como ya hace con `obs, info` de
`env.step()`:

    reservas = {}
    while True:
        ...
        decisiones, reservas = asignar_flota_h1(libres, estado, matriz_tiempos, capacidad, reservas)
        ...

**Por qué solo hace falta reservar en el caso de reposicionamiento (nunca en
demanda local):** un barco libre en A que decide servir una cola local
embarca de inmediato (el motor, `env._embarcar_para_viaje`, sube gente en el
mismo paso en que el barco parte) -- no queda nada pendiente que proteger. Un
barco que decide reposicionarse vacío hacia X, en cambio, tarda varios pasos
en llegar y no ha embarcado a nadie todavía -- ES ese compromiso, todavía sin
cumplir, el que hay que proteger de que otro barco libre, en un paso
posterior, decida ir también.

**Liberación -- un solo evento posible, no tres, dado cómo funciona el
motor:** la especificación original hablaba de liberar cuando (1) la cola
fue atendida, (2) quedó demanda pendiente que excede la capacidad del barco,
o (3) el compromiso deja de ser viable. En este simulador, un barco EN
TRÁNSITO nunca se redirige a medio camino (`simulador/src/env.py`, `step()`:
"Ignorado para barcos ocupados... la especificación no permite redirigir a
media ruta") y no existe ningún mecanismo de cancelación -- así que el caso
(3) es estructuralmente imposible antes de llegar, y el caso (2) no es un
evento de liberación sino algo que ya está resuelto por CÓMO se reserva (se
reserva la capacidad del barco, no la cola entera -- ver `cola_efectiva`,
así que el excedente por encima de esa capacidad queda visible para otro
barco desde el primer momento, sin esperar ninguna liberación). El único
evento real es el (1): el barco llega, queda libre, y en ESE momento se
libera su entrada -- lo hace `liberar_vencidas`, llamada al principio de
cada `asignar_flota_hX`.
"""
from __future__ import annotations


def liberar_vencidas(reservas: dict[str, tuple[str, str]], estado) -> dict[str, tuple[str, str]]:
    """Quita del diccionario cualquier reserva cuyo barco ya volvió a estar
    libre (llegó a destino). Devuelve un dict NUEVO (no muta el argumento).
    """
    barcos_por_id = {b.id: b for b in estado.barcos}
    return {
        barco_id: par
        for barco_id, par in reservas.items()
        if barco_id in barcos_por_id and not barcos_por_id[barco_id].libre
    }


def cola_efectiva(
    par: tuple[str, str],
    estado,
    reservas: dict[str, tuple[str, str]],
    capacidad_barco: int,
):
    """Unidades de `estado.colas[par]` que quedan disponibles después de
    descontar la capacidad reservada por TODOS los barcos actualmente
    comprometidos con ese mismo par (puede ser más de uno, si la cola es
    grande) -- se descuentan las más antiguas primero (FIFO), igual que
    embarcaría el motor.
    """
    cola = estado.colas.get(par, [])
    reservado = sum(capacidad_barco for p in reservas.values() if p == par)
    return cola[reservado:] if reservado < len(cola) else []
