"""H3 = H1 + costo GLOBAL (Alternativa A de la auditoría). Elimina la
partición local/remoto de H0: una cola remota SÍ puede competir
directamente con una local si su combinación de espera y pickup resulta más
urgente.

    C(b,q) = T_pickup(b,q) - W_max(q)

    candidatos = TODAS las colas con gente esperando (locales Y remotas)
    -> elegir globalmente, entre TODOS los barcos libres y TODAS las colas,
       la asignación conjunta de menor costo (greedy: se toma el par de
       menor C, se compromete, se descuenta, se repite).

Para un candidato LOCAL, T_pickup=0, así que C=-W_max -- mismo valor que en
H1/H2 para lo local. La diferencia real de H3 es que este valor SÍ se
compara, en pie de igualdad, contra el costo de cualquier candidato remoto
(en H1/H2 nunca se comparan entre sí). El umbral exacto de cuándo lo remoto
gana está caracterizado en la auditoría (Caso 4): sucede cuando

    espera_remota > T_pickup_remoto + espera_local

es decir, cuando la ventaja de espera de lo remoto supera el costo de ir a
buscarlo -- no por diferencias triviales.

Reserva persistente: igual mecanismo que H1/H2 (`heuristicas/comun/src/reservas.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "comun" / "src")))
from reservas import liberar_vencidas, cola_efectiva  # noqa: E402

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def costo_par_h3(tiempo_pickup: float, cola_disponible: list[Unidad], t_actual_min: float) -> float | None:
    """C(b,q) = T_pickup - W_max. `None` si la cola está vacía."""
    if not cola_disponible:
        return None
    espera_max = t_actual_min - cola_disponible[0].minuto_llegada
    return tiempo_pickup - espera_max


def asignar_flota_h3(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    reservas: dict[str, tuple[str, str]],
    cfg: dict | None = None,
) -> tuple[dict[str, str | None], dict[str, tuple[str, str]]]:
    """Asignación global de mínimo costo (greedy: par de menor C primero,
    descuenta, repite) -- sin ninguna restricción de tier. Reserva
    persistente creada solo para decisiones de reposicionamiento (par con
    origen != nodo actual del barco), igual que en H1/H2.
    """
    reservas = liberar_vencidas(reservas, estado)
    disponible: dict[tuple[str, str], list[Unidad]] = {
        par: cola_efectiva(par, estado, reservas, capacidad_barco) for par in estado.colas
    }

    pendientes = list(barcos_libres)
    decisiones: dict[str, str | None] = {}

    while pendientes:
        mejor: tuple[float, Barco, tuple[str, str]] | None = None
        for barco in pendientes:
            A = barco.nodo_origen
            for par, cola in disponible.items():
                o, d = par
                pickup = float(matriz_tiempos.loc[A, o])
                costo = costo_par_h3(pickup, cola, estado.t_actual_min)
                if costo is None:
                    continue
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
        if o_par != A:
            reservas[barco_elegido.id] = par_elegido

    for barco in pendientes:
        decisiones[barco.id] = None

    return decisiones, reservas
