"""H1 = H0 + reserva persistente de cola. SIN función de costo, SIN pesos,
SIN forecasting, SIN cambiar la lógica de boarding/capacidad (eso lo sigue
haciendo únicamente `simulador/src/env.py`).

Las reglas de decisión de un barco son EXACTAMENTE las de
`politica_base.politica_base` (no se reescriben aquí -- se reimplementan tal
cual, verificado línea a línea contra el original, porque la política de H0
decide un barco a la vez y H1 necesita evaluar varios barcos libres juntos
para poder ordenar por urgencia -- ver `asignar_flota_h1`):

    1. Demanda local (sale del nodo donde el barco ya está): si existe,
       elegir la de mayor espera del más antiguo. Prioridad estricta.
    2. Si no hay demanda local: demanda remota, mayor espera del más
       antiguo; empate -> menor tiempo de viaje.
    3. Si no hay nada: esperar (None).

La ÚNICA extensión conceptual sobre H0 es la reserva persistente
(`heuristicas/comun/src/reservas.py`) -- ver ese módulo para la definición
completa de creación/persistencia/liberación.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "comun" / "src")))
from reservas import liberar_vencidas, cola_efectiva  # noqa: E402

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def _mejor_local(nodo: str, disponible: dict[tuple[str, str], list[Unidad]], t_actual_min: float):
    """Candidatos que SALEN de `nodo` -- Regla 1 de H0. `None` si ninguno
    tiene gente esperando.
    """
    mejor: tuple[float, tuple[str, str]] | None = None
    for par, cola in disponible.items():
        o, _ = par
        if o != nodo or not cola:
            continue
        espera = t_actual_min - cola[0].minuto_llegada
        if mejor is None or espera > mejor[0]:
            mejor = (espera, par)
    return mejor


def _mejor_remoto(
    nodo: str,
    disponible: dict[tuple[str, str], list[Unidad]],
    t_actual_min: float,
    matriz_tiempos: pd.DataFrame,
):
    """Candidatos que salen de OTRO nodo -- Regla 2 de H0: mayor espera;
    empate por menor tiempo de viaje desde `nodo`.
    """
    mejor: tuple[float, float, tuple[str, str]] | None = None
    for par, cola in disponible.items():
        o, _ = par
        if o == nodo or not cola:
            continue
        espera = t_actual_min - cola[0].minuto_llegada
        tiempo_viaje = float(matriz_tiempos.loc[nodo, o])
        candidato = (espera, tiempo_viaje, par)
        if mejor is None or (candidato[0], -candidato[1]) > (mejor[0], -mejor[1]):
            mejor = candidato
    return mejor


def asignar_flota_h1(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    reservas: dict[str, tuple[str, str]],
    cfg: dict | None = None,
) -> tuple[dict[str, str | None], dict[str, tuple[str, str]]]:
    """Firma DISTINTA de `politica_base.asignar_flota`: recibe y devuelve
    `reservas` (el diccionario persistente, mantenido por el bucle de
    control -- ver `heuristicas/comun/src/reservas.py`). H0 no cambia su
    firma porque H0 no tiene reserva.

    Algoritmo: en cada ronda, cada barco libre pendiente propone su mejor
    opción con la regla EXACTA de H0 (local si existe, si no remoto); se
    compromete la propuesta de MAYOR espera entre todas (orden por urgencia,
    no por índice de lista -- la mejora real de H1 sobre la coordinación de
    H0 dentro de un mismo paso, ver auditoría); se descuenta esa cola; se
    repite. Si la decisión comprometida es de tipo "remoto" (reposicionamiento
    vacío), se crea la entrada `reservas[barco_id] = par` para que persista
    en los próximos pasos hasta que el barco llegue.
    """
    reservas = liberar_vencidas(reservas, estado)
    disponible: dict[tuple[str, str], list[Unidad]] = {
        par: cola_efectiva(par, estado, reservas, capacidad_barco) for par in estado.colas
    }

    pendientes = list(barcos_libres)
    decisiones: dict[str, str | None] = {}

    while pendientes:
        propuestas = []  # (espera, barco, tipo, par)
        for barco in pendientes:
            A = barco.nodo_origen
            local = _mejor_local(A, disponible, estado.t_actual_min)
            if local is not None:
                espera, par = local
                propuestas.append((espera, barco, "local", par))
                continue
            remoto = _mejor_remoto(A, disponible, estado.t_actual_min, matriz_tiempos)
            if remoto is not None:
                espera, _, par = remoto
                propuestas.append((espera, barco, "remoto", par))

        if not propuestas:
            break

        propuestas.sort(key=lambda p: -p[0])
        _, barco_elegido, tipo, par_elegido = propuestas[0]
        o_par, d_par = par_elegido
        decisiones[barco_elegido.id] = d_par if tipo == "local" else o_par
        pendientes.remove(barco_elegido)
        disponible[par_elegido] = disponible[par_elegido][capacidad_barco:]
        if tipo == "remoto":
            reservas[barco_elegido.id] = par_elegido  # CREAR la reserva persistente

    for barco in pendientes:
        decisiones[barco.id] = None

    return decisiones, reservas
