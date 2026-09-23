"""H2 = H1 + costo, MANTENIENDO la partición local/remoto de H0 (Alternativa
B de la auditoría). Una cola remota NUNCA compite directamente con una
local -- el costo solo se usa para elegir DENTRO de cada nivel.

    C(b,q) = T_pickup(b,q) - W_max(q)

Regla por barco:

    1. Si hay demanda local (sale del nodo del barco): comparar SOLO las
       colas locales mediante C, elegir la de menor costo.
       -> Para candidatos locales, T_pickup = 0 SIEMPRE, así que
          C = -W_max: minimizar C equivale exactamente a maximizar W_max --
          es decir, esta rama es matemáticamente IDÉNTICA a la Regla 1 de H0
          (mayor espera del más antiguo). No hay nada nuevo que decidir
          aquí -- se reutiliza `_mejor_local` de H1 tal cual, sin
          reimplementar el costo para este caso, porque daría lo mismo.
    2. Si no hay demanda local: comparar SOLO las colas remotas mediante C
       -- aquí sí es donde el costo aporta algo distinto de H0: en vez de
       "mayor espera, empate por menor viaje" (regla lexicográfica de H0),
       hace un trade-off continuo entre viaje y espera. Una cola remota algo
       menos urgente pero mucho más cercana puede ganarle a una más urgente
       pero muy lejana -- cosa que H0/H1 nunca permiten (su empate por
       viaje solo actúa si las esperas son EXACTAMENTE iguales).
    3. Si no hay nada: esperar.

Reserva persistente: igual mecanismo que H1 (`heuristicas/comun/src/reservas.py`),
sin cambios -- H2 es H1 con esta única extensión.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "comun" / "src")))
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "h1_reserva" / "src")))
from reservas import liberar_vencidas, cola_efectiva  # noqa: E402
from politica_h1 import _mejor_local  # noqa: E402 -- reutilizado tal cual, ver docstring

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def _mejor_remoto_costo(
    nodo: str,
    disponible: dict[tuple[str, str], list[Unidad]],
    t_actual_min: float,
    matriz_tiempos: pd.DataFrame,
):
    """Entre las colas que salen de OTRO nodo, la de MENOR C = T_pickup -
    W_max (no la de mayor espera con empate por viaje, como en H0/H1 -- ver
    docstring del módulo). `None` si ninguna tiene gente.
    """
    mejor: tuple[float, float, tuple[str, str]] | None = None  # (costo, espera, par)
    for par, cola in disponible.items():
        o, _ = par
        if o == nodo or not cola:
            continue
        espera_max = t_actual_min - cola[0].minuto_llegada
        pickup = float(matriz_tiempos.loc[nodo, o])
        costo = pickup - espera_max
        if mejor is None or costo < mejor[0]:
            mejor = (costo, espera_max, par)
    return mejor


def asignar_flota_h2(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    reservas: dict[str, tuple[str, str]],
    cfg: dict | None = None,
) -> tuple[dict[str, str | None], dict[str, tuple[str, str]]]:
    """Misma estructura de bucle que `politica_h1.asignar_flota_h1` (orden
    de resolución entre barcos por urgencia, reserva persistente para
    reposicionamiento) -- la única diferencia es `_mejor_remoto_costo` en
    vez de `_mejor_remoto`, y que la fase local nunca compite con la remota
    (cada barco propone SOLO de su rama correspondiente, nunca ambas).
    """
    reservas = liberar_vencidas(reservas, estado)
    disponible: dict[tuple[str, str], list[Unidad]] = {
        par: cola_efectiva(par, estado, reservas, capacidad_barco) for par in estado.colas
    }

    pendientes = list(barcos_libres)
    decisiones: dict[str, str | None] = {}

    while pendientes:
        propuestas = []
        for barco in pendientes:
            A = barco.nodo_origen
            local = _mejor_local(A, disponible, estado.t_actual_min)
            if local is not None:
                espera, par = local
                propuestas.append((espera, barco, "local", par))
                continue
            remoto = _mejor_remoto_costo(A, disponible, estado.t_actual_min, matriz_tiempos)
            if remoto is not None:
                _, espera, par = remoto
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
            reservas[barco_elegido.id] = par_elegido

    for barco in pendientes:
        decisiones[barco.id] = None

    return decisiones, reservas
