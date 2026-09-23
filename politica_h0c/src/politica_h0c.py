"""Heurística H0c -- H0 ("nearest-available", `politica_base.politica_base`)
con coordinación de verdad, SIN ninguna función de costo en minutos.

**Qué NO cambia respecto a H0:** la regla de decisión de cada barco es
EXACTAMENTE la misma que `politica_base.politica_base` -- prioridad estricta
a la demanda que sale de su propio nodo (si hay, se sirve la cola local con
la persona más antigua esperando, sin comparar contra nada remoto); si no
hay nada local, se reposiciona vacío hacia el nodo con la demanda remota más
urgente (mayor espera del más antiguo; empate por menor tiempo de viaje). No
hay minutos-persona, no hay suma de esperas, no hay pickup ponderado -- nada
de lo que trae H1/H2/H3. Es la MISMA heurística de H0.

**Qué SÍ cambia (los dos huecos de coordinación reales de H0, documentados
en `politica_base/README.md` sección 5):**

1. **Orden de decisión, de arbitrario a por urgencia.** `politica_base.asignar_flota`
   decide los barcos libres en el orden fijo de la lista (`estado.barcos`),
   descontando localmente lo que cada uno se llevaría -- pero SI dos barcos
   compiten indirectamente por la misma oportunidad, el orden de la lista
   (que no tiene ningún significado) decide quién la consigue. Acá, en cada
   ronda, se recalcula la mejor opción de CADA barco libre pendiente (local
   si tiene, si no remota) y se compromete primero la propuesta con MAYOR
   espera -- exactamente el mismo criterio de urgencia que ya usa H0 para
   elegir DENTRO de un barco, ahora usado también para decidir el ORDEN
   entre barcos. Sigue siendo la regla de H0 (local siempre gana sobre
   remoto, PARA CADA BARCO); lo único nuevo es el orden en que se resuelven
   los empates/competencia entre barcos.
2. **Reserva de capacidad entre pasos para reposicionamiento (H0 no tenía
   ninguna).** Un barco que ya viaja vacío hacia un nodo X (decidido en un
   paso ANTERIOR) no aparece de ninguna forma en la foto que ve
   `politica_base` para un barco que recién queda libre -- puede mandarlo,
   sin saberlo, a repetir el mismo viaje vacío. Acá se reconstruye, en cada
   paso, qué cola estaría cubriendo cada barco en reposicionamiento (con la
   MISMA regla remota de H0 -- mayor espera al llegar a X) y se descuenta esa
   capacidad de lo que ven los demás barcos libres -- recalculado cada paso,
   sin ningún estado que el bucle de control tenga que mantener (mismo
   mecanismo que H1/H2/H3, ver `politica_base/src/politica_h1.py`).

Con demanda local siempre resuelta antes que la remota, la única cola que
puede tener MÁS de un barco reservado/asignado en el mismo paso es una cola
remota muy grande -- el mismo descuento de capacidad (no de la cola entera)
que usan H1/H2/H3 aplica igual aquí.
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
    cola = estado.colas.get(par, [])
    n_reservado = reservado.get(par, 0)
    return cola[n_reservado:] if n_reservado < len(cola) else []


def _mejor_local(nodo: str, disponible: dict[tuple[str, str], list[Unidad]], t_actual_min: float):
    """Mejor cola que SALE de `nodo` (candidatos_directos de H0): la de
    mayor espera del más antiguo. `None` si no hay ninguna con gente.
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
    """Mejor cola que sale de OTRO nodo (candidatos_reposicion de H0): mayor
    espera del más antiguo; empate por menor tiempo de viaje desde `nodo`
    hasta el origen de esa cola -- misma regla exacta de
    `politica_base.politica_base`.
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


def _reservas_por_reposicionamiento(
    estado: EstadoSimulacion,
    capacidad_barco: int,
) -> dict[tuple[str, str], int]:
    """Igual mecanismo que H1/H2/H3 (recalculada cada paso, sin estado
    externo, barcos en reposicionamiento vacío procesados por ETA ascendente)
    -- pero usando la regla LOCAL de H0 (`_mejor_local`) anclada en el nodo
    destino X del barco: es exactamente lo que ese barco vería como "demanda
    de mi propio nodo" al llegar a X y quedar libre.
    """
    reservado: dict[tuple[str, str], int] = {}
    en_reposicionamiento = [b for b in estado.barcos if not b.libre and not b.a_bordo]
    en_reposicionamiento.sort(key=lambda b: b.min_para_llegar)

    disponible_temp: dict[tuple[str, str], list[Unidad]] = dict(estado.colas)
    for barco in en_reposicionamiento:
        X = barco.nodo_destino
        vista = {par: _cola_efectiva(par, estado, reservado) for par in estado.colas}
        mejor = _mejor_local(X, vista, estado.t_actual_min)
        if mejor is not None:
            _, par_reservado = mejor
            reservado[par_reservado] = reservado.get(par_reservado, 0) + capacidad_barco

    return reservado


def asignar_flota_h0c(
    barcos_libres: list[Barco],
    estado: EstadoSimulacion,
    matriz_tiempos: pd.DataFrame,
    capacidad_barco: int,
    cfg: dict | None = None,
) -> dict[str, str | None]:
    """Asignación conjunta con la regla EXACTA de H0 -- ver docstring del
    módulo para qué cambia (orden por urgencia + reservas) y qué no (la
    regla de decisión en sí).

    Cada ronda: cada barco libre pendiente propone su mejor opción aplicando
    la regla de H0 (local si tiene algo, si no remoto); se compromete la
    propuesta de MAYOR espera entre todos los barcos pendientes; se descuenta
    esa cola (hasta la capacidad del barco) de lo que ven los demás; se
    repite. Los barcos sin ninguna propuesta factible al final: esperan.
    """
    reservado = _reservas_por_reposicionamiento(estado, capacidad_barco)
    disponible: dict[tuple[str, str], list[Unidad]] = {
        par: _cola_efectiva(par, estado, reservado) for par in estado.colas
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

    for barco in pendientes:
        decisiones[barco.id] = None

    return decisiones
