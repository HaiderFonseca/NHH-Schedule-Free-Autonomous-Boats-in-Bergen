"""Métricas de una corrida (docs/especificacion_simulador_rl.md §12, paso 2).

Todas las funciones aquí son de ANÁLISIS -- consumen lo que el `env` ya
registró durante la corrida (`log_eventos`, `log_recompensa`,
`historial_estados`, `atendidas_historico`), nunca al revés: el
agente/política no ve nada de este módulo, solo el vector aplanado de
`estado.aplanar_estado`.

**Sin "perdidas" (ver `env.py`/`unidades.py`):** el simulador ya no retira
a nadie por paciencia -- toda persona generada termina "atendida" (subió a
un barco y llegó a destino) o, si la corrida se corta antes de que le
tocara, "sin atender al final" (todavía esperando en una cola, o a bordo
de un barco que no había llegado). Por eso las métricas de este módulo ya
no tienen una columna de "perdidas"; en cambio, los percentiles de espera
de `metricas_por_usuario` son la pieza clave para definir una garantía de
tiempo -- son datos REALES, sin censurar (nadie se "borró" antes de que se
supiera cuánto iba a esperar).

`reporte_completo(env)` es el punto de entrada único para los notebooks --
junta todo lo demás, incluida `verificar_conservacion`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from estado import pares_od


def verificar_conservacion(env) -> dict:
    """Toda persona/grupo generado debe terminar en un estado contable:
    atendida, o todavía esperando/a bordo si la corrida se truncó a mitad
    de camino (nadie se pierde -- ver docstring del módulo). Si `cuadra` es
    False es un bug real que hay que exponer, no esconder.
    """
    generadas = env.total_unidades_generadas
    atendidas = len(env.atendidas_historico)
    esperando = sum(len(cola) for cola in env.colas.values())
    a_bordo = sum(len(b.a_bordo) for b in env.barcos)
    suma = atendidas + esperando + a_bordo
    return {
        "generadas": generadas,
        "atendidas": atendidas,
        "esperando_al_final": esperando,
        "a_bordo_al_final": a_bordo,
        "suma": suma,
        "cuadra": suma == generadas,
    }


def _tiempos_viaje_por_unidad(env) -> dict[str, float]:
    """unidad_id -> tiempo de viaje (minutos a bordo), de los eventos
    'baja'. Precomputado una vez y reusado -- evita recorrer log_eventos
    por cada unidad.
    """
    return {e["unidad_id"]: e["tiempo_viaje"] for e in env.log_eventos if e["tipo"] == "baja"}


def _tiempos_sistema(env) -> dict[str, float]:
    """unidad_id -> tiempo total en el sistema (espera + viaje), solo para
    las unidades atendidas -- las que aún no fueron atendidas al final de
    la corrida no tienen un tiempo en sistema "cerrado" todavía.
    """
    viajes = _tiempos_viaje_por_unidad(env)
    return {u.id: (u.tiempo_espera_min or 0.0) + viajes.get(u.id, 0.0) for u in env.atendidas_historico}


def metricas_por_par(env) -> pd.DataFrame:
    """generadas, atendidas, sin atender al final, % atendidas, espera
    media/máxima, tiempo de viaje medio, tiempo en sistema medio/máximo --
    por cada uno de los 12 pares origen-destino. La espera aquí es REAL,
    sin censurar por paciencia (ver docstring del módulo).
    """
    viajes = _tiempos_viaje_por_unidad(env)
    sistema = _tiempos_sistema(env)

    filas = []
    for par in pares_od(env.nodos):
        atendidas = [u for u in env.atendidas_historico if (u.origen, u.destino) == par]
        sin_atender = [u for cola_par, cola in env.colas.items() for u in cola if cola_par == par]
        sin_atender += [u for b in env.barcos for u in b.a_bordo if (u.origen, u.destino) == par]
        generadas = len(atendidas) + len(sin_atender)

        esperas = [u.tiempo_espera_min for u in atendidas if u.tiempo_espera_min is not None]
        viajes_par = [viajes[u.id] for u in atendidas if u.id in viajes]
        sistema_par = [sistema[u.id] for u in atendidas if u.id in sistema]

        filas.append({
            "par": f"{par[0]}->{par[1]}",
            "generadas": generadas,
            "atendidas": len(atendidas),
            "sin_atender_al_final": len(sin_atender),
            "pct_atendidas": 100.0 * len(atendidas) / generadas if generadas else float("nan"),
            "espera_media_min": float(np.mean(esperas)) if esperas else float("nan"),
            "espera_maxima_min": float(np.max(esperas)) if esperas else float("nan"),
            "viaje_medio_min": float(np.mean(viajes_par)) if viajes_par else float("nan"),
            "sistema_medio_min": float(np.mean(sistema_par)) if sistema_par else float("nan"),
            "sistema_maximo_min": float(np.max(sistema_par)) if sistema_par else float("nan"),
        })
    return pd.DataFrame(filas)


def metricas_globales(env) -> dict:
    generadas = env.total_unidades_generadas
    atendidas = len(env.atendidas_historico)
    esperas = [u.tiempo_espera_min for u in env.atendidas_historico if u.tiempo_espera_min is not None]
    sistema = list(_tiempos_sistema(env).values())
    return {
        "unidades_generadas": generadas,
        "unidades_atendidas": atendidas,
        "unidades_sin_atender_al_final": generadas - atendidas,
        "pct_atendidas": 100.0 * atendidas / generadas if generadas else float("nan"),
        "espera_media_min": float(np.mean(esperas)) if esperas else float("nan"),
        "sistema_medio_min": float(np.mean(sistema)) if sistema else float("nan"),
        "sistema_maximo_min": float(np.max(sistema)) if sistema else float("nan"),
    }


@dataclass
class CorridaCombinada:
    """Junta varias corridas independientes (mismos nodos/flota -- p.ej. los
    7 días de una semana, ver `combinar_corridas`) en un solo objeto con la
    misma interfaz que el resto de este módulo espera de un `env`, para
    poder llamar `reporte_completo` sobre la semana entera sin duplicar
    ninguna lógica de métricas.
    """

    nodos: list[str]
    num_barcos: int
    capacidad_barco: int
    paso_tiempo_min: float
    total_unidades_generadas: int
    atendidas_historico: list
    colas: dict
    barcos: list
    log_eventos: list
    log_recompensa: list
    historial_estados: list


def combinar_corridas(envs: list) -> CorridaCombinada:
    """Junta N corridas independientes (mismos nodos/flota -- p.ej. una por
    día de una semana, `notebooks/03_escalon_3_semana.ipynb`) sumando sus
    resultados. Cada corrida es su propio episodio (su propio reloj
    0..hora_fin_min) -- aquí no se arma un timeline único, solo se
    concatenan/suman las listas y conteos que las funciones de arriba ya
    saben leer.
    """
    if not envs:
        raise ValueError("combinar_corridas necesita al menos una corrida")
    nodos, num_barcos = envs[0].nodos, envs[0].num_barcos
    for e in envs:
        if e.nodos != nodos or e.num_barcos != num_barcos:
            raise ValueError("combinar_corridas espera corridas con los mismos nodos/flota")

    colas: dict[tuple[str, str], list] = {par: [] for par in pares_od(nodos)}
    for e in envs:
        for par, cola in e.colas.items():
            colas[par].extend(cola)

    # Solo se necesita `.a_bordo` de cada barco (lo único que lee este
    # módulo) -- se combina por índice, no hace falta un `Barco` real.
    barcos = [
        SimpleNamespace(a_bordo=[u for e in envs for u in e.barcos[i].a_bordo])
        for i in range(num_barcos)
    ]

    return CorridaCombinada(
        nodos=nodos,
        num_barcos=num_barcos,
        capacidad_barco=envs[0].capacidad_barco,
        paso_tiempo_min=envs[0].paso_tiempo_min,
        total_unidades_generadas=sum(e.total_unidades_generadas for e in envs),
        atendidas_historico=[u for e in envs for u in e.atendidas_historico],
        colas=colas,
        barcos=barcos,
        log_eventos=[ev for e in envs for ev in e.log_eventos],
        log_recompensa=[r for e in envs for r in e.log_recompensa],
        historial_estados=[f for e in envs for f in e.historial_estados],
    )


def metricas_por_barco(env) -> pd.DataFrame:
    """movimientos, tiempo navegado, ocupación media/máxima, % ocioso vs.
    navegando, ruta más frecuente -- por cada barco de la flota. La
    ocupación/estado a lo largo del tiempo se lee de `env.historial_estados`
    (un snapshot por paso, mismo orden de barcos que `env.barcos`).
    """
    filas = []
    for barco_idx in range(env.num_barcos):
        barco_id = f"barco_{barco_idx}"
        movimientos = [e for e in env.log_eventos if e["tipo"] == "movimiento" and e["barco_id"] == barco_id]
        rutas = Counter((e["desde"], e["hasta"]) for e in movimientos)
        ruta_top = max(rutas.items(), key=lambda kv: kv[1]) if rutas else None

        ocupaciones = [frame["barcos"][barco_idx]["ocupacion"] for frame in env.historial_estados]
        pasos_navegando = sum(1 for frame in env.historial_estados if not frame["barcos"][barco_idx]["libre"])
        pasos_totales = len(env.historial_estados)

        filas.append({
            "barco_id": barco_id,
            "movimientos": len(movimientos),
            "tiempo_navegado_min": pasos_navegando * env.paso_tiempo_min,
            "ocupacion_media": float(np.mean(ocupaciones)) if ocupaciones else float("nan"),
            "ocupacion_maxima": int(np.max(ocupaciones)) if ocupaciones else 0,
            "pct_ocioso": 100.0 * (1 - pasos_navegando / pasos_totales) if pasos_totales else float("nan"),
            "ruta_mas_frecuente": f"{ruta_top[0][0]}->{ruta_top[0][1]} ({ruta_top[1]}x)" if ruta_top else "-",
        })
    return pd.DataFrame(filas)


def metricas_por_usuario(env) -> dict:
    """Distribución (percentiles) de espera, tiempo de viaje, y tiempo en
    sistema, sobre las unidades ATENDIDAS -- datos reales, sin censurar por
    ninguna paciencia artificial. Esta es la pieza clave para definir una
    garantía de tiempo con fundamento: por ejemplo, si p95 de espera es 22
    min, "servir al 95% de la gente en 22 min" es una garantía medida, no
    supuesta.
    """
    esperas = [u.tiempo_espera_min for u in env.atendidas_historico if u.tiempo_espera_min is not None]
    viajes = list(_tiempos_viaje_por_unidad(env).values())
    sistema = list(_tiempos_sistema(env).values())

    def percentiles(valores):
        if not valores:
            return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
        arr = np.array(valores)
        return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)), "max": float(np.max(arr))}

    return {"espera_min": percentiles(esperas), "viaje_min": percentiles(viajes), "sistema_min": percentiles(sistema)}


def sin_atender_al_final_por_par(env) -> pd.DataFrame:
    """Dónde queda gente sin atender cuando termina la corrida (todavía en
    cola, o a bordo de un barco que no había llegado) -- backlog por par,
    no "pérdidas": es gente que se habría atendido si la corrida siguiera.
    """
    conteos: dict[tuple[str, str], int] = {}
    for par, cola in env.colas.items():
        if cola:
            conteos[par] = conteos.get(par, 0) + len(cola)
    for b in env.barcos:
        for u in b.a_bordo:
            par = (u.origen, u.destino)
            conteos[par] = conteos.get(par, 0) + 1

    if not conteos:
        return pd.DataFrame(columns=["par", "sin_atender"])
    filas = [{"par": f"{par[0]}->{par[1]}", "sin_atender": n} for par, n in conteos.items()]
    return pd.DataFrame(filas).sort_values("sin_atender", ascending=False).reset_index(drop=True)


def desglose_recompensa(env) -> pd.DataFrame:
    """Serie de tiempo del desglose de recompensa (incomodidad, movimiento,
    total) por paso -- ya se calculaba en cada `step()`, esta función solo
    la devuelve como tabla para graficar/sumar.
    """
    return pd.DataFrame(env.log_recompensa)


def reporte_completo(env) -> dict:
    """Punto de entrada único: junta todas las métricas + la verificación
    de conservación. Si `conservacion["cuadra"]` es False, se imprime una
    advertencia explícita (no se esconde el problema).
    """
    conservacion = verificar_conservacion(env)
    if not conservacion["cuadra"]:
        print("ADVERTENCIA: la conservación de personas NO cuadra -- revisar el simulador.")
        print(conservacion)

    return {
        "conservacion": conservacion,
        "globales": metricas_globales(env),
        "por_par": metricas_por_par(env),
        "por_barco": metricas_por_barco(env),
        "por_usuario": metricas_por_usuario(env),
        "sin_atender_al_final": sin_atender_al_final_por_par(env),
        "desglose_recompensa": desglose_recompensa(env),
    }
