"""Métricas de una corrida (docs/especificacion_simulador_rl.md §12, paso 2).

Todas las funciones aquí son de ANÁLISIS -- consumen lo que el `env` ya
registró durante la corrida (`log_eventos`, `log_recompensa`,
`historial_estados`, `atendidas_historico`, `perdidas_historico`), nunca al
revés: el agente/política no ve nada de este módulo, solo el vector
aplanado de `estado.aplanar_estado`.

`reporte_completo(env)` es el punto de entrada único para los notebooks --
junta todo lo demás, incluida `verificar_conservacion`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from estado import pares_od


def verificar_conservacion(env) -> dict:
    """Toda persona/grupo generado debe terminar en un estado contable:
    atendida, perdida, o todavía esperando/a bordo si la corrida se truncó
    a mitad de camino. Si `cuadra` es False es un bug real que hay que
    exponer, no esconder -- se reporta explícito, no se silencia.
    """
    generadas = env.total_unidades_generadas
    atendidas = len(env.atendidas_historico)
    perdidas = len(env.perdidas_historico)
    esperando = sum(len(cola) for cola in env.colas.values())
    a_bordo = sum(len(b.a_bordo) for b in env.barcos)
    suma = atendidas + perdidas + esperando + a_bordo
    return {
        "generadas": generadas,
        "atendidas": atendidas,
        "perdidas": perdidas,
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
    """unidad_id -> tiempo total en el sistema (espera + viaje si fue
    atendida; solo espera si se perdió -- nunca llegó a viajar).
    """
    viajes = _tiempos_viaje_por_unidad(env)
    resultado = {}
    for u in env.atendidas_historico:
        espera = u.tiempo_espera_min or 0.0
        resultado[u.id] = espera + viajes.get(u.id, 0.0)
    for u in env.perdidas_historico:
        resultado[u.id] = u.tiempo_espera_min or 0.0
    return resultado


def metricas_por_par(env) -> pd.DataFrame:
    """generadas, atendidas, perdidas, % cumplimiento, espera media/máxima,
    tiempo de viaje medio, tiempo en sistema medio/máximo -- por cada uno de
    los 12 pares origen-destino.
    """
    viajes = _tiempos_viaje_por_unidad(env)
    sistema = _tiempos_sistema(env)

    filas = []
    for par in pares_od(env.nodos):
        atendidas = [u for u in env.atendidas_historico if (u.origen, u.destino) == par]
        perdidas = [u for u in env.perdidas_historico if (u.origen, u.destino) == par]
        generadas = len(atendidas) + len(perdidas)
        esperas = [u.tiempo_espera_min for u in atendidas + perdidas if u.tiempo_espera_min is not None]
        viajes_par = [viajes[u.id] for u in atendidas if u.id in viajes]
        sistema_par = [sistema[u.id] for u in atendidas + perdidas if u.id in sistema]

        filas.append({
            "par": f"{par[0]}->{par[1]}",
            "generadas": generadas,
            "atendidas": len(atendidas),
            "perdidas": len(perdidas),
            "pct_cumplimiento": 100.0 * len(atendidas) / generadas if generadas else float("nan"),
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
    perdidas = len(env.perdidas_historico)
    esperas = [u.tiempo_espera_min for u in env.atendidas_historico + env.perdidas_historico if u.tiempo_espera_min is not None]
    sistema = list(_tiempos_sistema(env).values())
    return {
        "unidades_generadas": generadas,
        "unidades_atendidas": atendidas,
        "unidades_perdidas": perdidas,
        "pct_cumplimiento_global": 100.0 * atendidas / generadas if generadas else float("nan"),
        "espera_media_min": float(np.mean(esperas)) if esperas else float("nan"),
        "sistema_medio_min": float(np.mean(sistema)) if sistema else float("nan"),
        "sistema_maximo_min": float(np.max(sistema)) if sistema else float("nan"),
    }


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
    sistema, sobre todas las unidades resueltas (atendidas + perdidas).
    """
    esperas = [u.tiempo_espera_min for u in env.atendidas_historico + env.perdidas_historico if u.tiempo_espera_min is not None]
    viajes = list(_tiempos_viaje_por_unidad(env).values())
    sistema = list(_tiempos_sistema(env).values())

    def percentiles(valores):
        if not valores:
            return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
        arr = np.array(valores)
        return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)), "max": float(np.max(arr))}

    return {"espera_min": percentiles(esperas), "viaje_min": percentiles(viajes), "sistema_min": percentiles(sistema)}


def perdidas_por_nodo(env) -> pd.DataFrame:
    """Dónde se pierde la gente: conteo por par completo (origen->destino)."""
    filas = [{"par": f"{e['par'][0]}->{e['par'][1]}"} for e in env.log_eventos if e["tipo"] == "perdida"]
    if not filas:
        return pd.DataFrame(columns=["par", "perdidas"])
    df = pd.DataFrame(filas)
    return df.groupby("par").size().reset_index(name="perdidas").sort_values("perdidas", ascending=False)


def desglose_recompensa(env) -> pd.DataFrame:
    """Serie de tiempo del desglose de recompensa (incomodidad, pérdida,
    movimiento, total) por paso -- ya se calculaba en cada `step()`, esta
    función solo la devuelve como tabla para graficar/sumar.
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
        "perdidas_por_nodo": perdidas_por_nodo(env),
        "desglose_recompensa": desglose_recompensa(env),
    }
