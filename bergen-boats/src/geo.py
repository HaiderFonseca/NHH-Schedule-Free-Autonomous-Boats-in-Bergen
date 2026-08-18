"""Geometría de la instancia: distancias Haversine, calibración de velocidad
y matrices de tiempo de viaje entre nodos.

Se usa tanto desde los notebooks de cada paso como desde la simulación más
adelante, para que la lógica de distancias/tiempos viva en un solo lugar.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import yaml

RADIO_TIERRA_KM = 6371.0088  # radio medio terrestre (IUGG)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en línea recta entre dos coordenadas (grados) usando Haversine.

    Aproxima la distancia sobre el agua en fiordos relativamente rectos;
    para tramos que puedan bordear tierra, esto es un mínimo optimista.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * asin(sqrt(a))


def cargar_instancia(config_path: str | Path) -> dict:
    """Carga config/instance.yaml completo como diccionario."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def nodos_a_dataframe(nodos: list[dict]) -> pd.DataFrame:
    """Convierte una lista de nodos (dicts con id/nombre/lat/lon/rol) en DataFrame
    indexado por id, en el orden en que aparecen en el YAML.
    """
    df = pd.DataFrame(nodos).set_index("id")
    return df.loc[[n["id"] for n in nodos]]  # preserva el orden del YAML


def matriz_distancias(nodos_df: pd.DataFrame) -> pd.DataFrame:
    """Matriz cuadrada de distancias Haversine (km) entre todos los nodos de
    `nodos_df` (que debe tener columnas lat/lon e índice = id del nodo).
    """
    ids = nodos_df.index.tolist()
    mat = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for i in ids:
        for j in ids:
            if i == j:
                mat.loc[i, j] = 0.0
            else:
                mat.loc[i, j] = haversine_km(
                    nodos_df.loc[i, "lat"], nodos_df.loc[i, "lon"],
                    nodos_df.loc[j, "lat"], nodos_df.loc[j, "lon"],
                )
    return mat


def calibrar_velocidad_kmh(
    nodos_df: pd.DataFrame,
    desde: str,
    hasta: str,
    tiempo_referencia_min: float,
    velocidad_forzada_kmh: float | None = None,
) -> float:
    """Calibra la velocidad efectiva (km/h) a partir de un tramo real conocido.

    Si `velocidad_forzada_kmh` viene dado (no None), se usa ese valor
    directamente (útil para análisis de sensibilidad) y se ignora el resto.
    """
    if velocidad_forzada_kmh is not None:
        return float(velocidad_forzada_kmh)

    dist_km = haversine_km(
        nodos_df.loc[desde, "lat"], nodos_df.loc[desde, "lon"],
        nodos_df.loc[hasta, "lat"], nodos_df.loc[hasta, "lon"],
    )
    horas = tiempo_referencia_min / 60.0
    return dist_km / horas


def matriz_tiempos(matriz_dist_km: pd.DataFrame, velocidad_kmh: float) -> pd.DataFrame:
    """Convierte una matriz de distancias (km) a tiempos (min) a velocidad constante."""
    return matriz_dist_km / velocidad_kmh * 60.0


def construir_matrices(config_path: str | Path, incluir_waypoints: bool = False):
    """Atajo de alto nivel: carga la instancia y devuelve
    (nodos_df, matriz_distancias_km, matriz_tiempos_min, velocidad_kmh).

    Por defecto solo usa los nodos de demanda (`incluir_waypoints=False`),
    que es lo correcto para la matriz que ve la política de despacho.
    Con `incluir_waypoints=True` se agregan los waypoints (p.ej. Hegreneset)
    únicamente como referencia geométrica para el ruteo, nunca como nodo de demanda.
    """
    cfg = cargar_instancia(config_path)
    nodos = list(cfg["nodos_demanda"])
    if incluir_waypoints:
        nodos = nodos + list(cfg.get("waypoints", []))

    nodos_df = nodos_a_dataframe(nodos)
    dist_km = matriz_distancias(nodos_df)

    cal = cfg["calibracion"]
    velocidad_kmh = calibrar_velocidad_kmh(
        nodos_df,
        desde=cal["referencia_desde"],
        hasta=cal["referencia_hasta"],
        tiempo_referencia_min=cal["tiempo_referencia_min"],
        velocidad_forzada_kmh=cal.get("velocidad_forzada_kmh"),
    )
    tiempos_min = matriz_tiempos(dist_km, velocidad_kmh)
    return nodos_df, dist_km, tiempos_min, velocidad_kmh
