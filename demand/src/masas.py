"""Recorte geográfico y masas por nodo (Tareas 0 y 1).

Carga las grillas SSB de 250 m (población y bedrifter/empleo, todo Noruega),
las recorta a un cuadro amplio alrededor de Bergen + Askøy, y para cada uno
de los 4 nodos de demanda suma la población y el empleo de las celdas cuyo
centro cae dentro de un radio de captación (la gente camina hasta el muelle).

Todas las operaciones de distancia se hacen en EPSG:32633 (UTM 33N, el mismo
CRS de las grillas SSB), nunca en grados.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import Point

CRS_METRICO = "EPSG:32633"
CRS_GEOGRAFICO = "EPSG:4326"


def cargar_config(path: str | Path) -> dict:
    """Carga demand/config/instance.yaml completo como diccionario."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_nodos(cfg: dict, base_dir: Path) -> gpd.GeoDataFrame:
    """Lee los 4 nodos de demanda desde el config de bergen-boats (fuente única
    de verdad de las coordenadas) y los devuelve como GeoDataFrame en
    EPSG:32633, listo para operaciones métricas.
    """
    nodos_path = (base_dir / cfg["fuentes_externas"]["nodos_config"]).resolve()
    with open(nodos_path, "r", encoding="utf-8") as f:
        cfg_bb = yaml.safe_load(f)

    filas = cfg_bb["nodos_demanda"]
    df = pd.DataFrame(filas).set_index("id")
    df = df.loc[[n["id"] for n in filas]]  # preserva el orden del YAML

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lat, lon in zip(df["lat"], df["lon"])],
        crs=CRS_GEOGRAFICO,
    ).to_crs(CRS_METRICO)
    return gdf


def bbox_bergen_askoy(nodos_utm: gpd.GeoDataFrame, margen_m: float) -> tuple[float, float, float, float]:
    """Bounding box (xmin, ymin, xmax, ymax) en EPSG:32633 alrededor de los 4
    nodos, con margen. Se documenta el bbox exacto usado en el README.
    """
    xmin, ymin, xmax, ymax = nodos_utm.total_bounds
    return (xmin - margen_m, ymin - margen_m, xmax + margen_m, ymax + margen_m)


def recortar_grilla(
    geojson_path: Path,
    bbox: tuple[float, float, float, float],
    cache_path: Path,
    forzar: bool = False,
) -> gpd.GeoDataFrame:
    """Recorta una grilla SSB (todo Noruega) al bbox dado y cachea el resultado
    como GeoParquet en `cache_path` para no repetir el recorte cada vez.

    Devuelve el GeoDataFrame recortado (desde caché si ya existe).
    """
    if cache_path.exists() and not forzar:
        return gpd.read_parquet(cache_path)

    xmin, ymin, xmax, ymax = bbox
    gdf = gpd.read_file(geojson_path, bbox=(xmin, ymin, xmax, ymax))
    # bbox de pyogrio filtra por intersección con el bbox; nos quedamos solo
    # con las celdas cuyo CENTRO cae dentro (criterio consistente con Tarea 1).
    centroides = gdf.geometry.centroid
    dentro = (
        (centroides.x >= xmin) & (centroides.x <= xmax) &
        (centroides.y >= ymin) & (centroides.y <= ymax)
    )
    gdf = gdf.loc[dentro].reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache_path)
    return gdf


def celdas_en_radio(grilla: gpd.GeoDataFrame, centro: Point, radio_m: float) -> gpd.GeoDataFrame:
    """Subconjunto de `grilla` cuyas celdas tienen su CENTROIDE dentro del
    círculo de radio `radio_m` alrededor de `centro` (ambos en CRS métrico).
    """
    buffer = centro.buffer(radio_m)
    centroides = grilla.geometry.centroid
    dentro = centroides.within(buffer)
    return grilla.loc[dentro].reset_index(drop=True)


def celdas_en_zona(grilla: gpd.GeoDataFrame, zona_geom) -> gpd.GeoDataFrame:
    """Igual que `celdas_en_radio` pero con una zona arbitraria (p. ej. la
    unión de las grunnkretser elegidas a mano para un nodo) en vez de un
    círculo — mismo criterio de "centro adentro", para que el método sea
    comparable con el de la Tarea 1 original.
    """
    centroides = grilla.geometry.centroid
    dentro = centroides.within(zona_geom)
    return grilla.loc[dentro].reset_index(drop=True)


def calcular_masas_por_zona(
    cfg: dict,
    nodos_utm: gpd.GeoDataFrame,
    zonas_por_nodo: dict,
    pop_recortada: gpd.GeoDataFrame,
    bedrifter_recortada: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, gpd.GeoDataFrame], dict[str, gpd.GeoDataFrame]]:
    """Como `calcular_masas_por_nodo`, pero la zona de cada nodo es un polígono
    arbitrario (`zonas_por_nodo[id_nodo]`, p. ej. la unión de grunnkretser
    elegidas a mano) en vez de un círculo de radio fijo.
    """
    col_pop = cfg["datos_ssb"]["poblacion_columna"]
    col_emp = cfg["datos_ssb"]["empleo_columna"]
    col_est = cfg["datos_ssb"]["establecimientos_columna"]

    filas = []
    celdas_pop_por_nodo = {}
    celdas_emp_por_nodo = {}

    for nid, row in nodos_utm.iterrows():
        zona = zonas_por_nodo[nid]
        cp = celdas_en_zona(pop_recortada, zona)
        ce = celdas_en_zona(bedrifter_recortada, zona)
        celdas_pop_por_nodo[nid] = cp
        celdas_emp_por_nodo[nid] = ce

        pop_vals = cp[col_pop]
        emp_vals = ce[col_emp]
        est_vals = ce[col_est]

        filas.append({
            "id": nid,
            "nombre": row["nombre"],
            "area_zona_km2": round(zona.area / 1e6, 2),
            "n_celdas_poblacion": len(cp),
            "poblacion_total": int(pop_vals.sum()) if len(cp) else 0,
            "poblacion_min_celda": int(pop_vals.min()) if len(cp) else 0,
            "poblacion_max_celda": int(pop_vals.max()) if len(cp) else 0,
            "poblacion_media_celda": round(float(pop_vals.mean()), 2) if len(cp) else 0.0,
            "poblacion_mediana_celda": float(pop_vals.median()) if len(cp) else 0.0,
            "n_celdas_bedrifter": len(ce),
            "establecimientos_total": int(est_vals.sum()) if len(ce) else 0,
            "empleo_total": int(emp_vals.sum()) if len(ce) else 0,
            "empleo_min_celda": int(emp_vals.min()) if len(ce) else 0,
            "empleo_max_celda": int(emp_vals.max()) if len(ce) else 0,
            "empleo_media_celda": round(float(emp_vals.mean()), 2) if len(ce) else 0.0,
            "empleo_mediana_celda": float(emp_vals.median()) if len(ce) else 0.0,
        })

    resumen = pd.DataFrame(filas).set_index("id")
    return resumen, celdas_pop_por_nodo, celdas_emp_por_nodo


def calcular_masas_por_nodo(
    cfg: dict,
    nodos_utm: gpd.GeoDataFrame,
    pop_recortada: gpd.GeoDataFrame,
    bedrifter_recortada: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, gpd.GeoDataFrame], dict[str, gpd.GeoDataFrame]]:
    """Para cada nodo, suma población y empleo de las celdas dentro del radio
    de captación, y calcula estadísticas descriptivas por celda.

    Devuelve:
      resumen: DataFrame con una fila por nodo (totales + estadísticas).
      celdas_pop_por_nodo: dict id_nodo -> GeoDataFrame de celdas de población capturadas.
      celdas_emp_por_nodo: dict id_nodo -> GeoDataFrame de celdas de empleo capturadas.
    """
    radio_m = cfg["captacion"]["radio_m"]
    col_pop = cfg["datos_ssb"]["poblacion_columna"]
    col_emp = cfg["datos_ssb"]["empleo_columna"]
    col_est = cfg["datos_ssb"]["establecimientos_columna"]

    filas = []
    celdas_pop_por_nodo = {}
    celdas_emp_por_nodo = {}

    for nid, row in nodos_utm.iterrows():
        centro = row.geometry
        cp = celdas_en_radio(pop_recortada, centro, radio_m)
        ce = celdas_en_radio(bedrifter_recortada, centro, radio_m)
        celdas_pop_por_nodo[nid] = cp
        celdas_emp_por_nodo[nid] = ce

        pop_vals = cp[col_pop]
        emp_vals = ce[col_emp]
        est_vals = ce[col_est]

        filas.append({
            "id": nid,
            "nombre": row["nombre"],
            "radio_m": radio_m,
            "n_celdas_poblacion": len(cp),
            "poblacion_total": int(pop_vals.sum()) if len(cp) else 0,
            "poblacion_min_celda": int(pop_vals.min()) if len(cp) else 0,
            "poblacion_max_celda": int(pop_vals.max()) if len(cp) else 0,
            "poblacion_media_celda": round(float(pop_vals.mean()), 2) if len(cp) else 0.0,
            "poblacion_mediana_celda": float(pop_vals.median()) if len(cp) else 0.0,
            "n_celdas_bedrifter": len(ce),
            "establecimientos_total": int(est_vals.sum()) if len(ce) else 0,
            "empleo_total": int(emp_vals.sum()) if len(ce) else 0,
            "empleo_min_celda": int(emp_vals.min()) if len(ce) else 0,
            "empleo_max_celda": int(emp_vals.max()) if len(ce) else 0,
            "empleo_media_celda": round(float(emp_vals.mean()), 2) if len(ce) else 0.0,
            "empleo_mediana_celda": float(emp_vals.median()) if len(ce) else 0.0,
        })

    resumen = pd.DataFrame(filas).set_index("id")
    return resumen, celdas_pop_por_nodo, celdas_emp_por_nodo
