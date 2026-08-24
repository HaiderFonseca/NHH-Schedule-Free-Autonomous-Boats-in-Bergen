"""Grunnkretser (unidades estadísticas pequeñas de Noruega) como alternativa
al círculo de captación fijo.

Un círculo de radio fijo no respeta el uso real del suelo: para Laksevåg
captura bastante agua/vegetación, y para Bryggen (denso, urbano) el mismo
radio concentra desproporcionadamente la masa. Las grunnkretser son la unidad
administrativa/estadística real que usa SSB — dejar que el equipo elija a
mano qué grunnkretser le corresponden a cada nodo es más defendible que un
círculo arbitrario igual para los 4.

Fuente: Kartverket, WFS "Statistiske enheter grunnkretser"
(https://kartkatalog.geonorge.no/metadata/statistiske-enheter-grunnkretser/cc7ded0b-7d34-4db6-8fdb-c5a7682b6836),
capa `Grunnkrets`, licencia CC BY 4.0, CRS nativo EPSG:4258 (ETRS89 — para
efectos prácticos intercambiable con WGS84 en esta escala).

Kommunenummer relevantes: Bergen = 4601, Askøy = 4627 (Kleppestø).
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd

WFS_URL = "WFS:https://wfs.geonorge.no/skwms1/wfs.grunnkretser"
CAPA = "Grunnkrets"
CRS_METRICO = "EPSG:32633"

KOMMUNENUMMER_BERGEN = 4601
KOMMUNENUMMER_ASKOY = 4627


def descargar_grunnkretser(cache_path: Path, forzar: bool = False) -> gpd.GeoDataFrame:
    """Descarga (o lee de caché) TODAS las grunnkretser de Noruega desde el
    WFS de Kartverket, reproyectadas a EPSG:32633.

    Nota técnica: en Windows, GDAL necesita `PYTHONUTF8=1` para no romperse
    con los nombres con letras noruegas (å/æ/ø) que trae el WFS — si esta
    función falla con UnicodeDecodeError, es por eso.
    """
    if cache_path.exists() and not forzar:
        return gpd.read_parquet(cache_path)

    os.environ.setdefault("PYTHONUTF8", "1")
    gdf = gpd.read_file(WFS_URL, layer=CAPA)
    gdf = gdf.to_crs(CRS_METRICO)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache_path)
    return gdf


def grunnkretser_bergen_askoy(gdf_norge: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filtra a las grunnkretser de Bergen (4601) y Askøy (4627)."""
    return gdf_norge[gdf_norge["kommunenummer"].isin([KOMMUNENUMMER_BERGEN, KOMMUNENUMMER_ASKOY])].reset_index(drop=True)


def cargar_areas_estudio(path: Path, nodos_utm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Carga las 4 áreas dibujadas a mano (QGIS, `data/Areas_estudio.gpkg`,
    sin atributos) y les asigna el nodo correspondiente por cercanía del
    centroide — con las 4 áreas y los 4 nodos, la distancia al nodo correcto
    es un orden de magnitud menor que a cualquier otro, así que no hay
    ambigüedad real (se verifica explícitamente en el notebook, no solo aquí).
    """
    import numpy as np

    areas = gpd.read_file(path).to_crs(CRS_METRICO).reset_index(drop=True)
    ids_nodos = nodos_utm.index.tolist()

    centroides = areas.geometry.centroid
    distancias = np.column_stack([
        centroides.distance(nodos_utm.loc[nid, "geometry"]).to_numpy()
        for nid in ids_nodos
    ])
    idx_min = distancias.argmin(axis=1)

    areas["nodo"] = [ids_nodos[i] for i in idx_min]
    areas["distancia_nodo_m"] = distancias[np.arange(len(areas)), idx_min]
    areas["area_km2"] = areas.geometry.area / 1e6
    return areas.set_index("nodo")


def seleccionar_grunnkretser_por_area(gdf_grunnkretser: gpd.GeoDataFrame, area_geom) -> gpd.GeoDataFrame:
    """Grunnkretser que TOCAN `area_geom` (criterio "intersecta", no "centro
    adentro"): si el área dibujada agarra aunque sea un pedacito de una
    grunnkrets, esa grunnkrets se toma COMPLETA.

    Decisión explícita del usuario, que dibujó las áreas pensando en esto: un
    círculo/polígono cortado a mitad de una grunnkrets no tiene sentido -- la
    grunnkrets es la unidad real (barrio), no se parte. Con "centro adentro"
    (el criterio anterior) se estaban quedando afuera zonas costeras de
    Sandviken que el área sí tocaba pero cuyo centroide caía justo del otro
    lado del borde dibujado.
    """
    toca = gdf_grunnkretser.geometry.intersects(area_geom)
    return gdf_grunnkretser.loc[toca].reset_index(drop=True)


def resolver_duplicados(seleccion_por_nodo: dict, areas: gpd.GeoDataFrame) -> dict:
    """Con el criterio "intersecta" (toma la grunnkrets completa si el área la
    toca), una grunnkrets que esté justo en el borde entre dos áreas puede
    quedar seleccionada en los dos nodos a la vez. Aquí se resuelve: se deja
    solo en el nodo cuya área dibujada tiene MÁS superposición con esa
    grunnkrets (no se reparte la población/empleo entre los dos).
    """
    import pandas as pd

    todas = pd.concat([
        df.assign(nodo=nid) for nid, df in seleccion_por_nodo.items()
    ], ignore_index=True)

    conteo = todas["grunnkretsnummer"].value_counts()
    duplicadas = conteo[conteo > 1].index.tolist()

    ganador_por_grunnkrets = {}
    for gknum in duplicadas:
        filas = todas[todas["grunnkretsnummer"] == gknum]
        geom = filas.geometry.iloc[0]
        overlaps = {
            nid: geom.intersection(areas.loc[nid, "geometry"]).area
            for nid in filas["nodo"]
        }
        ganador_por_grunnkrets[gknum] = max(overlaps, key=overlaps.get)

    resultado = {}
    for nid, df in seleccion_por_nodo.items():
        mask_no_disputada = ~df["grunnkretsnummer"].isin(duplicadas)
        mask_ganada = df["grunnkretsnummer"].map(
            lambda g: ganador_por_grunnkrets.get(g) == nid
        )
        resultado[nid] = df.loc[mask_no_disputada | mask_ganada].reset_index(drop=True)

    return resultado


def nodo_mas_cercano(gdf_grunnkretser: gpd.GeoDataFrame, nodos_utm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Agrega columnas `nodo_mas_cercano` / `distancia_nodo_mas_cercano_m`
    (solo SUGERENCIA, no asignación definitiva) con el nodo cuyo centro está
    más cerca del centroide de cada grunnkrets — para orientar la elección
    manual, no para reemplazarla.
    """
    import numpy as np

    gdf = gdf_grunnkretser.reset_index(drop=True).copy()
    centroides = gdf.geometry.centroid

    ids_nodos = nodos_utm.index.tolist()
    distancias = np.column_stack([
        centroides.distance(nodos_utm.loc[nid, "geometry"]).to_numpy()
        for nid in ids_nodos
    ])
    idx_min = distancias.argmin(axis=1)

    gdf["nodo_mas_cercano"] = [ids_nodos[i] for i in idx_min]
    gdf["distancia_nodo_mas_cercano_m"] = distancias[np.arange(len(gdf)), idx_min]
    return gdf
