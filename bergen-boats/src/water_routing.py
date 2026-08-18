"""Ruteo sobre agua: construye una malla navegable a partir de un mapa real
(clasificando píxeles agua/tierra) y calcula distancias de barco entre nodos
con el camino más corto que **no cruza tierra** (Dijkstra sobre la malla).

Por qué existe este módulo: `geo.py` calcula distancias en línea recta
(Haversine), que no sabe si esa línea pasa por encima de una península. Para
Bryggen (metido en la bahía de Vågen, junto a la península de Nordnes) esa
aproximación cruza tierra. Este módulo corrige eso ruteando sobre una malla
de píxeles de agua en vez de una línea recta.

Cómo funciona, en corto:
1. Se descarga un mapa base (CartoDB Positron sin etiquetas) del área.
2. Se clasifica cada píxel como agua o tierra por color.
3. Cada píxel de agua es un nodo de un grafo; se conecta con sus 8 vecinos
   (si también son agua) con un peso igual a la distancia real en km.
4. Cada nodo/puerto de la instancia se "engancha" (snap) al píxel de agua
   más cercano, restringido a la componente conexa principal (para no caer
   en un charco aislado).
5. Se corre Dijkstra desde cada puerto para obtener distancias y caminos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

RADIO_TIERRA_KM = 6371.0088  # radio medio terrestre (IUGG) — usar para distancias reales

# Web Mercator (EPSG:3857, el que usan los tiles XYZ) asume una ESFERA de
# radio 6378137.0 m (el semieje mayor de WGS84), no el radio medio. Usar el
# radio equivocado aquí desalinea por completo píxel <-> lon/lat.
RADIO_MERCATOR_M = 6378137.0

# Color de referencia del agua en los tiles CartoDB Positron (sin etiquetas),
# obtenido empíricamente muestreando los píxeles más frecuentes de un tile
# de Bergen. Si se cambia de proveedor de mapa, este valor hay que recalibrarlo.
COLOR_AGUA_POSITRON = (212, 218, 220)


# ---------------------------------------------------------------------------
# 1-2. Descargar el mapa base y clasificar agua/tierra
# ---------------------------------------------------------------------------

def descargar_mascara_agua(
    west: float, south: float, east: float, north: float,
    zoom: int = 14,
    color_agua_ref=COLOR_AGUA_POSITRON,
    umbral_color: float = 15.0,
    proveedor=None,
):
    """Descarga el basemap del bbox (lon/lat) y devuelve (water_mask, ext).

    `water_mask` es un array booleano (H, W): True = agua navegable.
    `ext` es (xmin, xmax, ymin, ymax) en EPSG:3857 (metros), tal como lo
    devuelve contextily — necesario para pasar de píxel a lon/lat y viceversa.
    """
    import contextily as cx

    if proveedor is None:
        proveedor = cx.providers.CartoDB.PositronNoLabels

    img, ext = cx.bounds2img(west, south, east, north, ll=True, zoom=zoom, source=proveedor)
    arr = np.asarray(img)[:, :, :3].astype(float)
    ref = np.array(color_agua_ref, dtype=float)
    water_mask = np.linalg.norm(arr - ref, axis=2) < umbral_color
    return water_mask, ext


# ---------------------------------------------------------------------------
# Conversión píxel <-> lon/lat (Web Mercator esférico, igual que los tiles XYZ)
# ---------------------------------------------------------------------------

def pixel_a_lonlat(row, col, ext, shape):
    """Centro del píxel (row, col) -> (lat, lon). Acepta escalares o arrays numpy."""
    xmin, xmax, ymin, ymax = ext
    H, W = shape
    x_merc = xmin + (np.asarray(col) + 0.5) / W * (xmax - xmin)
    y_merc = ymax - (np.asarray(row) + 0.5) / H * (ymax - ymin)
    lon = x_merc / RADIO_MERCATOR_M * 180.0 / np.pi
    lat = (2 * np.arctan(np.exp(y_merc / RADIO_MERCATOR_M)) - np.pi / 2) * 180.0 / np.pi
    return lat, lon


def lonlat_a_pixel(lat: float, lon: float, ext, shape) -> tuple[float, float]:
    """(lat, lon) -> (row, col) en coordenadas de píxel (float, sin redondear)."""
    xmin, xmax, ymin, ymax = ext
    H, W = shape
    x_merc = lon * np.pi / 180.0 * RADIO_MERCATOR_M
    y_merc = RADIO_MERCATOR_M * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    col = (x_merc - xmin) / (xmax - xmin) * W - 0.5
    row = (ymax - y_merc) / (ymax - ymin) * H - 0.5
    return row, col


# ---------------------------------------------------------------------------
# 3. Construir el grafo de navegación sobre los píxeles de agua
# ---------------------------------------------------------------------------

def construir_grafo_navegable(water_mask: np.ndarray, ext):
    """Construye el grafo (8-conectado) de píxeles de agua.

    Devuelve:
      grafo: scipy.sparse.csr_matrix (n_nodos x n_nodos), peso = distancia real en km.
      node_ids: array (H, W) int64, -1 si es tierra, si no el índice del nodo.
      pixel_de_nodo: array (n_nodos, 2) con (row, col) de cada nodo, para reconstruir rutas.
    """
    H, W = water_mask.shape
    xmin, xmax, ymin, ymax = ext

    row_idx = np.arange(H)
    y_merc = ymax - (row_idx + 0.5) / H * (ymax - ymin)
    lat_por_fila = (2 * np.arctan(np.exp(y_merc / RADIO_MERCATOR_M)) - np.pi / 2) * 180.0 / np.pi
    escala_por_fila = np.cos(np.radians(lat_por_fila))  # factor merc -> distancia real (conforme)

    dx_merc = (xmax - xmin) / W
    dy_merc = (ymax - ymin) / H

    node_ids = -np.ones((H, W), dtype=np.int64)
    water_flat_idx = np.flatnonzero(water_mask.ravel())
    node_ids.ravel()[water_flat_idx] = np.arange(water_flat_idx.size)
    n_nodos = water_flat_idx.size

    filas_agua, cols_agua = np.nonzero(water_mask)
    pixel_de_nodo = np.column_stack([filas_agua, cols_agua])

    # offsets "hacia adelante" (evita duplicar cada arista; se agregan ambos sentidos abajo)
    offsets = [
        (0, 1, dx_merc),
        (1, 0, dy_merc),
        (1, 1, np.hypot(dx_merc, dy_merc)),
        (1, -1, np.hypot(dx_merc, dy_merc)),
    ]

    filas_e, cols_e, pesos = [], [], []
    for dr, dc, base_dist_merc in offsets:
        r0, r1 = max(0, -dr), H - max(0, dr)
        c0, c1 = max(0, -dc), W - max(0, dc)
        a_mask = water_mask[r0:r1, c0:c1]
        b_mask = water_mask[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        both = a_mask & b_mask
        if not both.any():
            continue
        a_ids = node_ids[r0:r1, c0:c1][both]
        b_ids = node_ids[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both]
        rr = np.repeat(np.arange(r0, r1)[:, None], c1 - c0, axis=1)[both]
        escala = (escala_por_fila[rr] + escala_por_fila[rr + dr]) / 2.0
        dist_km = base_dist_merc * escala / 1000.0
        filas_e.append(a_ids); cols_e.append(b_ids); pesos.append(dist_km)
        filas_e.append(b_ids); cols_e.append(a_ids); pesos.append(dist_km)

    filas_e = np.concatenate(filas_e)
    cols_e = np.concatenate(cols_e)
    pesos = np.concatenate(pesos)
    grafo = coo_matrix((pesos, (filas_e, cols_e)), shape=(n_nodos, n_nodos)).tocsr()
    return grafo, node_ids, pixel_de_nodo


def componente_principal(grafo) -> tuple[np.ndarray, int]:
    """Etiqueta de componente conexa de cada nodo, y la etiqueta de la componente
    más grande (la malla de mar abierta). Sirve para no enganchar un puerto a un
    charco aislado que el color de agua clasificó bien pero que no está conectado
    al resto del mar.
    """
    n_comp, labels = connected_components(grafo, directed=False)
    tam = np.bincount(labels)
    return labels, int(np.argmax(tam))


# ---------------------------------------------------------------------------
# 4. Enganchar (snap) cada puerto a su nodo de agua más cercano
# ---------------------------------------------------------------------------

def snap_a_grafo(
    lat: float, lon: float,
    water_mask: np.ndarray, node_ids: np.ndarray, ext,
    labels: np.ndarray | None = None, label_valido: int | None = None,
    max_radio_px: int = 60,
) -> int:
    """Encuentra el node_id de agua más cercano a (lat, lon).

    Si se dan `labels`/`label_valido`, solo acepta píxeles de esa componente
    conexa (evita charcos/estanques aislados desconectados del mar abierto).
    """
    H, W = water_mask.shape
    row, col = lonlat_a_pixel(lat, lon, ext, water_mask.shape)
    row, col = int(round(row)), int(round(col))

    def es_valido(r, c):
        if not water_mask[r, c]:
            return False
        if labels is None:
            return True
        return labels[node_ids[r, c]] == label_valido

    if 0 <= row < H and 0 <= col < W and es_valido(row, col):
        return int(node_ids[row, col])

    for radio in range(1, max_radio_px + 1):
        r0, r1 = max(0, row - radio), min(H, row + radio + 1)
        c0, c1 = max(0, col - radio), min(W, col + radio + 1)
        sub_water = water_mask[r0:r1, c0:c1]
        if labels is not None:
            sub_ids = node_ids[r0:r1, c0:c1]
            sub_valid = sub_water & (labels[np.clip(sub_ids, 0, None)] == label_valido) & (sub_ids >= 0)
        else:
            sub_valid = sub_water
        if sub_valid.any():
            rr, cc = np.nonzero(sub_valid)
            d2 = (rr + r0 - row) ** 2 + (cc + c0 - col) ** 2
            k = np.argmin(d2)
            return int(node_ids[rr[k] + r0, cc[k] + c0])

    raise RuntimeError(f"No se encontró agua navegable cerca de ({lat}, {lon})")


# ---------------------------------------------------------------------------
# 5. Distancias y rutas
# ---------------------------------------------------------------------------

def matriz_distancias_navegables(nodos_df: pd.DataFrame, water_mask, ext, grafo, node_ids):
    """Calcula la matriz de distancias navegables (km) entre todos los nodos de
    `nodos_df` (índice = id, columnas lat/lon), y devuelve también lo necesario
    para reconstruir rutas: (matriz_km, snap_idx dict, predecessors, dist_matrix, ids_orden).
    """
    labels, label_principal = componente_principal(grafo)

    snap_idx = {}
    for nid, row in nodos_df.iterrows():
        snap_idx[nid] = snap_a_grafo(
            row["lat"], row["lon"], water_mask, node_ids, ext,
            labels=labels, label_valido=label_principal,
        )

    ids_orden = nodos_df.index.tolist()
    src_indices = [snap_idx[i] for i in ids_orden]
    dist_matrix, predecessors = dijkstra(grafo, directed=True, indices=src_indices, return_predecessors=True)

    mat = pd.DataFrame(index=ids_orden, columns=ids_orden, dtype=float)
    for i, a in enumerate(ids_orden):
        for b in ids_orden:
            mat.loc[a, b] = dist_matrix[i, snap_idx[b]]

    return mat, snap_idx, predecessors, ids_orden


def reconstruir_ruta_latlon(origen_id: str, destino_id: str, ids_orden, snap_idx, predecessors, pixel_de_nodo, ext, shape):
    """Reconstruye la ruta (lista de (lat, lon)) del camino más corto de
    `origen_id` a `destino_id`, caminando la matriz de predecesores de Dijkstra.
    """
    i_origen = ids_orden.index(origen_id)
    destino_node = snap_idx[destino_id]
    pred_fila = predecessors[i_origen]

    camino_nodos = [destino_node]
    actual = destino_node
    while pred_fila[actual] != -9999 and pred_fila[actual] != actual:
        actual = pred_fila[actual]
        camino_nodos.append(actual)
    camino_nodos.reverse()

    filas = pixel_de_nodo[camino_nodos, 0]
    cols = pixel_de_nodo[camino_nodos, 1]
    lats, lons = pixel_a_lonlat(filas, cols, ext, shape)
    return list(zip(lats, lons))
