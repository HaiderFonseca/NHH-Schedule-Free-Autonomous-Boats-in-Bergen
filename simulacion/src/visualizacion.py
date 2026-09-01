"""Visualización: mapa real de Bergen + rutas navegables reales + barcos y
colas en movimiento (docs/especificacion_simulador_rl.md, sección
"Visualización").

Reusa el estilo de `bergen-boats/02_ruteo_navegable` (contextily, EPSG:3857,
nodos como puntos con etiqueta) y la geometría
real de `bergen-boats/02_ruteo_navegable/output/rutas_navegables.geojson`
(persistida en este mismo trabajo) -- los barcos se interpolan SOBRE esa
geometría, nunca en línea recta cruzando tierra.

No dibuja desde el estado "rico" (`EstadoSimulacion`, que se muta en cada
paso) sino desde una lista de snapshots planos (`info["estado_dict"]` de
cada `step()`, ya son diccionarios nuevos e inmutables) -- así se puede
guardar el historial completo de una corrida sin que un paso posterior
sobreescriba a los anteriores.
"""
from __future__ import annotations

import contextily as cx
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Point

import metricas as met
from estado import CODIGO_CORTO, pares_od

CODIGO_A_ID = {v: k for k, v in CODIGO_CORTO.items()}

COLOR_NODO = "#1f6f8b"
COLOR_RUTA = "#c9d3d6"
COLOR_BARCO_VACIO = "#7f8c8d"
COLOR_BARCO_LLENO = "#c0392b"
COLOR_COLA = "#e67e22"


def cargar_capas_mapa(cfg_bb: dict, rutas_geojson_path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Nodos y rutas navegables como GeoDataFrames en EPSG:3857 (Web
    Mercator, lo que espera contextily) -- mismo patrón que
    `bergen-boats/02_ruteo_navegable/notebook.ipynb`.
    """
    filas = cfg_bb["nodos_demanda"]
    gdf_nodos = gpd.GeoDataFrame(
        filas,
        geometry=[Point(f["lon"], f["lat"]) for f in filas],
        crs="EPSG:4326",
    ).set_index("id").to_crs(epsg=3857)

    gdf_rutas = gpd.read_file(rutas_geojson_path).to_crs(epsg=3857)
    return gdf_nodos, gdf_rutas


def _fila_ruta(gdf_rutas: gpd.GeoDataFrame, origen: str, destino: str):
    m = ((gdf_rutas["desde"] == origen) & (gdf_rutas["hasta"] == destino)) | \
        ((gdf_rutas["desde"] == destino) & (gdf_rutas["hasta"] == origen))
    fila = gdf_rutas[m]
    if fila.empty:
        return None, False
    fila = fila.iloc[0]
    misma_direccion = fila["desde"] == origen
    return fila.geometry, misma_direccion


def punto_en_ruta(gdf_nodos: gpd.GeoDataFrame, gdf_rutas: gpd.GeoDataFrame, origen: str, destino: str, fraccion: float) -> Point:
    """Posición interpolada de un barco viajando de `origen` a `destino`,
    con `fraccion` en [0, 1] del trayecto ya recorrido -- sobre la
    geometría real de la ruta (bordea tierra), no una línea recta.
    """
    if origen == destino:
        return gdf_nodos.loc[origen, "geometry"]
    geom, misma_direccion = _fila_ruta(gdf_rutas, origen, destino)
    if geom is None:
        # Sin geometria para este par (no deberia pasar con 4 nodos, C(4,2)=6
        # cubre todos) -- se cae a linea recta entre los nodos como respaldo.
        p0, p1 = gdf_nodos.loc[origen, "geometry"], gdf_nodos.loc[destino, "geometry"]
        return Point(p0.x + (p1.x - p0.x) * fraccion, p0.y + (p1.y - p0.y) * fraccion)
    f = fraccion if misma_direccion else (1.0 - fraccion)
    return geom.interpolate(f, normalized=True)


def descargar_fondo_mapa(gdf_nodos: gpd.GeoDataFrame, margen: float = 0.35):
    """Descarga el mosaico de tiles del mapa UNA sola vez (no una por
    frame): pedirle el tile a contextily en cada frame de la animación
    multiplicaría las llamadas de red por el número de pasos -- más lento e
    innecesariamente frágil ante bloqueos intermitentes del servidor de
    tiles. Devuelve `(imagen, extent)`, listos para `ax.imshow` en cada
    frame sin volver a tocar la red.
    """
    xmin, ymin, xmax, ymax = gdf_nodos.total_bounds
    mx, my = (xmax - xmin) * margen, (ymax - ymin) * margen
    w, s, e, n = xmin - mx, ymin - my, xmax + mx, ymax + my
    # Esri.WorldGrayCanvas, no CartoDB.Positron (el que usa bergen-boats/02):
    # verificado que CartoDB ahora exige API key para ese estilo (cambio de
    # su politica -- devuelve un tile con marca de agua "API KEY REQUIRED"
    # en vez de un error limpio). OpenStreetMap.Mapnik tambien se probo y
    # bloqueo la solicitud ("App is not following the tile usage policy").
    # Esri.WorldGrayCanvas es keyless y respondio limpio.
    imagen, extent = cx.bounds2img(w, s, e, n, zoom=12, source=cx.providers.Esri.WorldGrayCanvas)
    return imagen, extent, (w, s, e, n)


def dibujar_frame(ax, frame: dict, gdf_nodos: gpd.GeoDataFrame, gdf_rutas: gpd.GeoDataFrame, matriz_tiempos, capacidad_barco: int, fondo=None):
    """Dibuja un instante (`frame` = un `estado_dict`, ver `estado.py`
    `.to_dict()`) sobre un mapa real. `fondo` = `(imagen, extent, limites)`
    de `descargar_fondo_mapa` -- si se pasa, se reusa (`ax.imshow`, sin red);
    si no, se descarga aquí mismo (uso puntual, un solo frame suelto).
    """
    ax.clear()

    gdf_rutas.plot(ax=ax, color=COLOR_RUTA, linewidth=2.0, alpha=0.8, zorder=2)
    gdf_nodos.plot(ax=ax, color=COLOR_NODO, edgecolor="white", markersize=180, zorder=3)

    demanda_por_nodo = {n: 0 for n in gdf_nodos.index}
    for clave, v in frame["demanda"].items():
        origen_cod = clave.split("->")[0]
        demanda_por_nodo[CODIGO_A_ID[origen_cod]] += v["personas"]

    for nid, row in gdf_nodos.iterrows():
        ax.annotate(
            row["nombre"], xy=(row.geometry.x, row.geometry.y), xytext=(9, -14),
            textcoords="offset points", fontsize=10, fontweight="bold", color=COLOR_NODO,
            zorder=4, path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        espera = demanda_por_nodo[nid]
        if espera > 0:
            ax.annotate(
                f"esperan: {espera}", xy=(row.geometry.x, row.geometry.y), xytext=(9, 8),
                textcoords="offset points", fontsize=10, fontweight="bold", color=COLOR_COLA,
                zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

    for b in frame["barcos"]:
        origen_id, destino_id = CODIGO_A_ID[b["origen"]], CODIGO_A_ID[b["destino"]]
        if b["libre"]:
            fraccion = 0.0
        else:
            total = float(matriz_tiempos.loc[origen_id, destino_id])
            fraccion = max(0.0, min(1.0, 1.0 - b["min_para_llegar"] / total)) if total > 0 else 1.0
        p = punto_en_ruta(gdf_nodos, gdf_rutas, origen_id, destino_id, fraccion)
        color = COLOR_BARCO_LLENO if b["ocupacion"] > 0 else COLOR_BARCO_VACIO
        tam = 90 + 8 * b["ocupacion"]
        ax.scatter([p.x], [p.y], s=tam, color=color, edgecolor="white", linewidth=1.2, zorder=6, marker="^")
        if b["ocupacion"] > 0:
            ax.annotate(str(b["ocupacion"]), xy=(p.x, p.y), xytext=(0, 10), textcoords="offset points",
                        fontsize=8, fontweight="bold", ha="center", color=color, zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    if fondo is None:
        fondo = descargar_fondo_mapa(gdf_nodos)
    imagen, extent, _ = fondo
    ax.imshow(imagen, extent=extent, zorder=1)

    # imshow ajusta los límites al extent de la imagen por defecto -- se
    # fijan explícito DESPUÉS para que manden los nuestros (un poco de
    # margen alrededor de los 4 nodos), no el tamaño del mosaico descargado.
    xmin, ymin, xmax, ymax = gdf_nodos.total_bounds
    mx, my = (xmax - xmin) * 0.35, (ymax - ymin) * 0.35
    ax.set_xlim(xmin - mx, xmax + mx)
    ax.set_ylim(ymin - my, ymax + my)

    ax.set_axis_off()
    hh, mm = divmod(int(frame["tiempo"]["minuto_del_dia"]), 60)
    ax.set_title(f"Despacho de barcos -- {hh:02d}:{mm:02d}", fontsize=13)


def animar_corrida(historial: list[dict], gdf_nodos, gdf_rutas, matriz_tiempos, capacidad_barco: int, out_path, fps: int = 6):
    """Guarda un GIF (`PillowWriter`, siempre disponible, no depende de
    ffmpeg) con un frame por paso del historial. El mapa de fondo se
    descarga UNA vez (`descargar_fondo_mapa`) y se reusa en los ~90 frames
    -- pedirlo de nuevo en cada frame sería 90 llamadas de red por
    animación, lento e innecesariamente frágil.
    """
    fondo = descargar_fondo_mapa(gdf_nodos)
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

    def _actualizar(i):
        dibujar_frame(ax, historial[i], gdf_nodos, gdf_rutas, matriz_tiempos, capacidad_barco, fondo=fondo)

    anim = FuncAnimation(fig, _actualizar, frames=len(historial), interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# -- Gráficas de métricas (docs/especificacion_simulador_rl.md, "presenta con tablas y gráficas") --

def graficar_perfil_espera(env, ax=None):
    """Personas esperando (suma de las 12 colas) en cada paso -- de
    `env.historial_estados`, no requiere volver a correr nada.
    """
    minutos = [f["tiempo"]["minuto_del_dia"] for f in env.historial_estados]
    esperando = [sum(v["personas"] for v in f["demanda"].values()) for f in env.historial_estados]
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.plot(minutos, esperando, color="#e67e22", linewidth=1.8)
    ax.fill_between(minutos, esperando, color="#e67e22", alpha=0.15)
    ax.set_xlabel("Minuto del día")
    ax.set_ylabel("Personas esperando")
    ax.set_title("Perfil temporal de personas esperando")
    return ax


def graficar_ocupacion_flota(env, ax=None):
    """Ocupación de cada barco en el tiempo -- de `env.historial_estados`."""
    minutos = [f["tiempo"]["minuto_del_dia"] for f in env.historial_estados]
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    for i in range(env.num_barcos):
        ocup = [f["barcos"][i]["ocupacion"] for f in env.historial_estados]
        ax.plot(minutos, ocup, linewidth=1.5, label=f"barco_{i}")
    ax.axhline(env.capacidad_barco, color="grey", linestyle="--", linewidth=1, label="capacidad")
    ax.set_xlabel("Minuto del día")
    ax.set_ylabel("Ocupación (personas)")
    ax.set_title("Ocupación de la flota en el tiempo")
    ax.legend(fontsize=8)
    return ax


def graficar_heatmap_cumplimiento(env, ax=None):
    """Heatmap 4x4 de % de cumplimiento (atendidas/generadas) por par
    origen-destino -- de `metricas.metricas_por_par`.
    """
    df = met.metricas_por_par(env)
    nodos = env.nodos
    matriz = pd.DataFrame(np.nan, index=nodos, columns=nodos)
    for _, r in df.iterrows():
        o, d = r["par"].split("->")
        matriz.loc[o, d] = r["pct_cumplimiento"]

    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    im = ax.imshow(matriz.values, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(nodos)))
    ax.set_xticklabels([CODIGO_CORTO[n] for n in nodos])
    ax.set_yticks(range(len(nodos)))
    ax.set_yticklabels([CODIGO_CORTO[n] for n in nodos])
    ax.set_xlabel("Destino")
    ax.set_ylabel("Origen")
    ax.set_title("% cumplimiento por par (atendidas / generadas)")
    for i in range(len(nodos)):
        for j in range(len(nodos)):
            v = matriz.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="% cumplimiento", fraction=0.046)
    return ax


def graficar_desglose_recompensa(env, ax=None):
    """Desglose de recompensa (incomodidad, pérdida, movimiento) en el
    tiempo -- de `env.log_recompensa`.
    """
    df = met.desglose_recompensa(env)
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.stackplot(
        df["minuto"], df["incomodidad"], df["perdida"], df["movimiento"],
        labels=["incomodidad", "pérdida", "movimiento"],
        colors=["#3498db", "#c0392b", "#7f8c8d"], alpha=0.85,
    )
    ax.set_xlabel("Minuto del día")
    ax.set_ylabel("Penalización del paso (positiva = magnitud)")
    ax.set_title("Desglose de la recompensa en el tiempo")
    ax.legend(fontsize=8, loc="upper left")
    return ax


def graficar_perdidas_por_nodo(env, ax=None):
    """Barras de pérdidas por par -- de `metricas.perdidas_por_nodo`."""
    df = met.perdidas_por_nodo(env)
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    if df.empty:
        ax.text(0.5, 0.5, "Sin pérdidas en esta corrida", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(df["par"], df["perdidas"], color="#c0392b")
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Personas perdidas")
    ax.set_title("Dónde se pierde la gente (por par origen-destino)")
    return ax


# -- Inspector de un minuto concreto -------------------------------------------

def _frame_mas_cercano(env, minuto: float) -> dict:
    return min(env.historial_estados, key=lambda f: abs(f["tiempo"]["minuto_del_dia"] - minuto))


def inspeccionar(minuto: float, env, gdf_nodos, gdf_rutas, matriz_tiempos, fondo=None):
    """Pausa en el minuto pedido (usa el paso más cercano de
    `env.historial_estados`) y muestra el estado completo: barcos, las 12
    colas, qué decidió cada barco ese paso, y la recompensa con su
    desglose. Imprime texto y devuelve la figura dibujada.

    Funciona siempre (no requiere un kernel de Jupyter vivo) -- para un
    control deslizante real, ver `inspeccionar_interactivo`.
    """
    frame = _frame_mas_cercano(env, minuto)
    t = frame["tiempo"]["minuto_del_dia"]

    decisiones = [e for e in env.log_eventos if e["tipo"] == "decision" and abs(e["minuto"] - t) < 1e-6]
    recompensa = next((r for r in env.log_recompensa if abs(r["minuto"] - t) < 1e-6), None)

    print(f"=== Minuto {t:.0f} ===")
    print("Barcos:")
    for i, b in enumerate(frame["barcos"]):
        dec = next((d["decision"] for d in decisiones if d["barco_id"] == f"barco_{i}"), None)
        estado_txt = "libre" if b["libre"] else f"{b['origen']}->{b['destino']} ({b['min_para_llegar']:.0f} min)"
        dec_txt = f", decidió: {dec}" if dec is not None else ""
        print(f"  barco_{i}: {estado_txt}, ocupación={b['ocupacion']}{dec_txt}")

    print("Colas (pares con gente esperando):")
    for clave, v in frame["demanda"].items():
        if v["personas"] > 0:
            print(f"  {clave}: {v['personas']} personas, la más antigua lleva {v['espera_max']:.1f} min")

    if recompensa:
        print(f"Recompensa del paso: {recompensa['total']:.3f} "
              f"(incomodidad={recompensa['incomodidad']:.3f}, perdida={recompensa['perdida']:.3f}, "
              f"movimiento={recompensa['movimiento']:.3f})")

    fig, ax = plt.subplots(figsize=(7, 7))
    dibujar_frame(ax, frame, gdf_nodos, gdf_rutas, matriz_tiempos, env.capacidad_barco, fondo=fondo)
    return fig


def inspeccionar_interactivo(env, gdf_nodos, gdf_rutas, matriz_tiempos):
    """Envuelve `inspeccionar` con un control deslizante de `ipywidgets`.

    IMPORTANTE: el slider solo aparece y funciona si este notebook se abre
    con un kernel de Jupyter VIVO (VS Code / Jupyter Lab) -- un notebook ya
    ejecutado y guardado (como los que se generan con `nbconvert`) no tiene
    kernel corriendo, así que el slider no se puede mover ahí. Para
    cualquier otro uso (scripts, notebooks ejecutados), usar `inspeccionar`
    directo con un número.
    """
    import ipywidgets as widgets
    from IPython.display import display

    minutos = [f["tiempo"]["minuto_del_dia"] for f in env.historial_estados]
    fondo = descargar_fondo_mapa(gdf_nodos)
    slider = widgets.IntSlider(min=int(min(minutos)), max=int(max(minutos)), step=1, description="Minuto")

    def _actualizar(minuto):
        inspeccionar(minuto, env, gdf_nodos, gdf_rutas, matriz_tiempos, fondo=fondo)
        plt.show()

    return widgets.interact(_actualizar, minuto=slider)
