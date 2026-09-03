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
import plotly.graph_objects as go
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


def animar_corrida(historial: list[dict], gdf_nodos, gdf_rutas, matriz_tiempos, capacidad_barco: int, out_path, fps: int = 6, fondo=None):
    """Guarda un GIF (`PillowWriter`, siempre disponible, no depende de
    ffmpeg) con un frame por paso del historial. El mapa de fondo se
    descarga UNA vez (`descargar_fondo_mapa`) y se reusa en los ~90 frames
    -- pedirlo de nuevo en cada frame sería 90 llamadas de red por
    animación, lento e innecesariamente frágil. Si el notebook ya descargó
    un `fondo` para otra celda (inspector, reproductor), pásalo aquí para
    no volver a pedirlo -- el servidor de tiles a veces tarda, y no hay
    razón para pedir el mismo mapa de fondo más de una vez por notebook.
    """
    if fondo is None:
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
#
# Interactivas (Plotly, no matplotlib): zoom/pan nativo, y la leyenda ya
# permite aislar una serie (click = ocultar/mostrar, doble click = aislar) --
# es como se resuelve "elegir qué barco ver o todos", sin widgets aparte.
# En inglés (títulos/ejes/leyendas), a partir de esta reescritura.

def _eje_tiempo(minutos: list[float], paso_tiempo_min: float, limites_episodio: list[int] | None = None):
    """Eje X para una serie de tiempo del simulador, a partir de la lista de
    `minuto_del_dia` (uno por paso). Se usa para las 3 gráficas de series de
    tiempo de abajo -- misma lógica, no se repite en cada una.

    `limites_episodio`: cuántos frames aportó cada corrida fuente, en orden
    (`metricas.CorridaCombinada.limites_episodio`, ver `combinar_corridas`).
    Si se da, los límites de episodio se toman de ahí EXACTOS -- si no (un
    `env` de un solo episodio, nunca combinado), no hace falta ninguno.

    **Por qué no se detectan los límites mirando si `minuto_del_dia` baja**
    (versión anterior de esta función): para un escalón cuyo `hora_fin_min`
    es un múltiplo exacto de 1440 (medianoche -- p.ej. escalón 2/3, que
    terminan a las 24:00), el ÚLTIMO frame de CADA episodio cae justo en
    `t_actual_min == 1440`, y `minuto_del_dia = t_actual_min % 1440` da `0`
    -- indistinguible de un verdadero cambio de episodio. Eso hacía que el
    escalón 3 (7 días) detectara 8 "episodios" en vez de 7 (uno de sobra,
    con los límites corridos un frame). Con `limites_episodio` exacto, este
    caso límite no puede volver a pasar -- no se adivina nada de los datos.

    Devuelve `(x, tick_vals, tick_text, hover)`:
    - `x`: índice de paso (0..N-1) -- SIEMPRE monótono, nunca se superpone,
      a diferencia de `minuto_del_dia` crudo (que se reinicia en 0 cada
      episodio). Antes de este cambio, una corrida combinada de varios
      episodios (escalón 3 = 7 días, o la comparación = 5 semillas de
      evaluación, ver `metricas.combinar_corridas`) dibujaba la MISMA línea
      de un barco superpuesta varias veces en el mismo rango 0-1439 --
      parecía "varias líneas del mismo barco". Usar el índice de paso lo
      resuelve de raíz.
    - `tick_vals`/`tick_text`: para un solo episodio, ticks de hora real
      (`HH:MM`) cada ~1h -- así se sabe qué hora es mirando el eje, tal como
      se pidió. Para una corrida combinada, los ticks marcan el inicio de
      cada episodio ("Ep 1", "Ep 2", ...) -- meter 7 días de horas legibles
      en un solo eje no cabe, así que la hora exacta de cada punto se
      muestra al pasar el mouse (`hover`) en vez de en el eje.
    - `hover`: texto por punto (`"09:34"` o `"Ep 3, 09:34"`) para el tooltip.
    """
    n = len(minutos)
    x = list(range(n))
    combinado = limites_episodio is not None and len(limites_episodio) > 1
    segmentos = [sum(limites_episodio[:k]) for k in range(len(limites_episodio))] if combinado else [0]

    def _hhmm(m):
        hh, mm = divmod(int(round(m)), 60)
        return f"{hh:02d}:{mm:02d}"

    if not combinado:
        pasos_por_hora = max(1, round(60 / paso_tiempo_min))
        tick_vals = list(range(0, n, pasos_por_hora))
        if tick_vals[-1] != n - 1:
            tick_vals.append(n - 1)
        tick_text = [_hhmm(minutos[i]) for i in tick_vals]
        hover = [_hhmm(m) for m in minutos]
    else:
        tick_vals = segmentos
        tick_text = [f"Ep {k + 1}" for k in range(len(segmentos))]
        episodio_de = [0] * n
        for k, inicio in enumerate(segmentos):
            fin = segmentos[k + 1] if k + 1 < len(segmentos) else n
            for i in range(inicio, fin):
                episodio_de[i] = k
        hover = [f"Ep {episodio_de[i] + 1}, {_hhmm(minutos[i])}" for i in range(n)]

    return x, tick_vals, tick_text, hover


def graficar_perfil_espera(env) -> go.Figure:
    """People waiting (sum of the 12 queues) at each step -- from
    `env.historial_estados`, no need to rerun anything.
    """
    minutos = [f["tiempo"]["minuto_del_dia"] for f in env.historial_estados]
    x, tick_vals, tick_text, hover = _eje_tiempo(minutos, env.paso_tiempo_min, getattr(env, "limites_episodio", None))
    esperando = [sum(v["personas"] for v in f["demanda"].values()) for f in env.historial_estados]

    fig = go.Figure(go.Scatter(
        x=x, y=esperando, mode="lines", fill="tozeroy",
        line=dict(color="#e67e22", width=2),
        name="waiting",
        customdata=hover, hovertemplate="%{customdata}<br>waiting: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="People waiting over time", xaxis_title="Time", yaxis_title="People waiting",
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),
        template="plotly_white",
    )
    return fig


def graficar_ocupacion_flota(env) -> go.Figure:
    """Occupancy of each boat over time -- from `env.historial_estados`.
    One line per boat; click a legend entry to hide/show it, double-click
    to isolate it (Plotly's native legend behaviour) -- that's how to look
    at just one boat, or all of them.
    """
    minutos = [f["tiempo"]["minuto_del_dia"] for f in env.historial_estados]
    x, tick_vals, tick_text, hover = _eje_tiempo(minutos, env.paso_tiempo_min, getattr(env, "limites_episodio", None))

    fig = go.Figure()
    for i in range(env.num_barcos):
        ocup = [f["barcos"][i]["ocupacion"] for f in env.historial_estados]
        fig.add_trace(go.Scatter(
            x=x, y=ocup, mode="lines", name=f"boat_{i}",
            customdata=hover, hovertemplate="%{customdata}<br>occupancy: %{y}<extra>%{fullData.name}</extra>",
        ))
    fig.add_hline(y=env.capacidad_barco, line_dash="dash", line_color="grey",
                  annotation_text="capacity", annotation_position="top left")
    fig.update_layout(
        title="Fleet occupancy over time", xaxis_title="Time", yaxis_title="Occupancy (people)",
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),
        template="plotly_white",
    )
    return fig


def graficar_heatmap_cumplimiento(env) -> go.Figure:
    """4x4 heatmap of % served (served/generated, within the run window) by
    origin-destination pair -- from `metricas.metricas_por_par`.
    """
    df = met.metricas_por_par(env)
    nodos = env.nodos
    matriz = pd.DataFrame(np.nan, index=nodos, columns=nodos)
    for _, r in df.iterrows():
        o, d = r["par"].split("->")
        matriz.loc[o, d] = r["pct_atendidas"]

    etiquetas = [CODIGO_CORTO[n] for n in nodos]
    texto = [[f"{v:.0f}%" if not np.isnan(v) else "" for v in fila] for fila in matriz.values]
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=etiquetas, y=etiquetas, zmin=0, zmax=100, colorscale="RdYlGn",
        text=texto, texttemplate="%{text}",
        hovertemplate="origin %{y} -> destination %{x}<br>%{z:.1f}% served<extra></extra>",
        colorbar=dict(title="% served"),
    ))
    fig.update_layout(
        title="% served by O-D pair (within the run window)",
        xaxis_title="Destination", yaxis_title="Origin",
        template="plotly_white",
    )
    return fig


def graficar_desglose_recompensa(env) -> go.Figure:
    """Reward breakdown over time -- from `env.log_recompensa`. Lines, not
    a stacked area: since the delivery bonus (`recompensa.py`,
    `premio_por_persona_entregada`) is positive and discomfort/movement are
    negative, a stackplot of mixed signs would be misleading -- each
    component keeps its true sign here, plus the total reward line.
    """
    df = met.desglose_recompensa(env)
    x, tick_vals, tick_text, hover = _eje_tiempo(df["minuto"].tolist(), env.paso_tiempo_min, getattr(env, "limites_episodio", None))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=-df["incomodidad"], mode="lines", name="discomfort",
                              line=dict(color="#3498db"), customdata=hover,
                              hovertemplate="%{customdata}<br>discomfort: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=-df["movimiento"], mode="lines", name="movement",
                              line=dict(color="#7f8c8d"), customdata=hover,
                              hovertemplate="%{customdata}<br>movement: %{y:.3f}<extra></extra>"))
    if "entrega" in df.columns and df["entrega"].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=x, y=df["entrega"], mode="lines", name="delivery bonus",
                                  line=dict(color="#27ae60"), customdata=hover,
                                  hovertemplate="%{customdata}<br>delivery bonus: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=df["total"], mode="lines", name="total reward",
                              line=dict(color="black", dash="dot"), customdata=hover,
                              hovertemplate="%{customdata}<br>total: %{y:.3f}<extra></extra>"))
    fig.update_layout(
        title="Reward breakdown over time", xaxis_title="Time", yaxis_title="Reward component",
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),
        template="plotly_white",
    )
    return fig


def graficar_sin_atender_al_final(env) -> go.Figure:
    """Backlog at the end of the run (people still waiting or on board
    when the run ended), by pair -- from
    `metricas.sin_atender_al_final_por_par`. Not "losses" (the simulator
    never drops anyone, see `env.py`): it's people who would have been
    served if the run had continued a bit longer.
    """
    df = met.sin_atender_al_final_por_par(env)
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="Everyone served by the end of the run", showarrow=False,
                            xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=14))
    else:
        fig.add_trace(go.Bar(x=df["par"], y=df["sin_atender"], marker_color="#e67e22"))
    fig.update_layout(
        title="Backlog at the end of the run (by O-D pair)",
        xaxis_title="O-D pair", yaxis_title="People not yet served",
        template="plotly_white",
    )
    return fig


# -- Inspector de un minuto concreto -------------------------------------------

def _frame_mas_cercano(env, minuto: float) -> dict:
    return min(env.historial_estados, key=lambda f: abs(f["tiempo"]["minuto_del_dia"] - minuto))


def _texto_estado(env, frame: dict) -> str:
    """Arma el texto organizado de un paso: barcos (posición/ocupación/qué
    decidió cada uno), las 12 colas con gente esperando, y la recompensa
    con su desglose -- usado tanto por `inspeccionar` (imprime directo)
    como por `reproductor_interactivo` (lo muestra en el panel de texto).
    """
    t = frame["tiempo"]["minuto_del_dia"]
    decisiones = [e for e in env.log_eventos if e["tipo"] == "decision" and abs(e["minuto"] - t) < 1e-6]
    recompensa = next((r for r in env.log_recompensa if abs(r["minuto"] - t) < 1e-6), None)

    lineas = [f"=== Minuto {t:.0f} ===", "", "Barcos:"]
    for i, b in enumerate(frame["barcos"]):
        dec = next((d["decision"] for d in decisiones if d["barco_id"] == f"barco_{i}"), None)
        estado_txt = "libre" if b["libre"] else f"{b['origen']}->{b['destino']} ({b['min_para_llegar']:.0f} min)"
        dec_txt = f", decidió: {dec}" if dec is not None else ""
        lineas.append(f"  barco_{i}: {estado_txt}, ocupación={b['ocupacion']}{dec_txt}")

    lineas.append("")
    lineas.append("Colas (pares con gente esperando):")
    hay_cola = False
    for clave, v in frame["demanda"].items():
        if v["personas"] > 0:
            hay_cola = True
            lineas.append(f"  {clave}: {v['personas']} personas, la más antigua lleva {v['espera_max']:.1f} min")
    if not hay_cola:
        lineas.append("  (ninguna)")

    lineas.append("")
    if recompensa:
        lineas.append(f"Recompensa del paso: {recompensa['total']:.3f}")
        lineas.append(f"  incomodidad = {recompensa['incomodidad']:.3f}")
        lineas.append(f"  movimiento  = {recompensa['movimiento']:.3f}")

    return "\n".join(lineas)


def inspeccionar(minuto: float, env, gdf_nodos, gdf_rutas, matriz_tiempos, fondo=None):
    """Pausa en el minuto pedido (usa el paso más cercano de
    `env.historial_estados`) y muestra el estado completo: barcos, las 12
    colas, qué decidió cada barco ese paso, y la recompensa con su
    desglose. Imprime texto y devuelve la figura dibujada.

    Funciona siempre (no requiere un kernel de Jupyter vivo) -- para un
    reproductor con play/pausa y barra de tiempo real, ver
    `reproductor_interactivo`.
    """
    frame = _frame_mas_cercano(env, minuto)
    print(_texto_estado(env, frame))
    fig, ax = plt.subplots(figsize=(7, 7))
    dibujar_frame(ax, frame, gdf_nodos, gdf_rutas, matriz_tiempos, env.capacidad_barco, fondo=fondo)
    return fig


def reproductor_interactivo(env, gdf_nodos, gdf_rutas, matriz_tiempos, fps: int = 4, fondo=None):
    """Reproductor tipo video de la corrida completa: botón de
    reproducir/pausar, barra de tiempo para saltar a cualquier paso (hacia
    adelante o atrás), mapa animado, y un panel de texto con el estado
    completo de ese paso (barcos, colas, decisión de cada barco, y
    recompensa con su desglose) -- todo de `env.historial_estados`,
    `env.log_eventos` y `env.log_recompensa`, sin volver a correr nada.

    IMPORTANTE: como cualquier control interactivo de `ipywidgets`, esto
    SOLO funciona si el notebook se abre con un kernel de Jupyter VIVO (VS
    Code / Jupyter Lab, con "Run All" hecho por ti) -- un notebook ya
    ejecutado y guardado (como los que genera `nbconvert`, que es como se
    corrieron y guardaron los notebooks de este proyecto) no tiene kernel
    corriendo, así que los botones no van a responder ahí ni yo te lo
    puedo mostrar funcionando. Para inspeccionar un paso puntual sin
    depender de un kernel vivo, usar `inspeccionar(minuto, ...)`.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    historial = env.historial_estados
    n_pasos = len(historial)
    if fondo is None:
        fondo = descargar_fondo_mapa(gdf_nodos)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plt.close(fig)  # se muestra a mano dentro del Output, no como celda aparte

    out_mapa = widgets.Output()
    out_texto = widgets.Output(layout=widgets.Layout(width="380px", border="1px solid #ccc", padding="8px"))

    slider = widgets.IntSlider(min=0, max=n_pasos - 1, step=1, description="Paso", continuous_update=True)
    play = widgets.Play(min=0, max=n_pasos - 1, step=1, interval=int(1000 / fps), description="Reproducir")
    widgets.jslink((play, "value"), (slider, "value"))

    boton_atras = widgets.Button(description="◀ Paso anterior", layout=widgets.Layout(width="140px"))
    boton_adelante = widgets.Button(description="Paso siguiente ▶", layout=widgets.Layout(width="140px"))

    def _actualizar(change):
        i = slider.value
        frame = historial[i]
        with out_mapa:
            clear_output(wait=True)
            dibujar_frame(ax, frame, gdf_nodos, gdf_rutas, matriz_tiempos, env.capacidad_barco, fondo=fondo)
            display(fig)
        with out_texto:
            clear_output(wait=True)
            print(_texto_estado(env, frame))

    def _paso_adelante(b):
        slider.value = min(slider.value + 1, n_pasos - 1)

    def _paso_atras(b):
        slider.value = max(slider.value - 1, 0)

    boton_adelante.on_click(_paso_adelante)
    boton_atras.on_click(_paso_atras)

    slider.observe(_actualizar, names="value")
    _actualizar(None)  # dibuja el primer paso antes de que el usuario mueva nada

    controles = widgets.HBox([boton_atras, boton_adelante, play, slider])
    panel = widgets.HBox([out_mapa, out_texto])
    ui = widgets.VBox([controles, panel])
    display(ui)
    return ui
