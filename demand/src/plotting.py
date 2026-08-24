"""Gráficas de verificación para la generación de demanda.

Mismo estilo visual que bergen-boats (CartoDB, paleta consistente) para que
los mapas de los dos proyectos se vean como parte de un mismo trabajo.
"""
from __future__ import annotations

import contextily as cx
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLOR_NODO = "#1f6f8b"
COLOR_POBLACION = "#2f7d32"
COLOR_EMPLEO = "#c0392b"
COLOR_CIRCULO = "#444444"


def mapa_captacion_nodo(nombre_nodo, centro_utm, radio_m, celdas_pop, celdas_emp, ax=None, titulo=None):
    """Dibuja, sobre un basemap real, el círculo de captación de un nodo y las
    celdas de población/empleo capturadas dentro de él.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if len(celdas_pop):
        celdas_pop.plot(ax=ax, facecolor=COLOR_POBLACION, edgecolor="white", linewidth=0.3, alpha=0.55, zorder=2)
    if len(celdas_emp):
        celdas_emp.plot(ax=ax, facecolor="none", edgecolor=COLOR_EMPLEO, linewidth=1.1, hatch="//", alpha=0.85, zorder=3)

    circulo = centro_utm.buffer(radio_m)
    gpd_circulo_x, gpd_circulo_y = circulo.exterior.xy
    ax.plot(gpd_circulo_x, gpd_circulo_y, color=COLOR_CIRCULO, linewidth=1.8, linestyle="--", zorder=4)

    ax.scatter([centro_utm.x], [centro_utm.y], color=COLOR_NODO, edgecolor="white", s=180, zorder=5)
    ax.annotate(
        nombre_nodo, xy=(centro_utm.x, centro_utm.y), xytext=(10, 10), textcoords="offset points",
        fontsize=12, fontweight="bold", color=COLOR_NODO, zorder=6,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    margen = radio_m * 1.5
    ax.set_xlim(centro_utm.x - margen, centro_utm.x + margen)
    ax.set_ylim(centro_utm.y - margen, centro_utm.y + margen)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs="EPSG:32633", attribution_size=6)
    ax.set_axis_off()
    ax.set_title(titulo or f"Radio de captación — {nombre_nodo} ({radio_m:.0f} m)", fontsize=12)

    leyenda = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_POBLACION, alpha=0.55, markersize=12, label="Celdas de población capturadas"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor=COLOR_EMPLEO, markerfacecolor="none", markersize=12, label="Celdas de empleo capturadas"),
        Line2D([0], [0], color=COLOR_CIRCULO, linestyle="--", label=f"Radio de captación ({radio_m:.0f} m)"),
    ]
    ax.legend(handles=leyenda, loc="lower left", fontsize=8, frameon=True)

    return fig, ax
