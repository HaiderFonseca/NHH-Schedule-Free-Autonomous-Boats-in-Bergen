"""Patrón origen-destino por franja horaria (Tarea 3).

Adapta el método multiplicativo de Braathen, Goez & Guajardo (2024) §4.1
("Generating passenger groups and routes"). En el paper:

    pop_pg := ln(1 + p_origen * p_destino * p_dia * p_hora)

con 4 puntajes de popularidad puestos a criterio (uno por estación, el mismo
si la estación juega de origen o de destino). Nuestro aporte: sustituir esos
puntajes de estación por masas reales de SSB (Tareas 0-2) — población si el
nodo juega de origen, empleo si juega de destino (o al revés, según la
franja) — y agregar un factor de distancia decreciente que el paper no tiene
en esta fórmula (§4.1 no pondera por distancia; sí la usa después para filtrar
"rutas realistas", que es un problema distinto).

Es importante para la trazabilidad: la fórmula multiplicativa es préstamo
metodológico del paper; los puntajes de estación NO lo son (el paper no
publica una tabla de puntajes) — se derivan de SSB, que es aporte propio de
esta tesis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cargar_matriz_tiempos(cfg: dict, base_dir: Path) -> pd.DataFrame:
    """Lee la matriz de tiempos navegables (bergen-boats paso 02) — no se
    recalculan tiempos de viaje aquí, se reusa esa fuente única.
    """
    path = (base_dir / cfg["fuentes_externas"]["matriz_tiempos_min"]).resolve()
    return pd.read_csv(path, index_col=0)


def normalizar_masas(resumen_masas: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Agrega columnas `poblacion_norm` y `empleo_norm` a la tabla de masas por
    nodo (Tarea 1), normalizadas a (0, 1] dividiendo por el máximo entre los 4
    nodos. Aplica también el `factor_universitario` (placeholder, Tarea 3) a
    la masa de empleo normalizada.
    """
    df = resumen_masas.copy()
    df["poblacion_norm"] = df["poblacion_total"] / df["poblacion_total"].max()
    df["empleo_norm"] = df["empleo_total"] / df["empleo_total"].max()

    factor_uni = cfg["masas"]["factor_universitario"]
    df["factor_universitario"] = df.index.map(lambda nid: factor_uni.get(nid, 1.0))
    df["empleo_norm_ajustado"] = df["empleo_norm"] * df["factor_universitario"]

    # `mixta` (franjas valle/noche): promedio simple de población y empleo normalizados.
    df["mixta_norm"] = (df["poblacion_norm"] + df["empleo_norm_ajustado"]) / 2.0
    return df


def _peso_por_rol(masas_norm: pd.DataFrame, nodo_id: str, rol: str) -> float:
    if rol == "poblacion":
        return masas_norm.loc[nodo_id, "poblacion_norm"]
    if rol == "empleo":
        return masas_norm.loc[nodo_id, "empleo_norm_ajustado"]
    if rol == "mixta":
        return masas_norm.loc[nodo_id, "mixta_norm"]
    raise ValueError(f"rol desconocido: {rol}")


def construir_intensidad_od(cfg: dict, masas_norm: pd.DataFrame, matriz_tiempos: pd.DataFrame) -> pd.DataFrame:
    """Construye la intensidad O-D para cada franja horaria y cada par
    ordenado (origen != destino), en formato "long" (una fila por
    franja-origen-destino).

    intensidad = peso_origen * peso_destino * factor_distancia * factor_volumen_franja

    (el factor de día de semana NO se aplica aquí — es un escalar del día
    completo, se aplica en la etapa de llegadas, Tarea 4)
    """
    tau_min = cfg["factor_distancia"]["tau_min"]
    ids_nodos = masas_norm.index.tolist()

    filas = []
    for franja in cfg["franjas_horarias"]:
        for o in ids_nodos:
            for d in ids_nodos:
                if o == d:
                    continue
                peso_o = _peso_por_rol(masas_norm, o, franja["rol_origen"])
                peso_d = _peso_por_rol(masas_norm, d, franja["rol_destino"])
                tiempo_min = matriz_tiempos.loc[o, d]
                factor_dist = np.exp(-tiempo_min / tau_min)
                intensidad = peso_o * peso_d * factor_dist * franja["factor_volumen"]

                filas.append({
                    "franja": franja["id"],
                    "hora_inicio": franja["horas"][0],
                    "hora_fin": franja["horas"][1],
                    "origen": o,
                    "destino": d,
                    "rol_origen": franja["rol_origen"],
                    "rol_destino": franja["rol_destino"],
                    "peso_origen": peso_o,
                    "peso_destino": peso_d,
                    "tiempo_viaje_min": tiempo_min,
                    "factor_distancia": factor_dist,
                    "factor_volumen_franja": franja["factor_volumen"],
                    "intensidad": intensidad,
                })

    return pd.DataFrame(filas)


def resumen_direccion_bryggen(intensidad_od: pd.DataFrame, franja_id: str, nodo_hub: str = "bryggen") -> dict:
    """Para una franja dada, compara la intensidad total hacia `nodo_hub` vs.
    la intensidad total saliendo de `nodo_hub` — para verificar (no forzar) si
    aparece la asimetría direccional esperada (p. ej. mañana hacia Bryggen).
    """
    sub = intensidad_od[intensidad_od["franja"] == franja_id]
    hacia = sub.loc[sub["destino"] == nodo_hub, "intensidad"].sum()
    desde = sub.loc[sub["origen"] == nodo_hub, "intensidad"].sum()
    total = hacia + desde
    return {
        "franja": franja_id,
        "intensidad_hacia_hub": hacia,
        "intensidad_desde_hub": desde,
        "pct_hacia_hub": round(100 * hacia / total, 1) if total else float("nan"),
        "pct_desde_hub": round(100 * desde / total, 1) if total else float("nan"),
    }
