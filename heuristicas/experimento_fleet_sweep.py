"""Experimento completo: H0 / H1 (reserva persistente) / H2 (costo + partición
local) / H3 (costo global) -- definicion CERRADA tras la auditoria y los 5
casos controlados (ver heuristicas/*/src/politica_hX.py, cada uno documenta
su regla exacta).

Reusa la MISMA demanda ya generada para el trabajo anterior
(`politica_base/output/escalon_dia_10pct/grupos_seed*.csv` -- dia completo,
10% poblacion oficial, 5 semillas de evaluacion) para que los resultados
sean directamente comparables con el punto de partida de esta fase.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
SIMULADOR_DIR = BASE_DIR / "simulador"
BB_DIR = BASE_DIR / "bergen-boats"
POLITICA_BASE_DIR = BASE_DIR / "politica_base"
HEURISTICAS_DIR = BASE_DIR / "heuristicas"

sys.path.insert(0, str(SIMULADOR_DIR / "src"))
sys.path.insert(0, str(POLITICA_BASE_DIR / "src"))
sys.path.insert(0, str(HEURISTICAS_DIR / "comun" / "src"))
sys.path.insert(0, str(HEURISTICAS_DIR / "h1_reserva" / "src"))
sys.path.insert(0, str(HEURISTICAS_DIR / "h2_costo_local" / "src"))
sys.path.insert(0, str(HEURISTICAS_DIR / "h3_costo_global" / "src"))

from env import SimuladorBarcosBergen                # noqa: E402
from politica_base import asignar_flota               # noqa: E402
from politica_h1 import asignar_flota_h1               # noqa: E402
from politica_h2 import asignar_flota_h2               # noqa: E402
from politica_h3 import asignar_flota_h3               # noqa: E402
import metricas as met                                  # noqa: E402

OUT_DIR = HEURISTICAS_DIR / "outputs" / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEMANDA_DIR = POLITICA_BASE_DIR / "output" / "escalon_dia_10pct"

cfg_sim = yaml.safe_load(open(SIMULADOR_DIR / "config" / "instance.yaml", encoding="utf-8"))
cfg_bb = yaml.safe_load(open(BB_DIR / "config" / "instance.yaml", encoding="utf-8"))

nodos = [n["id"] for n in cfg_bb["nodos_demanda"]]
matriz_tiempos = pd.read_csv(BB_DIR / "02_ruteo_navegable" / "output" / "matriz_tiempos_min.csv", index_col=0)
capacidad = cfg_bb["flota"]["capacidad_pasajeros"]

cfg_escalon = cfg_sim["escalones"]["escalon_dia_10pct"]
hora_ini_min, hora_fin_min = cfg_escalon["horas"][0] * 60, cfg_escalon["horas"][1] * 60
FLOTAS = cfg_escalon["num_barcos_barrido"]                  # [4, 6, 8, 10, 12]
SEMILLAS_EVAL = cfg_sim["agente"]["evaluacion"]["semillas"]  # [1001..1005]

POLITICAS_CON_RESERVA = {"H1": asignar_flota_h1, "H2": asignar_flota_h2, "H3": asignar_flota_h3}


def correr(grupos_df: pd.DataFrame, num_barcos: int, nombre_politica: str):
    env = SimuladorBarcosBergen(
        grupos_df=grupos_df, matriz_tiempos=matriz_tiempos, nodos=nodos,
        num_barcos=num_barcos, capacidad_barco=capacidad,
        nodo_inicial=cfg_bb["flota"]["nodo_inicial"], paso_tiempo_min=cfg_sim["paso_tiempo_min"],
        hora_inicio_min=hora_ini_min, hora_fin_min=hora_fin_min,
        cfg_recompensa=cfg_sim["recompensa"], unidad_demanda=cfg_sim["unidad_demanda"],
    )
    obs, info = env.reset(seed=cfg_sim["semilla"])
    reservas: dict[str, tuple[str, str]] = {}  # estado del BUCLE DE CONTROL, no de la politica

    while True:
        estado = info["_estado_obj"]
        libres = [b for b in estado.barcos if b.libre]

        if nombre_politica == "H0":
            decisiones = asignar_flota(libres, estado, matriz_tiempos, capacidad, cfg_sim["recompensa"])
        else:
            politica_fn = POLITICAS_CON_RESERVA[nombre_politica]
            decisiones, reservas = politica_fn(libres, estado, matriz_tiempos, capacidad, reservas, cfg_sim["recompensa"])

        accion = np.array([
            env.codificar_accion_barco(decisiones[b.id]) if b.libre else 0
            for b in estado.barcos
        ])
        obs, r, terminated, truncated, info = env.step(accion)
        if truncated or terminated:
            break

    return env


def fila_metricas(env, politica: str, num_barcos: int, semilla: int) -> dict:
    cons = met.verificar_conservacion(env)
    glob = met.metricas_globales(env)
    usr = met.metricas_por_usuario(env)
    por_barco = met.metricas_por_barco(env)
    backlog = met.sin_atender_al_final_por_par(env)
    return {
        "politica": politica, "num_barcos": num_barcos, "semilla": semilla,
        "generadas": cons["generadas"], "atendidas": cons["atendidas"],
        "conservacion_cuadra": cons["cuadra"],
        "pct_atendidas": glob["pct_atendidas"],
        "backlog_final_total": int(backlog["sin_atender"].sum()) if len(backlog) else 0,
        "espera_media_min": glob["espera_media_min"],
        "espera_p50_min": usr["espera_min"]["p50"], "espera_p95_min": usr["espera_min"]["p95"],
        "espera_max_min": usr["espera_min"]["max"],
        "sistema_medio_min": glob["sistema_medio_min"],
        "sistema_p50_min": usr["sistema_min"]["p50"], "sistema_p95_min": usr["sistema_min"]["p95"],
        "sistema_max_min": usr["sistema_min"]["max"],
        "movimientos_totales": int(por_barco["movimientos"].sum()),
        "ocupacion_media_flota": float(por_barco["ocupacion_media"].mean()),
        "pct_esperando_flota": float(por_barco["pct_esperando"].mean()),
    }


def main():
    print(f"Flotas: {FLOTAS}. Semillas eval: {SEMILLAS_EVAL}. Capacidad: {capacidad}.")
    grupos_por_semilla = {
        semilla: pd.read_csv(DEMANDA_DIR / f"grupos_seed{semilla}.csv") for semilla in SEMILLAS_EVAL
    }
    for semilla, g in grupos_por_semilla.items():
        print(f"  semilla {semilla}: {len(g)} grupos, {g['tamano_grupo'].sum()} personas (reusada de politica_base/output/escalon_dia_10pct/)")

    filas = []
    t0 = time.time()
    total = 4 * len(FLOTAS) * len(SEMILLAS_EVAL)
    i = 0
    for semilla, grupos in grupos_por_semilla.items():
        for num_barcos in FLOTAS:
            for politica in ["H0", "H1", "H2", "H3"]:
                i += 1
                env = correr(grupos, num_barcos, politica)
                filas.append(fila_metricas(env, politica, num_barcos, semilla))
                print(f"  [{i}/{total}] {politica} x {num_barcos} barcos x semilla {semilla} -- "
                      f"espera media {filas[-1]['espera_media_min']:.2f} min "
                      f"({time.time()-t0:.0f}s acumulados)")

    resultados = pd.DataFrame(filas)
    resultados.to_csv(OUT_DIR / "resultados_h0_h1_h2_h3.csv", index=False)
    assert resultados["conservacion_cuadra"].all(), "Conservacion NO cuadra en alguna corrida"

    columnas_num = [c for c in resultados.columns if c not in ("politica", "num_barcos", "semilla", "conservacion_cuadra")]
    resumen = resultados.groupby(["politica", "num_barcos"])[columnas_num].agg(["mean", "std"])
    resumen.columns = [f"{c}_{s}" for c, s in resumen.columns]
    resumen = resumen.reset_index()
    resumen.to_csv(OUT_DIR / "resumen_h0_h1_h2_h3.csv", index=False)

    print(f"\nListo en {time.time()-t0:.0f}s -> {OUT_DIR}")
    print(resumen[["politica", "num_barcos", "espera_media_min_mean", "espera_p95_min_mean",
                    "espera_max_min_mean", "pct_atendidas_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
