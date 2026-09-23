"""Estudio de flota (fleet sweep) H0 vs. H1 -- dia completo, demanda oficial
(10% poblacion), 4/6/8 barcos, capacidad 30 -- documento de diseño de la fase
heuristica fuerte, secciones G-I.

Script standalone (no notebook) por simplicidad de mantenimiento -- reusa
exactamente el mismo patron de generacion de demanda que ya usan los
notebooks de politica_base/ (`politica_base/notebooks/02_escalon_2_metricas.ipynb`,
celda de reproduccion por semilla) y las mismas funciones de metricas.py, sin
inventar nada nuevo salvo la orquestacion del barrido.

Que corre:
- Genera demanda de un dia completo (6:00-24:00) al 10% oficial de poblacion
  (`demand/config/instance.yaml`, sin overrides) para cada semilla.
- Para cada (politica, num_barcos, semilla): corre el motor, junta metricas
  via `metricas.py` (sin duplicar logica de metricas).
- Politicas del ablation completo (documento de diseño, seccion I): H0
  (politica_base.asignar_flota), H1a/H1b/H1c (politica_h1.asignar_flota_h1
  con los toggles correspondientes).
- Guarda:
  - `output/escalon_dia_10pct/grupos_seed{semilla}.csv` -- demanda generada,
    una vez por semilla (se reusa para las 4 politicas x 3 flotas).
  - `output/escalon_dia_10pct/resultados_fleet_sweep.csv` -- una fila por
    (politica, num_barcos, semilla), todas las metricas pedidas.
  - `output/escalon_dia_10pct/resumen_fleet_sweep.csv` -- promedio +/- std
    sobre semillas, por (politica, num_barcos).
  - `output/escalon_dia_10pct/curva_servicio.png` -- espera media/P95 vs.
    num_barcos, una linea por politica (el entregable central del punto 8).

Semillas: separa calibracion (42 -- la semilla de desarrollo del proyecto,
`simulador/config/instance.yaml -> semilla`) de evaluacion (1001-1005, ya
definidas en `simulador/config/instance.yaml -> agente.evaluacion.semillas`,
reusadas aqui tal cual -- no se inventa un pool de semillas nuevo). El diseño
de la funcion de costo y del ablation ya se cerro mirando 42 (ver
`politica_base/output/escalon2` y el smoke test de verificacion) -- este
script corre SOLO sobre las semillas de evaluacion para la matriz final; 42
no entra en `resultados_fleet_sweep.csv`.
"""
from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
SIMULADOR_DIR = BASE_DIR / "simulador"
BB_DIR = BASE_DIR / "bergen-boats"
DEMAND_DIR = BASE_DIR / "demand"
POLITICA_DIR = BASE_DIR / "politica_base"
H2_DIR = BASE_DIR / "politica_h2"
H3_DIR = BASE_DIR / "politica_h3"
H0C_DIR = BASE_DIR / "politica_h0c"

sys.path.insert(0, str(SIMULADOR_DIR / "src"))
sys.path.insert(0, str(POLITICA_DIR / "src"))
sys.path.insert(0, str(H2_DIR / "src"))
sys.path.insert(0, str(H3_DIR / "src"))
sys.path.insert(0, str(H0C_DIR / "src"))
sys.path.insert(0, str(DEMAND_DIR / "src"))

from env import SimuladorBarcosBergen                       # noqa: E402
from politica_base import asignar_flota                     # noqa: E402
from politica_h1 import asignar_flota_h1                    # noqa: E402
from politica_h2 import asignar_flota_h2                    # noqa: E402
from politica_h3 import asignar_flota_h3                    # noqa: E402
from politica_h0c import asignar_flota_h0c                  # noqa: E402
import metricas as met                                       # noqa: E402
import llegadas as demand_llegadas                            # noqa: E402
import masas as demand_masas                                  # noqa: E402

OUT_DIR = POLITICA_DIR / "output" / "escalon_dia_10pct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg_sim = yaml.safe_load(open(SIMULADOR_DIR / "config" / "instance.yaml", encoding="utf-8"))
cfg_bb = yaml.safe_load(open(BB_DIR / "config" / "instance.yaml", encoding="utf-8"))
cfg_demand = demand_masas.cargar_config(DEMAND_DIR / "config" / "instance.yaml")

nodos = [n["id"] for n in cfg_bb["nodos_demanda"]]
matriz_tiempos = pd.read_csv(BB_DIR / "02_ruteo_navegable" / "output" / "matriz_tiempos_min.csv", index_col=0)
resumen_masas = pd.read_csv(DEMAND_DIR / "output" / "masas_por_nodo.csv", index_col="id")
intensidad_od = pd.read_csv(DEMAND_DIR / "output" / "matriz_intensidad_od.csv")
poblacion_total_zonas = resumen_masas["poblacion_total"].sum()
conexiones_fuertes = cfg_bb["garantia"]["conexiones_fuertes"]

cfg_escalon = cfg_sim["escalones"]["escalon_dia_10pct"]
hora_ini_min, hora_fin_min = cfg_escalon["horas"][0] * 60, cfg_escalon["horas"][1] * 60
capacidad = cfg_bb["flota"]["capacidad_pasajeros"]

FLOTAS = cfg_escalon["num_barcos_barrido"]                                  # [4, 6, 8, 10, 12]
SEMILLAS_EVAL = cfg_sim["agente"]["evaluacion"]["semillas"]                  # [1001..1005], reusadas tal cual

# H2 (politica_h2/src/politica_h2.py) es la version CORREGIDA de H1c, tras el
# diagnostico por par O-D que encontro que el termino tiempo_viaje de H1c
# castigaba sistematicamente los pares largos que salen del hub (bryggen ->
# kleppesto/laksevag, las "conexiones fuertes" de bergen-boats/config) frente
# a H0 -- ver politica_h2/src/politica_h2.py y politica_h2/README.md para el
# diagnostico completo y la correccion. H1a/H1b/H1c se mantienen en la
# matriz para que el ablation completo (incluyendo el defecto encontrado)
# quede documentado, no se borra nada.
# H0c (politica_h0c/) = la regla EXACTA de H0 (local siempre gana, remoto por
# mayor espera), sin ninguna funcion de costo -- solo con coordinacion
# conjunta real (orden por urgencia en vez de orden de lista) y reserva de
# capacidad entre pasos para reposicionamiento (que H0 no tenia). Es el punto
# de comparacion mas directo para saber cuanto de la mejora de H2/H3 viene de
# la COORDINACION sola, sin tocar la regla de decision de H0.
#
# H3 (politica_h3/) = misma arquitectura de H1/H2 (asignacion global +
# reservas) pero con costo = tiempo_pickup - espera_max (SIN suma de
# personas, sin ponderar por k) -- prueba si evitar la suma resuelve el
# problema de espera MAXIMA que H2 dejo peor que H0 en las 5 flotas.
POLITICAS: dict[str, callable] = {
    "H0": asignar_flota,
    "H0c": asignar_flota_h0c,
    "H1a": functools.partial(asignar_flota_h1, usar_reservas=False, ponderar_minutos=False),
    "H1b": functools.partial(asignar_flota_h1, usar_reservas=True, ponderar_minutos=False),
    "H1c": asignar_flota_h1,
    "H2": asignar_flota_h2,
    "H3": asignar_flota_h3,
}


def generar_grupos_dia(semilla: int) -> pd.DataFrame:
    """Dia completo, entre semana, a la escala oficial (10%, sin overrides
    -- `cfg_demand["demanda"]["porcentaje_poblacion_dia"]` ya es 0.10). Mismo
    patron que `politica_base/notebooks/02_escalon_2_metricas.ipynb`, celda
    de reproduccion por semilla (`generar_grupos_con_semilla`), solo que sin
    override de porcentaje (aqui SI se usa el oficial).
    """
    rng = np.random.default_rng(semilla)
    grupos = demand_llegadas.generar_llegadas_dia(
        cfg_demand, intensidad_od, conexiones_fuertes, poblacion_total_zonas,
        es_fin_de_semana=False, rng=rng, dia_id=f"sweep_{semilla}",
    )
    hora_ini, hora_fin = cfg_escalon["horas"]
    return grupos[(grupos["hora"] >= hora_ini) & (grupos["hora"] < hora_fin)].reset_index(drop=True)


def correr(grupos_df: pd.DataFrame, num_barcos: int, politica_fn) -> "object":
    env = SimuladorBarcosBergen(
        grupos_df=grupos_df, matriz_tiempos=matriz_tiempos, nodos=nodos,
        num_barcos=num_barcos, capacidad_barco=capacidad,
        nodo_inicial=cfg_bb["flota"]["nodo_inicial"], paso_tiempo_min=cfg_sim["paso_tiempo_min"],
        hora_inicio_min=hora_ini_min, hora_fin_min=hora_fin_min,
        cfg_recompensa=cfg_sim["recompensa"], unidad_demanda=cfg_sim["unidad_demanda"],
    )
    obs, info = env.reset(seed=cfg_sim["semilla"])
    while True:
        estado = info["_estado_obj"]
        libres = [b for b in estado.barcos if b.libre]
        decisiones = politica_fn(libres, estado, matriz_tiempos, env.capacidad_barco, cfg_sim["recompensa"])
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
        "politica": politica,
        "num_barcos": num_barcos,
        "semilla": semilla,
        "generadas": cons["generadas"],
        "atendidas": cons["atendidas"],
        "esperando_al_final": cons["esperando_al_final"],
        "a_bordo_al_final": cons["a_bordo_al_final"],
        "conservacion_cuadra": cons["cuadra"],
        "pct_atendidas": glob["pct_atendidas"],
        "backlog_final_total": int(backlog["sin_atender"].sum()) if len(backlog) else 0,
        "espera_media_min": glob["espera_media_min"],
        "espera_p50_min": usr["espera_min"]["p50"],
        "espera_p95_min": usr["espera_min"]["p95"],
        "espera_max_min": usr["espera_min"]["max"],
        "sistema_medio_min": glob["sistema_medio_min"],
        "sistema_p50_min": usr["sistema_min"]["p50"],
        "sistema_p95_min": usr["sistema_min"]["p95"],
        "sistema_max_min": usr["sistema_min"]["max"],
        "movimientos_totales": int(por_barco["movimientos"].sum()),
        "ocupacion_media_flota": float(por_barco["ocupacion_media"].mean()),
        "pct_esperando_flota": float(por_barco["pct_esperando"].mean()),
    }


def main() -> None:
    print(f"Nodos: {nodos}. Capacidad/barco: {capacidad}. Flotas: {FLOTAS}. Semillas eval: {SEMILLAS_EVAL}")
    print(f"Horizonte: {hora_ini_min}-{hora_fin_min} min, paso {cfg_sim['paso_tiempo_min']} min")

    grupos_por_semilla: dict[int, pd.DataFrame] = {}
    for semilla in SEMILLAS_EVAL:
        ruta = OUT_DIR / f"grupos_seed{semilla}.csv"
        if ruta.exists():
            # Reusa la demanda ya generada en la corrida anterior (misma
            # semilla, mismo cfg_demand -- exactamente reproducible) en vez
            # de regenerarla, para que H0/H1a/H1b/H1c/H2 se comparen sobre
            # la MISMA demanda exacta que ya se uso en el analisis previo.
            grupos = pd.read_csv(ruta)
        else:
            grupos = generar_grupos_dia(semilla)
            grupos.to_csv(ruta, index=False)
        grupos_por_semilla[semilla] = grupos
        print(f"  semilla {semilla}: {len(grupos)} grupos, {grupos['tamano_grupo'].sum()} personas -> {ruta.name}")

    filas = []
    t0 = time.time()
    total_corridas = len(POLITICAS) * len(FLOTAS) * len(SEMILLAS_EVAL)
    i = 0
    for semilla, grupos in grupos_por_semilla.items():
        for num_barcos in FLOTAS:
            for nombre_politica, politica_fn in POLITICAS.items():
                i += 1
                env = correr(grupos, num_barcos, politica_fn)
                filas.append(fila_metricas(env, nombre_politica, num_barcos, semilla))
                print(f"  [{i}/{total_corridas}] {nombre_politica} x {num_barcos} barcos x semilla {semilla} "
                      f"-- espera media {filas[-1]['espera_media_min']:.2f} min, "
                      f"{filas[-1]['pct_atendidas']:.1f}% atendidas "
                      f"({time.time()-t0:.0f}s acumulados)")

    resultados = pd.DataFrame(filas)
    resultados.to_csv(OUT_DIR / "resultados_fleet_sweep.csv", index=False)

    columnas_num = [c for c in resultados.columns if c not in ("politica", "num_barcos", "semilla", "conservacion_cuadra")]
    resumen = (
        resultados.groupby(["politica", "num_barcos"])[columnas_num]
        .agg(["mean", "std"])
    )
    resumen.columns = [f"{c}_{stat}" for c, stat in resumen.columns]
    resumen = resumen.reset_index()
    resumen.to_csv(OUT_DIR / "resumen_fleet_sweep.csv", index=False)

    assert resultados["conservacion_cuadra"].all(), "Conservacion NO cuadra en alguna corrida -- revisar antes de reportar nada"

    print(f"\nListo en {time.time()-t0:.0f}s. {len(resultados)} corridas -> {OUT_DIR}")
    print(resumen[["politica", "num_barcos", "espera_media_min_mean", "espera_p95_min_mean", "pct_atendidas_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
