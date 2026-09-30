"""Evaluacion del PPO entrenado (`entrenar.py`) sobre las 5 semillas de
evaluacion, y comparacion final H0 vs H3 vs PPO -- MISMO escenario (12
barcos, capacidad 30, demanda 10%, dia completo) para los tres.

No se re-entrena nada aqui. H0/H3 NO se recalculan -- se reusan tal cual
de `heuristicas/outputs/resultados/resultados_h0_h1_h2_h3.csv` (ya
generado, mismas semillas 1001-1005, mismos 12 barcos, misma demanda).

Patron de evaluacion con VecNormalize -- reusado TAL CUAL de
`comparacion/notebooks/05_comparacion_agente_vs_base.ipynb` (ya depurado
ahi: stepear el entorno CRUDO directamente, NO a traves de
`DummyVecEnv.step()`, porque ese auto-resetea en el mismo step() en que
`done=True` y borra `atendidas_historico`/`log_eventos` antes de poder
leerlos -- ver el comentario original en esa celda).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTO_DIR))
import entrenar as E                              # noqa: E402 -- reusa constantes/construir_entorno, no re-entrena

sys.path.insert(0, str(E.SIMULADOR_DIR / "src"))
sys.path.insert(0, str((E.BASE_DIR / "politica_base" / "src")))
import metricas as met                             # noqa: E402

from stable_baselines3 import PPO                  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

MODELO_PATH = EXPERIMENTO_DIR / "modelo_ppo.zip"
VECNORMALIZE_PATH = EXPERIMENTO_DIR / "vecnormalize.pkl"
HEURISTICAS_RESULTADOS = E.BASE_DIR / "heuristicas" / "outputs" / "resultados" / "resultados_h0_h1_h2_h3.csv"

assert MODELO_PATH.exists(), f"No existe el modelo entrenado en {MODELO_PATH}"
assert VECNORMALIZE_PATH.exists(), f"No existe VecNormalize en {VECNORMALIZE_PATH}"
assert HEURISTICAS_RESULTADOS.exists(), f"No existen los resultados de H0/H3 en {HEURISTICAS_RESULTADOS}"


def construir_entorno_eval():
    return E.construir_entorno(semilla_entrenamiento=E.TRAINING_SEED)  # irrelevante: siempre se llama reset(seed=X) explicito


def correr_ppo(model, venv_stats, semilla):
    env = construir_entorno_eval()
    obs, info = env.reset(seed=semilla)
    reward_total = 0.0
    while True:
        obs_norm = venv_stats.normalize_obs(obs)
        accion, _ = model.predict(obs_norm, deterministic=True)
        obs, r, terminated, truncated, info = env.step(accion)
        reward_total += r
        if truncated or terminated:
            break
    return env, reward_total


def fila_metricas_ppo(env, semilla) -> dict:
    """Mismas definiciones EXACTAS que `heuristicas/experimento_fleet_sweep.py`
    (metricas.py sin modificar) -- para que la comparacion con H0/H3 sea
    directa, columna por columna. Se agregan sistema_p95_min/sistema_max_min
    (que la evaluacion existente de H0/H3 no guardo) porque para PPO SI se
    pueden calcular aqui, sin recalcular nada de H0/H3.
    """
    cons = met.verificar_conservacion(env)
    glob = met.metricas_globales(env)
    usr = met.metricas_por_usuario(env)
    por_barco = met.metricas_por_barco(env)
    backlog = met.sin_atender_al_final_por_par(env)
    movs = [ev for ev in env.log_eventos if ev["tipo"] == "movimiento"]
    movimientos_vacios = sum(1 for m in movs if m["ocupacion"] == 0)
    viajes = list(met._tiempos_viaje_por_unidad(env).values())
    return {
        "politica": "PPO", "num_barcos": env.num_barcos, "semilla": semilla,
        "conservacion_cuadra": cons["cuadra"], "pct_atendidas": glob["pct_atendidas"],
        "backlog_final_total": int(backlog["sin_atender"].sum()) if len(backlog) else 0,
        "espera_media_min": glob["espera_media_min"], "espera_p95_min": usr["espera_min"]["p95"],
        "espera_max_min": usr["espera_min"]["max"],
        "viaje_medio_min": float(np.mean(viajes)) if viajes else float("nan"),
        "sistema_medio_min": glob["sistema_medio_min"],
        "sistema_p95_min": usr["sistema_min"]["p95"], "sistema_max_min": usr["sistema_min"]["max"],
        "movimientos_totales": int(por_barco["movimientos"].sum()),
        "movimientos_vacios": movimientos_vacios, "movimientos_con_carga": len(movs) - movimientos_vacios,
        "ocupacion_pct_capacidad": float(por_barco["ocupacion_media"].mean()) / env.capacidad_barco * 100.0,
        "pct_esperando_flota": float(por_barco["pct_esperando"].mean()),
    }


def main():
    print("Cargando modelo entrenado y VecNormalize...")
    model = PPO.load(str(MODELO_PATH))
    venv_stats = VecNormalize.load(str(VECNORMALIZE_PATH), DummyVecEnv([construir_entorno_eval]))
    venv_stats.training = False
    venv_stats.norm_reward = False  # comparar sobre metricas operativas/reward crudo, no normalizado

    print(f"Evaluando PPO en semillas {E.EVAL_SEEDS} (12 barcos, capacidad 30, demanda 10%, 06:00-24:00)")
    filas = []
    for semilla in E.EVAL_SEEDS:
        env, reward_total = correr_ppo(model, venv_stats, semilla)
        cons = met.verificar_conservacion(env)
        assert cons["cuadra"], f"Conservacion NO cuadra en semilla {semilla}: {cons}"
        fila = fila_metricas_ppo(env, semilla)
        fila["reward_total_crudo"] = reward_total
        filas.append(fila)
        print(f"  semilla {semilla}: pct_atendidas={fila['pct_atendidas']:.2f}%  "
              f"espera_media={fila['espera_media_min']:.2f}min  conservacion_ok={cons['cuadra']}")

    resultados_ppo = pd.DataFrame(filas)
    resultados_ppo.to_csv(EXPERIMENTO_DIR / "resultados_evaluacion_ppo.csv", index=False)

    cols_num = [c for c in resultados_ppo.columns if c not in ("politica", "num_barcos", "semilla", "conservacion_cuadra")]
    resumen_ppo = resultados_ppo[cols_num].agg(["mean", "std"]).T
    resumen_ppo.columns = ["mean", "std"]
    resumen_ppo.to_csv(EXPERIMENTO_DIR / "resumen_evaluacion_ppo.csv")
    print("\nResumen PPO (media +/- std sobre 5 semillas):")
    print(resumen_ppo.round(3).to_string())

    # --- Comparacion final H0 vs H3 vs PPO -- H0/H3 REUSADOS, no recalculados ---
    heur = pd.read_csv(HEURISTICAS_RESULTADOS)
    heur_12 = heur[(heur["politica"].isin(["H0", "H3"])) & (heur["num_barcos"] == 12)]

    metricas_comparar = [
        "pct_atendidas", "backlog_final_total", "espera_media_min", "espera_p95_min", "espera_max_min",
        "viaje_medio_min", "sistema_medio_min", "movimientos_totales", "movimientos_vacios",
        "movimientos_con_carga", "ocupacion_pct_capacidad", "pct_esperando_flota",
    ]
    filas_comp = []
    for pol, df_pol in [("H0", heur_12[heur_12.politica == "H0"]), ("H3", heur_12[heur_12.politica == "H3"]),
                        ("PPO", resultados_ppo)]:
        fila = {"politica": pol}
        for m in metricas_comparar:
            fila[f"{m}_mean"] = df_pol[m].mean()
            fila[f"{m}_std"] = df_pol[m].std()
        filas_comp.append(fila)
    tabla_comparativa = pd.DataFrame(filas_comp).set_index("politica")
    tabla_comparativa.to_csv(EXPERIMENTO_DIR / "tabla_comparativa_h0_h3_ppo.csv")

    # sistema_p95/max: disponibles para PPO, NO para H0/H3 (esa corrida no los guardo) -- se reporta aparte, sin inventar nada para H0/H3.
    print("\n(sistema_p95_min / sistema_max_min de PPO -- no disponibles para H0/H3 en los resultados existentes, no se recalculan):")
    print(resumen_ppo.loc[["sistema_p95_min", "sistema_max_min"]].round(3).to_string())

    print("\n" + "=" * 100)
    print("TABLA COMPARATIVA FINAL -- H0 vs H3 vs PPO (12 barcos, capacidad 30, demanda 10%, semillas 1001-1005)")
    print("=" * 100)
    cols_mostrar = [c for c in tabla_comparativa.columns if c.endswith("_mean")]
    print(tabla_comparativa[cols_mostrar].round(2).to_string())

    with open(EXPERIMENTO_DIR / "resultado_evaluacion_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_seeds": E.EVAL_SEEDS,
            "num_boats": 12, "capacity": 30, "demand_level": 0.10, "hours": [6, 24],
            "modelo": str(MODELO_PATH), "vecnormalize": str(VECNORMALIZE_PATH),
            "h0_h3_source": str(HEURISTICAS_RESULTADOS),
            "nota": "H0 y H3 reusados de heuristicas/, NO recalculados. sistema_p95_min/sistema_max_min "
                    "no estan disponibles para H0/H3 en esa fuente.",
        }, f, indent=2, ensure_ascii=False)

    print(f"\nArchivos escritos en {EXPERIMENTO_DIR}:")
    print("  resultados_evaluacion_ppo.csv, resumen_evaluacion_ppo.csv, tabla_comparativa_h0_h3_ppo.csv, resultado_evaluacion_meta.json")


if __name__ == "__main__":
    main()
