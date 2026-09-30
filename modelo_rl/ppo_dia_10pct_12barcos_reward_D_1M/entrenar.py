"""Entrenamiento FINAL de PPO -- dia completo (06:00-24:00), 12 barcos,
capacidad 30, demanda 10%, recompensa D, n_steps=540, total_timesteps=1_000_000,
**4 entornos en paralelo** (ver "CORRIDA 2" abajo).

Reusa la infraestructura EXISTENTE sin modificarla:
- `modelo_rl/src/entrenamiento.py` (EntornoDemandaAleatoria) -- regenera la
  demanda en cada reset(), sin tocar `simulador/src/env.py`.
- Los hiperparametros compartidos (gamma, ent_coef, learning_rate,
  usar_vecnormalize) y la formula de recompensa D, leidos TAL CUAL de
  `simulador/config/instance.yaml` -> `agente` (no se modifica ese archivo).
- El patron de entrenamiento (Monitor + VecEnv + VecNormalize + PPO
  MlpPolicy) de `modelo_rl/notebooks/01_enfoque_y_entrenamiento_final.ipynb`
  (unica diferencia: `SubprocVecEnv` con 4 copias en vez de `DummyVecEnv`
  con 1 -- ver mas abajo).

Lo unico que NO viene de `simulador/config/instance.yaml` (porque ese
archivo solo tiene escenarios de 2 barcos / 90 pasos, el "escenario antiguo"
que este entrenamiento tiene explicitamente prohibido usar) es el escenario
en si -- 12 barcos, dia completo, 10% -- y los overrides de n_steps/
total_timesteps pedidos para esta corrida. Se definen como constantes
explicitas mas abajo, documentadas una por una.

**HISTORIAL DE ESTA CORRIDA (todo documentado, nada se esconde):**

*Corrida 1 (300,000 pasos, 1 entorno serial)* -- resultados archivados en
`corrida1_300k_serial_referencia/`. Motivo del recorte original: medido en
esta maquina, 1,000,000 pasos en 1 entorno tomaba ~48.5h (5.7 pasos/seg) por
contencion de OneDrive (el proyecto vive en una carpeta sincronizada, cada
escritura del log competia con el servicio de sincronizacion). Sacando el
log de OneDrive durante el entrenamiento subio a ~22.8 pasos/seg (~12-13h
para 1,000,000) -- seguia siendo demasiado, el usuario autorizo bajar a
300,000 (~3.7-4h) como primera corrida. Esa corrida SI se completo y evaluo
(ver la carpeta de referencia) -- resultado: PPO NO superaba a H0 ni a H3 en
ninguna metrica de servicio (espera media 17.96 min vs 6.18/4.81 de H0/H3).

*Corrida 2 (esta, 1,000,000 pasos, 4 entornos en paralelo)* -- diagnostico:
con solo 300,000 pasos (556 episodios) el agente no tuvo presupuesto
suficiente para un problema de este tamano (12 barcos, ~8000 personas/dia,
vs. el escenario historico de 2 barcos/~200 personas para el que se habian
validado los hiperparametros). La palanca de mayor impacto es MAS
entrenamiento, no otro hiperparametro -- pero a 1 entorno, 1,000,000 pasos
tardaba ~12-13h. Se probo paralelizar con `SubprocVecEnv` (4 copias del
entorno, cada una en su propio proceso/nucleo): 53.5 pasos/seg medidos
(vs 22.8 con 1 solo) -> ~5.2h para 1,000,000 pasos. El usuario aprobo
explicitamente esta corrida (n_envs=4, total_timesteps=1,000,000, el valor
ORIGINAL, ya no reducido).

**Matiz importante, que el usuario acepto explicitamente:** con `n_envs=4`,
cada actualizacion de PPO junta `n_steps * n_envs = 540*4 = 2160` pasos --
es decir, 4 jornadas completas simuladas EN PARALELO por actualizacion, no
1 sola como en la Corrida 1 (donde "1 rollout = 1 dia" era exacto).
`n_steps=540` se mantiene igual (por entorno) -- lo que cambia es que el
tamano efectivo del lote de actualizacion es 4x. Ningun otro hiperparametro
(gamma, learning_rate, ent_coef, reward D, semillas) se toco.

**Carpeta final vs. carpeta de trabajo durante el entrenamiento:** el
resultado (modelo, VecNormalize, metadata, log de monitor) se guarda en ESTA
carpeta (dentro de OneDrive) al terminar -- una sola escritura final, no
miles repetidas. Mientras entrena, el log de progreso (que SI se reescribe
en cada episodio, por CADA uno de los 4 entornos) se escribe en una carpeta
local temporal (`LOCAL_SCRATCH_DIR`, fuera de OneDrive) para no competir con
la sincronizacion, y se copia a esta carpeta recien al final.

NO se modifica ningun archivo fuera de esta carpeta (y la carpeta de scratch
local, que no forma parte del repositorio).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPERIMENTO_DIR = Path(__file__).resolve().parent
BASE_DIR = EXPERIMENTO_DIR.parent.parent
SIMULADOR_DIR = BASE_DIR / "simulador"
BB_DIR = BASE_DIR / "bergen-boats"
DEMAND_DIR = BASE_DIR / "demand"
MODELO_RL_DIR = BASE_DIR / "modelo_rl"

sys.path.insert(0, str(DEMAND_DIR / "src"))
sys.path.insert(0, str(SIMULADOR_DIR / "src"))
sys.path.insert(0, str(MODELO_RL_DIR / "src"))

import masas as demand_masas                      # noqa: E402
import llegadas as demand_llegadas                 # noqa: E402
from entrenamiento import EntornoDemandaAleatoria   # noqa: E402

# ---------------------------------------------------------------------------
# ESCENARIO Y OVERRIDES DE ESTA CORRIDA -- unicos valores que NO vienen leidos
# de simulador/config/instance.yaml (ese archivo no se toca). Cada uno
# corresponde 1:1 a una instruccion explicita del pedido de este experimento.
# ---------------------------------------------------------------------------
NUM_BARCOS = 12
HORAS = (6, 24)                     # 06:00 - 24:00
PORCENTAJE_POBLACION_DIA = 0.10     # 10% -- coincide con el default oficial de demand/config/instance.yaml
N_STEPS = 540                       # 18h * 60 / 2min = 540 pasos = 1 episodio completo por rollout, POR ENTORNO
N_ENVS = 2                          # entornos en paralelo (SubprocVecEnv) -- ver docstring, "Corrida 3"
TOTAL_TIMESTEPS_ORIGINAL = 1_000_000   # el pedido original
TOTAL_TIMESTEPS = 700_000              # Corrida 3 -- ver MOTIVO_CAMBIO_TIMESTEPS
MOTIVO_CAMBIO_TIMESTEPS = (
    "Corrida 1 (300,000, 1 entorno) completada y evaluada -- PPO no supero a H0 "
    "ni H3 en ninguna metrica de servicio (ver corrida1_300k_serial_referencia/). "
    "Corrida 2 (1,000,000, 4 entornos en paralelo) se INTERRUMPIO a los ~92 min: "
    "la RAM libre de la maquina (15.8GB totales) cayo de 3.7GB a 1.4GB con 4 "
    "procesos en paralelo, y el fps colapso de 69 a 7 (proyectaba ~38h). Se "
    "mato el proceso antes de que degradara mas. Corrida 3 (esta): 2 entornos "
    "en vez de 4 (menos presion de memoria), validado con una prueba sostenida "
    "de 20 rollouts (~11 min) antes de lanzar la corrida larga -- la memoria SI "
    "se recupero completa despues de esa prueba (no hubo fuga persistente), "
    "aunque el fps igual bajo de forma mas suave (67->31) sin causa clara "
    "identificada. total_timesteps=700,000 (no el 1,000,000 original) como "
    "punto medio realista dado el fps medido (~31 pasos/seg sostenidos, "
    "~6.2h) -- aprobado explicitamente por el usuario (Opcion A). Ningun otro "
    "hiperparametro (gamma, learning_rate, ent_coef, n_steps POR ENTORNO, "
    "reward D, semillas) se modifico."
)
TRAINING_SEED = 123
EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]

OUT_DIR = EXPERIMENTO_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta LOCAL (fuera de OneDrive) solo para el log de progreso durante el
# entrenamiento (se reescribe en cada episodio -- cientos de veces). El
# resultado final se copia a OUT_DIR (OneDrive) una sola vez, al terminar.
LOCAL_SCRATCH_DIR = Path(r"C:\tmp_ppo_training") / EXPERIMENTO_DIR.name
LOCAL_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Todo lo demas se LEE, sin modificar, de los config ya existentes.
# ---------------------------------------------------------------------------
cfg_sim = yaml.safe_load(open(SIMULADOR_DIR / "config" / "instance.yaml", encoding="utf-8"))
cfg_bb = yaml.safe_load(open(BB_DIR / "config" / "instance.yaml", encoding="utf-8"))
cfg_demand = demand_masas.cargar_config(DEMAND_DIR / "config" / "instance.yaml")

resumen_masas = pd.read_csv(DEMAND_DIR / "output" / "masas_por_nodo.csv", index_col="id")
intensidad_od = pd.read_csv(DEMAND_DIR / "output" / "matriz_intensidad_od.csv")
poblacion_total_zonas = resumen_masas["poblacion_total"].sum()
conexiones_fuertes = cfg_bb["garantia"]["conexiones_fuertes"]

nodos = [n["id"] for n in cfg_bb["nodos_demanda"]]
matriz_tiempos = pd.read_csv(BB_DIR / "02_ruteo_navegable" / "output" / "matriz_tiempos_min.csv", index_col=0)

cfg_agente = cfg_sim["agente"]
GAMMA = cfg_agente["gamma"]
cfg_hiper = cfg_agente["hiperparametros"]
ENT_COEF = cfg_hiper["ent_coef"]
LEARNING_RATE = cfg_hiper["learning_rate"]
USAR_VECNORMALIZE = cfg_hiper["usar_vecnormalize"]

CAPACIDAD = cfg_bb["flota"]["capacidad_pasajeros"]
NODO_INICIAL = cfg_bb["flota"]["nodo_inicial"]
PASO_TIEMPO_MIN = cfg_sim["paso_tiempo_min"]
UNIDAD_DEMANDA = cfg_sim["unidad_demanda"]

# Recompensa D -- MISMA fuente que usa el proyecto para "D" (no se redefine
# aqui, se lee tal cual): simulador/config/instance.yaml -> agente.entrenamiento.recompensa_overrides.
REWARD_D_OVERRIDES = cfg_agente["entrenamiento"]["recompensa_overrides"]
cfg_recompensa = dict(cfg_sim["recompensa"])
cfg_recompensa.update(REWARD_D_OVERRIDES)

HORA_INI_MIN, HORA_FIN_MIN = HORAS[0] * 60, HORAS[1] * 60


def construir_entorno(semilla_entrenamiento):
    return EntornoDemandaAleatoria(
        generar_llegadas_dia_fn=demand_llegadas.generar_llegadas_dia,
        cfg_demand=cfg_demand, intensidad_od=intensidad_od, conexiones_fuertes=conexiones_fuertes,
        poblacion_total_zonas=poblacion_total_zonas, horas=HORAS,
        porcentaje_poblacion_dia=PORCENTAJE_POBLACION_DIA,
        semilla_entrenamiento=semilla_entrenamiento,
        matriz_tiempos=matriz_tiempos, nodos=nodos, num_barcos=NUM_BARCOS,
        capacidad_barco=CAPACIDAD, nodo_inicial=NODO_INICIAL,
        paso_tiempo_min=PASO_TIEMPO_MIN, hora_inicio_min=HORA_INI_MIN, hora_fin_min=HORA_FIN_MIN,
        cfg_recompensa=cfg_recompensa, unidad_demanda=UNIDAD_DEMANDA,
    )


def sanity_check() -> None:
    print("=" * 78)
    print("SANITY CHECK -- antes de entrenar")
    print("=" * 78)
    errores = []

    env = construir_entorno(semilla_entrenamiento=TRAINING_SEED)

    # 1) Escenario
    print(f"num_barcos = {env.num_barcos} (esperado 12)")
    if env.num_barcos != 12:
        errores.append("num_barcos != 12")
    print(f"capacidad_barco = {env.capacidad_barco} (esperado 30)")
    if env.capacidad_barco != 30:
        errores.append("capacidad_barco != 30")
    print(f"porcentaje_poblacion_dia usado = {PORCENTAJE_POBLACION_DIA} (esperado 0.10)")
    if PORCENTAJE_POBLACION_DIA != 0.10:
        errores.append("porcentaje_poblacion_dia != 0.10")
    print(f"horario = {HORAS} -> [{env.hora_inicio_min},{env.hora_fin_min}] min (esperado [360,1440])")
    if (env.hora_inicio_min, env.hora_fin_min) != (360.0, 1440.0):
        errores.append("horario != [360,1440]")
    print(f"paso_tiempo_min = {env.paso_tiempo_min} (esperado 2)")
    if env.paso_tiempo_min != 2:
        errores.append("paso_tiempo_min != 2")

    # 2) Episodio completo = 540 pasos exactos (politica dummy: siempre "esperar")
    obs, info = env.reset(seed=EVAL_SEEDS[0])
    n_pasos = 0
    accion_esperar = np.array([env.codificar_accion_barco(None)] * env.num_barcos)
    while True:
        obs, r, terminated, truncated, info = env.step(accion_esperar)
        n_pasos += 1
        if terminated or truncated:
            break
    print(f"pasos en un episodio completo = {n_pasos} (esperado 540)")
    if n_pasos != 540:
        errores.append(f"episodio tiene {n_pasos} pasos, no 540")

    # 3) Reward = D
    print(f"recompensa_overrides usados = {REWARD_D_OVERRIDES}")
    d_esperado = {
        "tolerancia_incomodidad_min": 0, "sobrante_normalizador_min": 12,
        "penalizacion_maxima_persona": 100000000000.0, "peso_movimiento": 0,
        "premio_por_persona_entregada": 0,
    }
    if REWARD_D_OVERRIDES != d_esperado:
        errores.append(f"recompensa_overrides no coincide con D esperado: {d_esperado}")
    else:
        print("-> coincide exactamente con la formulacion D (tiempo cuadratico puro, sin tolerancia).")

    # 4) VecNormalize activado
    print(f"usar_vecnormalize = {USAR_VECNORMALIZE} (esperado True)")
    if not USAR_VECNORMALIZE:
        errores.append("usar_vecnormalize no esta activado en la config")

    # 5) n_steps
    print(f"n_steps = {N_STEPS} (esperado 540, NO 90, NO 512, NO 2048)")
    if N_STEPS != 540:
        errores.append("n_steps != 540")

    # 6) total_timesteps -- 700,000, Corrida 3 (ver MOTIVO_CAMBIO_TIMESTEPS).
    print(f"total_timesteps = {TOTAL_TIMESTEPS} (pedido original: {TOTAL_TIMESTEPS_ORIGINAL} -- "
          f"ver MOTIVO_CAMBIO_TIMESTEPS para el historial completo de corridas)")
    if TOTAL_TIMESTEPS != 700_000:
        errores.append("total_timesteps != 700,000 (el valor acordado para esta corrida)")
    print(f"n_envs = {N_ENVS} (paralelizacion -- n_steps se mantiene en 540 POR ENTORNO, ver docstring)")

    # 7) La demanda cambia entre episodios (mismo chequeo que notebook 01)
    env.reset(seed=42)
    n_a = len(env.grupos_df)
    env.reset(seed=42)
    n_b = len(env.grupos_df)
    env.reset()
    n_c = len(env.grupos_df)
    env.reset()
    n_d = len(env.grupos_df)
    print(f"reset(seed=42) dos veces: {n_a} y {n_b} grupos -- iguales: {n_a == n_b} (esperado True)")
    print(f"reset() sin semilla dos veces: {n_c} y {n_d} grupos -- distintos: {n_c != n_d} (esperado True)")
    if n_a != n_b:
        errores.append("reset(seed=X) no es reproducible")
    if n_c == n_d:
        errores.append("reset() sin semilla no genera demanda distinta")

    # 8) No usar por accidente el modelo historico del escenario viejo
    modelo_viejo = MODELO_RL_DIR / "output" / "modelo_final" / "modelo_ppo.zip"
    print(f"Carpeta de salida de este experimento: {OUT_DIR}")
    print(f"Modelo historico (NO se toca ni se reusa): {modelo_viejo} (existe: {modelo_viejo.exists()})")
    if OUT_DIR == modelo_viejo.parent:
        errores.append("OUT_DIR coincide con la carpeta del modelo historico -- NO usar esa carpeta")
    modelo_nuevo_path = OUT_DIR / "modelo_ppo.zip"
    if modelo_nuevo_path.exists():
        errores.append(f"Ya existe un modelo en {modelo_nuevo_path} -- no se sobrescribe, usar otra carpeta")

    # check_env estandar de SB3
    from stable_baselines3.common.env_checker import check_env
    check_env(construir_entorno(semilla_entrenamiento=TRAINING_SEED), warn=True)
    print("check_env: sin problemas.")

    print("\n" + "=" * 78)
    if errores:
        print("SANITY CHECK FALLIDO -- se detiene ANTES de entrenar. Errores:")
        for e in errores:
            print(f"  - {e}")
        print("=" * 78)
        raise SystemExit(1)
    print("SANITY CHECK OK -- todas las condiciones se cumplen. Se procede a entrenar.")
    print("=" * 78)


def _make_env_fn(semilla_entrenamiento: int, monitor_filename: str):
    """Fabrica de entornos para SubprocVecEnv -- cada entorno paralelo con su
    PROPIA semilla de entrenamiento (independiente, via SeedSequence.spawn,
    mismo mecanismo que ya usa el proyecto para dias/semillas independientes
    -- ver `demand/src/llegadas.py`, `generar_llegadas_semana`) y su propio
    archivo de log (si comparten nombre, Monitor los pisa entre si).
    """
    from stable_baselines3.common.monitor import Monitor

    def _f():
        return Monitor(construir_entorno(semilla_entrenamiento=semilla_entrenamiento), filename=monitor_filename)
    return _f


def entrenar() -> dict:
    import shutil

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import load_results
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # El log de Monitor se reescribe en CADA episodio, por CADA uno de los
    # N_ENVS entornos -- se escribe en LOCAL_SCRATCH_DIR (fuera de OneDrive)
    # mientras se entrena, para no competir con la sincronizacion (ver
    # docstring del modulo). Se agrega y copia a OUT_DIR (OneDrive) al final,
    # una sola vez.
    print(f"Log de progreso durante el entrenamiento (temporal, fuera de OneDrive): {LOCAL_SCRATCH_DIR}")
    print(f"Paralelizando con {N_ENVS} entornos (SubprocVecEnv) -- cada uno con semilla de entrenamiento independiente")

    # Semillas independientes por entorno, derivadas de TRAINING_SEED -- mismo
    # mecanismo (SeedSequence.spawn) que ya usa demand/src/llegadas.py para
    # derivar dias/semillas independientes de una semilla raiz.
    semillas_envs = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(TRAINING_SEED).spawn(N_ENVS)]
    print(f"Semillas de entrenamiento por entorno (derivadas de {TRAINING_SEED}): {semillas_envs}")

    env_fns = [
        _make_env_fn(semillas_envs[i], str(LOCAL_SCRATCH_DIR / f"monitor_{i}"))
        for i in range(N_ENVS)
    ]
    venv = SubprocVecEnv(env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=GAMMA)

    model = PPO(
        "MlpPolicy", venv,
        gamma=GAMMA, ent_coef=ENT_COEF, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
        seed=TRAINING_SEED, verbose=1,
    )
    print(f"\nPPO creado: gamma={model.gamma}, ent_coef={model.ent_coef}, "
          f"learning_rate={model.learning_rate}, n_steps={model.n_steps} (x {N_ENVS} entornos = "
          f"{model.n_steps * N_ENVS} pasos por actualizacion), seed={TRAINING_SEED}")

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    tiempo_s = time.time() - t0
    print(f"\nEntrenamiento terminado en {tiempo_s:.1f}s ({tiempo_s/60:.1f} min)")
    print(f"Timesteps reales entrenados (model.num_timesteps): {model.num_timesteps}")

    # Guardado FINAL -- unica escritura repetida al directorio de OneDrive
    # (el modelo y el VecNormalize se guardan una sola vez, no cientos de veces).
    model.save(str(OUT_DIR / "modelo_ppo"))
    venv.save(str(OUT_DIR / "vecnormalize.pkl"))
    venv.close()

    # Agrega los N_ENVS logs de monitor en uno solo (load_results junta todos
    # los *.monitor.csv de la carpeta, ordenados por tiempo de inicio de episodio).
    df_monitor = load_results(str(LOCAL_SCRATCH_DIR))
    df_monitor.to_csv(OUT_DIR / "monitor.monitor.csv", index=False)
    for i in range(N_ENVS):
        origen = LOCAL_SCRATCH_DIR / f"monitor_{i}.monitor.csv"
        if origen.exists():
            shutil.copy(origen, OUT_DIR / f"monitor_{i}.monitor.csv")
    print(f"Modelo, VecNormalize y logs de monitor ({N_ENVS} entornos, agregados) copiados a {OUT_DIR}")

    actualizaciones_ppo = model.num_timesteps // (N_STEPS * N_ENVS)

    metadata = {
        "demand_level": PORCENTAJE_POBLACION_DIA,
        "hours": list(HORAS),
        "num_boats": NUM_BARCOS,
        "capacity": CAPACIDAD,
        "step_minutes": PASO_TIEMPO_MIN,
        "episode_steps": 540,
        "reward": "D",
        "reward_overrides_used": REWARD_D_OVERRIDES,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "ent_coef": ENT_COEF,
        "n_steps": N_STEPS,
        "n_envs": N_ENVS,
        "pasos_por_actualizacion": N_STEPS * N_ENVS,
        "vecnormalize": USAR_VECNORMALIZE,
        "total_timesteps_original_request": TOTAL_TIMESTEPS_ORIGINAL,
        "total_timesteps_usado": TOTAL_TIMESTEPS,
        "motivo_cambio_total_timesteps": MOTIVO_CAMBIO_TIMESTEPS,
        "total_timesteps_requested": TOTAL_TIMESTEPS,
        "total_timesteps_real": int(model.num_timesteps),
        "actualizaciones_ppo": int(actualizaciones_ppo),
        "tiempo_entrenamiento_s": tiempo_s,
        "tiempo_entrenamiento_min": tiempo_s / 60,
        "episodios_completados": len(df_monitor),
        "pasos_por_segundo": model.num_timesteps / tiempo_s,
        "training_seed": TRAINING_SEED,
        "evaluation_seeds": EVAL_SEEDS,
        "nodo_inicial": NODO_INICIAL,
        "unidad_demanda": UNIDAD_DEMANDA,
    }
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("\nmetadata.json escrito:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    sanity_check()
    entrenar()
