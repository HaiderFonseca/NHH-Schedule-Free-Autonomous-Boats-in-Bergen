"""Llegadas de grupos de pasajeros minuto a minuto (Tarea 4).

De la intensidad O-D por franja (Tarea 3) se deriva, para cada hora del día,
una tasa esperada de llegada de GRUPOS de pasajeros (no de personas sueltas:
se agrupan desde esta primera iteración, siguiendo a Braathen et al.). Dentro
de cada hora, las llegadas son un proceso de Poisson homogéneo: se sortea
cuántos grupos llegan esa hora (Poisson) y se reparte cada uno en un minuto
al azar dentro de la hora.

La escala total de pasajeros/día se deriva como un PORCENTAJE de la
población real de las 4 zonas (Tarea 1 v2), no un número fijo inventado —
ver `config/instance.yaml` (`demanda.porcentaje_poblacion_dia`) y
`demand/README.md` para cómo se escogió ese porcentaje cruzando dos papers
de referencia contra la población.

El tamaño de cada grupo se muestrea de una Binomial(n, p) (no Normal: son
personas, una cantidad discreta), calibrada por método de momentos para
igualar la media/desviación reportada por Braathen et al. (2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expandir_intensidad_a_horas(intensidad_od: pd.DataFrame) -> pd.DataFrame:
    """Expande la intensidad por franja (Tarea 3) a una fila por hora del día
    (cada hora hereda la intensidad de su franja — la tasa es constante a
    trozos dentro de cada franja, no una curva suavizada hora a hora).
    """
    filas = []
    for _, r in intensidad_od.iterrows():
        for hora in range(int(r["hora_inicio"]), int(r["hora_fin"])):
            filas.append({**r.to_dict(), "hora": hora})
    return pd.DataFrame(filas)


def calcular_escala_pasajeros_dia(cfg: dict, poblacion_total_zonas: float) -> float:
    """Escala total de pasajeros/día = porcentaje configurado x población real
    de las 4 zonas de captación (Tarea 1 v2, `masas_por_nodo.csv`).
    """
    pct = cfg["demanda"]["porcentaje_poblacion_dia"]
    return poblacion_total_zonas * pct


def calcular_lambda_grupos(
    intensidad_horaria: pd.DataFrame,
    cfg: dict,
    poblacion_total_zonas: float,
    factor_dia_semana: float,
) -> pd.DataFrame:
    """Convierte intensidad (adimensional) en tasa esperada de GRUPOS por hora
    para cada (origen, destino, hora), a partir de la escala de pasajeros/día
    (derivada de la población real, ver `calcular_escala_pasajeros_dia`) y el
    tamaño medio de grupo (media de la Binomial: n*p).
    """
    cfg_tam = cfg["demanda"]["tamano_grupo"]
    media_grupo = cfg_tam["n"] * cfg_tam["p"]

    escala_pasajeros_dia = calcular_escala_pasajeros_dia(cfg, poblacion_total_zonas) * factor_dia_semana
    grupos_totales_dia = escala_pasajeros_dia / media_grupo

    df = intensidad_horaria.copy()
    total_intensidad = df["intensidad"].sum()
    df["lambda_grupos_hora"] = df["intensidad"] / total_intensidad * grupos_totales_dia
    return df


def _es_conexion_fuerte(o: str, d: str, conexiones_fuertes: list[list[str]]) -> bool:
    return any({o, d} == set(par) for par in conexiones_fuertes)


def generar_llegadas_dia(
    cfg: dict,
    intensidad_od: pd.DataFrame,
    conexiones_fuertes: list[list[str]],
    poblacion_total_zonas: float,
    es_fin_de_semana: bool,
    rng: np.random.Generator,
    dia_id,
) -> pd.DataFrame:
    """Genera la tabla de grupos de pasajeros de un día completo.

    `dia_id` es solo una etiqueta (índice 0-6, fecha, o nombre de día) que se
    guarda en la columna `dia` de salida, para poder concatenar varios días
    (Tarea 4 - semana) sin perder de cuál día viene cada grupo.
    """
    factor_dia = (
        cfg["factor_dia_semana"]["fin_de_semana"] if es_fin_de_semana
        else cfg["factor_dia_semana"]["entre_semana"]
    )

    intensidad_horaria = expandir_intensidad_a_horas(intensidad_od)
    con_lambda = calcular_lambda_grupos(intensidad_horaria, cfg, poblacion_total_zonas, factor_dia)

    cfg_tam = cfg["demanda"]["tamano_grupo"]
    cfg_esp = cfg["demanda"]["espera_maxima_min"]

    grupos = []
    contador = 0
    for _, r in con_lambda.iterrows():
        n_grupos = rng.poisson(r["lambda_grupos_hora"])
        if n_grupos == 0:
            continue

        minutos = rng.uniform(0, 60, size=n_grupos)
        tamanos = rng.binomial(cfg_tam["n"], cfg_tam["p"], size=n_grupos)
        tamanos = np.maximum(tamanos, cfg_tam["minimo"])

        es_fuerte = _es_conexion_fuerte(r["origen"], r["destino"], conexiones_fuertes)
        base_espera = cfg_esp["conexiones_fuertes_min"] if es_fuerte else cfg_esp["resto_min"]
        jitter = rng.uniform(-cfg_esp["jitter_min"], cfg_esp["jitter_min"], size=n_grupos)
        esperas = np.maximum(base_espera + jitter, 1.0)

        for i in range(n_grupos):
            contador += 1
            minuto_dia = int(r["hora"]) * 60 + minutos[i]
            grupos.append({
                "grupo_id": f"{dia_id}_{contador:05d}",
                "dia": dia_id,
                "franja": r["franja"],
                "hora": int(r["hora"]),
                "minuto_dia": round(minuto_dia, 2),
                "origen": r["origen"],
                "destino": r["destino"],
                "tamano_grupo": int(tamanos[i]),
                "espera_maxima_min": round(float(esperas[i]), 1),
                "conexion_fuerte": es_fuerte,
                "fin_de_semana": es_fin_de_semana,
            })

    df = pd.DataFrame(grupos)
    if len(df):
        df = df.sort_values("minuto_dia").reset_index(drop=True)
    return df


def generar_llegadas_semana(
    cfg: dict,
    intensidad_od: pd.DataFrame,
    conexiones_fuertes: list[list[str]],
    poblacion_total_zonas: float,
    semilla: int,
) -> pd.DataFrame:
    """Genera 7 días (lunes=0 .. domingo=6) con el mismo patrón de fondo pero
    llegadas concretas distintas y reproducibles: una semilla hija por día,
    derivada de `semilla` con `numpy.random.SeedSequence` (mismo mecanismo
    recomendado por NumPy para streams independientes reproducibles).
    """
    ss = np.random.SeedSequence(semilla)
    semillas_hijas = ss.spawn(7)

    dias = []
    for dia_idx in range(7):
        rng = np.random.default_rng(semillas_hijas[dia_idx])
        es_finde = dia_idx >= 5  # 5=sabado, 6=domingo
        df_dia = generar_llegadas_dia(
            cfg, intensidad_od, conexiones_fuertes, poblacion_total_zonas, es_finde, rng, dia_id=dia_idx,
        )
        dias.append(df_dia)

    return pd.concat(dias, ignore_index=True)
