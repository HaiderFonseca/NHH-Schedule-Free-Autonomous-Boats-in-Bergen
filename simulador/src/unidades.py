"""Unidades de demanda: personas o grupos, bajo una sola representación.

La especificación original (docs/especificacion_simulador_rl.md §4-§6) habla
de "grupos indivisibles". El ajuste posterior pide manejar personas
individuales por defecto. En vez de dos rutas de código separadas, se define
una única clase `Unidad` con un campo `tamano`: con `unidad_demanda="personas"`
cada persona es una `Unidad` con `tamano=1`; con `"grupos"`, cada `Unidad` es
un grupo completo con `tamano=tamano_grupo`. La regla de embarque en
`estado.py` es una sola función que resta `tamano` en orden FIFO -- con
`tamano=1` esto colapsa exactamente a "sube de a uno hasta llenar capacidad",
sin ninguna rama especial para el caso "personas".

**Sin paciencia (cambio de diseño):** la especificación original tenía una
"paciencia" (15/30 min +- jitter) que hacía perder a alguien si esperaba de
más -- se quitó de todo el simulador (ver `env.py`, `recompensa.py`,
`politica_base.py`). Nadie se va nunca: espera hasta ser atendido, o hasta
que termina la corrida. Por eso `Unidad` ya no tiene un campo de paciencia
-- no se usa en ningún lado, no tenía sentido seguir cargándolo. El motivo:
con paciencia, a alguien que esperaba de más se le "borraba" antes de saber
cuánto hubiera esperado en realidad -- eso escondía justo el dato que hace
falta para definir una garantía de tiempo con fundamento (ver
`metricas.metricas_por_usuario`, percentiles de espera real, sin censurar).

El generador de demanda real (`demand/src/llegadas.py`, calibrado con
gravedad + SSB + Poisson por franja) no se toca: aquí solo se "explota" su
salida. La columna `espera_maxima_min` que trae esa tabla (la paciencia
original) simplemente no se lee -- sigue existiendo en `demand/`, solo que
este paso ya no la usa.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Unidad:
    """Una solicitud de viaje activa en el simulador (persona o grupo)."""

    id: str
    origen: str
    destino: str
    minuto_llegada: float   # minuto del día (0-1439) en que llegó a la parada
    tamano: int              # personas que representa esta unidad (1 si unidad_demanda="personas")
    tiempo_espera_min: float | None = None   # fijado por env.py al subir (recogida) -- para métricas


def grupos_a_unidades(grupos_df: pd.DataFrame, unidad_demanda: str) -> list[Unidad]:
    """Convierte la tabla de grupos de `demand/src/llegadas.py` (columnas
    `grupo_id, origen, destino, minuto_dia, tamano_grupo`, entre otras) en
    una lista de `Unidad`, ordenada por minuto de llegada.

    `unidad_demanda="personas"` (default de este paso): cada grupo se explota
    en `tamano_grupo` unidades-persona, todas con el mismo origen/destino/
    minuto_llegada del grupo del que vinieron -- no cambia CUÁNDO ni DÓNDE
    aparece gente, solo la unidad atómica que ve el simulador.
    `unidad_demanda="grupos"`: una `Unidad` por fila, `tamano=tamano_grupo`.
    """
    if unidad_demanda not in ("personas", "grupos"):
        raise ValueError(f"unidad_demanda debe ser 'personas' o 'grupos', llegó {unidad_demanda!r}")

    unidades: list[Unidad] = []
    for r in grupos_df.itertuples(index=False):
        if unidad_demanda == "grupos":
            unidades.append(Unidad(
                id=str(r.grupo_id), origen=r.origen, destino=r.destino,
                minuto_llegada=float(r.minuto_dia), tamano=int(r.tamano_grupo),
            ))
        else:
            for i in range(int(r.tamano_grupo)):
                unidades.append(Unidad(
                    id=f"{r.grupo_id}_p{i}", origen=r.origen, destino=r.destino,
                    minuto_llegada=float(r.minuto_dia), tamano=1,
                ))

    unidades.sort(key=lambda u: u.minuto_llegada)
    return unidades
