"""Envoltura de entrenamiento: demanda fresca en cada `reset()` (CRÍTICO
para RL, ver `simulacion/README.md`) -- sin tocar `env.py`.

**Por qué un wrapper y no modificar `SimuladorBarcosBergen`.** `env.py` no
sabe nada de `demand/` -- recibe un `grupos_df` ya construido, nunca
recalcula nada (ver su docstring: "nada se recalcula aquí"). Ese desacople
es deliberado: política base y los escalones 1-3 (demanda FIJA, pregenerada
una vez) siguen usando `SimuladorBarcosBergen` tal cual. Meterle a `env.py`
la generación de demanda (rutas a `demand/`, `intensidad_od`,
`conexiones_fuertes`, etc.) rompería ese desacople para todo el mundo, no
solo para quien entrena. En vez de eso, `EntornoDemandaAleatoria` es una
subclase que solo cambia una cosa: qué `grupos_df` usa cada episodio.

**Por qué hace falta esto para RL, y no para la política base.** La
política base es una regla fija (`politica_base.py`) -- no tiene parámetros
que ajustar, así que correrla siempre sobre la MISMA demanda (como hacen
los escalones 1-3) es correcto: se está verificando el simulador, no
entrenando nada. Un agente de RL sí tiene parámetros que se ajustan viendo
episodios repetidos -- si esos episodios fueran siempre la misma tabla de
grupos, el agente podría memorizar esa realización particular (qué grupo
exacto aparece en qué minuto exacto) en vez de aprender una política que
generaliza sobre el patrón de demanda real (mismas franjas/intensidades,
llegadas concretas distintas). Cada `reset()` sin semilla explícita genera
una realización NUEVA (muestreo Montecarlo de episodios), para que lo que
el agente aprenda sea la ESTRUCTURA del problema, no una instancia de ella.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from env import SimuladorBarcosBergen


class EntornoDemandaAleatoria(SimuladorBarcosBergen):
    """Mismo entorno que `SimuladorBarcosBergen`, pero regenera `grupos_df`
    en cada `reset()` en vez de usar una tabla fija.

    - `reset(seed=None)` (el caso normal durante el entrenamiento -- SB3 no
      manda una semilla en cada episodio): genera una demanda NUEVA, con una
      semilla sacada de una secuencia propia (`semilla_entrenamiento` al
      construir -> secuencia reproducible; `None` -> no reproducible).
    - `reset(seed=42)` (caso explícito -- evaluación, comparación contra la
      política base, verificación): genera la demanda con ESA semilla
      exacta, reproducible igual que `SimuladorBarcosBergen` hoy. La misma
      clase sirve para entrenar (semilla libre) y para evaluar (semilla
      fija, para comparar en igualdad de condiciones).
    """

    def __init__(
        self,
        generar_llegadas_dia_fn: Callable,
        cfg_demand: dict,
        intensidad_od: pd.DataFrame,
        conexiones_fuertes: list,
        poblacion_total_zonas: float,
        horas: tuple[float, float],
        porcentaje_poblacion_dia: float,
        es_fin_de_semana: bool = False,
        semilla_entrenamiento: int | None = None,
        **kwargs_env,
    ):
        # `generar_llegadas_dia_fn` se pasa como parámetro (no se importa
        # `demand/src/llegadas.py` aquí) -- así este módulo no necesita
        # saber la ruta de `demand/src` ni tocar `sys.path`, exactamente
        # como ya hace el notebook que construye este entorno (igual patrón
        # que `00_preparar_demanda_escalones.ipynb`, inyección de
        # dependencia en vez de import cruzado hardcodeado entre carpetas).
        self._generar_llegadas_dia_fn = generar_llegadas_dia_fn
        self._cfg_demand = cfg_demand
        self._intensidad_od = intensidad_od
        self._conexiones_fuertes = conexiones_fuertes
        self._poblacion_total_zonas = poblacion_total_zonas
        self._horas = horas
        self._porcentaje_poblacion_dia = porcentaje_poblacion_dia
        self._es_fin_de_semana = es_fin_de_semana
        self._rng_semillas = np.random.default_rng(semilla_entrenamiento)
        self._contador_episodio = 0

        grupos_vacio = pd.DataFrame(columns=["grupo_id", "origen", "destino", "minuto_dia", "tamano_grupo"])
        super().__init__(grupos_df=grupos_vacio, **kwargs_env)

    def _generar_grupos(self, semilla: int) -> pd.DataFrame:
        cfg_mod = dict(self._cfg_demand)
        cfg_mod["demanda"] = dict(self._cfg_demand["demanda"])
        cfg_mod["demanda"]["porcentaje_poblacion_dia"] = self._porcentaje_poblacion_dia
        rng = np.random.default_rng(semilla)
        grupos = self._generar_llegadas_dia_fn(
            cfg_mod, self._intensidad_od, self._conexiones_fuertes, self._poblacion_total_zonas,
            es_fin_de_semana=self._es_fin_de_semana, rng=rng, dia_id=f"ep{self._contador_episodio}",
        )
        hora_ini, hora_fin = self._horas
        grupos = grupos[(grupos["hora"] >= hora_ini) & (grupos["hora"] < hora_fin)]
        # `self.nodos` ya existe (lo fija `SimuladorBarcosBergen.__init__`, corrido en
        # `super().__init__()` del constructor de esta clase, antes de que `reset()` -- y por
        # lo tanto este metodo -- se pueda llamar). Si es un subconjunto de los 4 nodos reales
        # (p.ej. `escalon_toy`, solo bryggen/kleppesto), la demanda generada se restringe a los
        # pares dentro de ese subconjunto -- sin tocar `demand/`, mismo patron que el filtro de
        # horas de arriba.
        grupos = grupos[grupos["origen"].isin(self.nodos) & grupos["destino"].isin(self.nodos)]
        return grupos.reset_index(drop=True)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._contador_episodio += 1
        semilla_demanda = seed if seed is not None else int(self._rng_semillas.integers(0, 2**31 - 1))
        self.grupos_df = self._generar_grupos(semilla_demanda)
        return super().reset(seed=seed, options=options)
