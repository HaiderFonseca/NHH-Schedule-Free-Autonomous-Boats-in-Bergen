"""`EntornoRecompensaIntercambiable` -- subclase de `EntornoDemandaAleatoria`
que recalcula la recompensa de cada paso con la fórmula elegida (A/B/C/D),
SIN tocar `env.py` ni `recompensa.py`.

`env.py.step()` ya llama a `calcular_recompensa` por dentro y no se puede
intercambiar desde afuera -- la salida se descarta y se recalcula acá, con
el MISMO `EstadoSimulacion` que `env.py` ya construyó (`info["_estado_obj"]`,
ya expuesto por el padre), sin reconstruir ni duplicar nada de la mecánica
de simulación. Mismo patrón que ya usa `EntornoDemandaAleatoria` para no
tocar `env.py`: subclasear, no editar.
"""
from __future__ import annotations

from entrenamiento import EntornoDemandaAleatoria
from recompensas_alternativas import CALCULADORAS


class EntornoRecompensaIntercambiable(EntornoDemandaAleatoria):
    """`tipo_recompensa`: "A"|"B"|"C"|"D". `cfg_recompensa_tipo`: el cfg de
    ESE tipo (`prueba_rewards/config/instance.yaml` -> `recompensas.<tipo>`).
    `gamma`: solo lo usa C (descuento del término potencial) -- se pasa
    aparte porque `env.py`/`recompensa.py` no tienen noción de descuento
    (es un concepto del agente, no del simulador).

    Rastrea `self._estado_previo` (el estado ANTES del paso que se está por
    dar) para que C pueda calcular Phi(s) y Phi(s') -- se actualiza solo,
    en `reset()` y al final de cada `step()`; quien usa esta clase no tiene
    que hacer nada especial.
    """

    def __init__(self, tipo_recompensa: str, cfg_recompensa_tipo: dict, gamma: float, **kwargs_env):
        if tipo_recompensa not in CALCULADORAS:
            raise ValueError(f"tipo_recompensa debe ser uno de {list(CALCULADORAS)}, no {tipo_recompensa!r}")
        self.tipo_recompensa = tipo_recompensa
        self.cfg_recompensa_tipo = cfg_recompensa_tipo
        self.gamma = gamma
        self._estado_previo = None
        super().__init__(**kwargs_env)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._estado_previo = info["_estado_obj"]
        return obs, info

    def step(self, action):
        # `entregas_este_paso` no viene expuesto en `info` (env.py no lo
        # devuelve) -- se pasa 0.0 fijo, correcto para las 4 config de este
        # experimento (premio_por_persona_entregada=0.0 en todas). Si algún
        # día una de las 4 usara un premio != 0, habría que exponer el
        # valor real desde `env.py` primero.
        obs, _r_descartado, terminated, truncated, info = super().step(action)
        estado_despues = info["_estado_obj"]

        if self.tipo_recompensa == "C":
            r, desglose = CALCULADORAS["C"](
                self._estado_previo, estado_despues, self.cfg_recompensa_tipo, self.gamma,
                entregas_este_paso=0.0,
            )
        elif self.tipo_recompensa == "A":
            r, desglose = CALCULADORAS["A"](
                estado_despues, self.cfg_recompensa_tipo, self.paso_tiempo_min,
                entregas_este_paso=0.0,
            )
        else:  # "B" o "D" -- misma firma que calcular_recompensa original
            r, desglose = CALCULADORAS[self.tipo_recompensa](
                estado_despues, self.cfg_recompensa_tipo, entregas_este_paso=0.0,
            )

        self._estado_previo = estado_despues
        info["recompensa_desglose"] = desglose
        return obs, r, terminated, truncated, info
