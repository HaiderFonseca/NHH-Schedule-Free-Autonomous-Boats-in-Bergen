"""`EntornoRecompensaIntercambiable` -- subclase de `EntornoDemandaAleatoria`
que recalcula la recompensa de cada paso con la fórmula elegida (A a H),
SIN tocar `env.py` ni `recompensa.py`.

`env.py.step()` ya llama a `calcular_recompensa` por dentro y no se puede
intercambiar desde afuera -- la salida se descarta y se recalcula acá, con
el MISMO `EstadoSimulacion` que `env.py` ya construyó (`info["_estado_obj"]`,
ya expuesto por el padre), sin reconstruir ni duplicar nada de la mecánica
de simulación. Mismo patrón que ya usa `EntornoDemandaAleatoria` para no
tocar `env.py`: subclasear, no editar.

**Por qué C y G no guardan un `EstadoSimulacion` de "antes":** se probó
así al principio (guardar `self._estado_previo = estado_despues` y leerlo
en el paso siguiente) y tenía un bug real -- `EstadoSimulacion.colas`/
`.barcos` (`estado.py`) se construyen pasando `self.colas`/`self.barcos`
de `env.py` POR REFERENCIA (`_construir_estado`), no una copia. Un
`EstadoSimulacion` guardado de un paso anterior no queda "congelado": sus
colas/barcos son los MISMOS objetos que el simulador sigue mutando en cada
paso siguiente, así que leerlo más tarde devuelve el estado ACTUAL, no el
histórico (confirmado con una prueba directa: un snapshot tomado en
`reset()`, leído 8 pasos después, mostraba unidades "aparecidas" que en
realidad llegaron después -- solo el campo `t_actual_min`, un float simple
copiado por valor, se mantenía correcto). Por eso acá se calcula y guarda
un ESCALAR (`self._potencial_previo`, `self._s_tiempo_previo`) en el
momento exacto en que el estado todavía es fresco -- nunca se guarda el
objeto `EstadoSimulacion` en sí para leerlo después.
"""
from __future__ import annotations

from entrenamiento import EntornoDemandaAleatoria
from recompensas_alternativas import CALCULADORAS, potencial, suma_tiempo_activo


class EntornoRecompensaIntercambiable(EntornoDemandaAleatoria):
    """`tipo_recompensa`: una clave de `recompensas_alternativas.CALCULADORAS`
    ("A" a "H"). `cfg_recompensa_tipo`: el cfg de ESE tipo
    (`prueba_rewards/config/instance.yaml` -> `recompensas.<tipo>`).
    `gamma`: solo lo usa C (descuento del término potencial) -- se pasa
    aparte porque `env.py`/`recompensa.py` no tienen noción de descuento
    (es un concepto del agente, no del simulador).

    Para C y G, que necesitan "cómo estaba el mundo antes de este paso",
    se precalculan y guardan ESCALARES (`self._potencial_previo`,
    `self._s_tiempo_previo`) al final de cada `step()`/`reset()`, mientras
    el estado todavía es fresco -- ver el docstring del módulo para por qué
    no alcanza con guardar el `EstadoSimulacion`. Quien usa esta clase no
    tiene que hacer nada especial, se actualiza solo.
    """

    def __init__(self, tipo_recompensa: str, cfg_recompensa_tipo: dict, gamma: float, **kwargs_env):
        if tipo_recompensa not in CALCULADORAS:
            raise ValueError(f"tipo_recompensa debe ser uno de {list(CALCULADORAS)}, no {tipo_recompensa!r}")
        self.tipo_recompensa = tipo_recompensa
        self.cfg_recompensa_tipo = cfg_recompensa_tipo
        self.gamma = gamma
        self._potencial_previo = None
        self._s_tiempo_previo = None
        super().__init__(**kwargs_env)

    def _actualizar_escalares_previos(self, estado) -> None:
        """Se llama con el estado RECIÉN construido (fresco, sin mutar
        todavía) -- al final de `reset()` y de cada `step()`."""
        if self.tipo_recompensa == "C":
            self._potencial_previo = potencial(estado, self.cfg_recompensa_tipo["eta_potencial"])
        elif self.tipo_recompensa == "G":
            self._s_tiempo_previo = suma_tiempo_activo(estado)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._actualizar_escalares_previos(info["_estado_obj"])
        return obs, info

    def step(self, action):
        # `entregas_este_paso` no viene expuesto en `info` (env.py no lo
        # devuelve) -- se pasa 0.0 fijo, correcto para todas las config de
        # este experimento (premio_por_persona_entregada=0.0 en todas). Si
        # algún día alguna usara un premio != 0, habría que exponer el
        # valor real desde `env.py` primero.
        #
        # `len_atendidas_antes`: para G (tiempo incremental), hace falta
        # saber exactamente qué unidades fueron ENTREGADAS en este paso --
        # `env.atendidas_historico` (heredado de `SimuladorBarcosBergen`) se
        # extiende una sola vez por paso, dentro de `super().step()`
        # (`env.py`, `_desembarcar`), así que la porción nueva del final de
        # la lista es exactamente eso, sin tener que tocar `env.py`.
        len_atendidas_antes = len(self.atendidas_historico)
        obs, _r_descartado, terminated, truncated, info = super().step(action)
        estado_despues = info["_estado_obj"]
        unidades_entregadas_este_paso = self.atendidas_historico[len_atendidas_antes:]

        if self.tipo_recompensa == "C":
            r, desglose = CALCULADORAS["C"](
                estado_despues, self._potencial_previo, self.cfg_recompensa_tipo, self.gamma,
                entregas_este_paso=0.0,
            )
        elif self.tipo_recompensa == "G":
            r, desglose = CALCULADORAS["G"](
                estado_despues, self._s_tiempo_previo, unidades_entregadas_este_paso, self.cfg_recompensa_tipo,
                entregas_este_paso=0.0,
            )
        elif self.tipo_recompensa == "A":
            r, desglose = CALCULADORAS["A"](
                estado_despues, self.cfg_recompensa_tipo, self.paso_tiempo_min,
                entregas_este_paso=0.0,
            )
        else:  # "B", "D", "E", "F", "H" -- misma firma que calcular_recompensa original
            r, desglose = CALCULADORAS[self.tipo_recompensa](
                estado_despues, self.cfg_recompensa_tipo, entregas_este_paso=0.0,
            )

        # Se actualiza DESPUES de calcular la recompensa de este paso (para
        # que arriba se use el valor del paso ANTERIOR), y con el estado
        # recien construido, todavia fresco -- ver docstring del modulo.
        self._actualizar_escalares_previos(estado_despues)
        info["recompensa_desglose"] = desglose
        return obs, r, terminated, truncated, info
