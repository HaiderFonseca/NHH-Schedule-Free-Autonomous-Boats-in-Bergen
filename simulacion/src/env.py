"""Entorno Gymnasium del simulador (docs/especificacion_simulador_rl.md §3-§8,
con un cambio de diseño deliberado: sin paciencia/pérdida, ver más abajo).

`SimuladorBarcosBergen` es la TRANSICIÓN del MDP: recibe una acción por
barco, avanza el mundo `paso_tiempo_min` (2 min por defecto) y devuelve el
nuevo estado (aplanado) y la recompensa. No decide nada por sí mismo -- la
política de referencia (`politica_base.py`) o, más adelante, el agente,
viven FUERA del entorno y llaman a `step()` desde un bucle de control,
exactamente como pide la especificación (el "no reescribir el simulador"
al conectar el agente después).

**Viajes directos punto a punto.** Un barco libre en A que recibe la orden
"ir a B" embarca únicamente desde `colas[(A,B)]` (FIFO, hasta capacidad) y
viaja directo a B, donde baja a TODOS los que lleva (por construcción,
todos van a B -- nunca hay destinos mixtos a bordo). Esto es más simple y
más honesto que una versión anterior que permitía embarcar de cualquier
destino y solo bajar a los que coincidían con la parada: esa lógica asumía
rutas con paradas intermedias que este proyecto no modela todavía.

**Sin paciencia ni pérdidas (cambio de diseño sobre la especificación
original).** La especificación original retiraba de la cola, como
"perdido", a quien esperara más que su paciencia (15/30 min +- jitter).
Se quitó por completo: nadie se va nunca, espera hasta ser atendido (o
hasta que termina la corrida, quedando "esperando al final" -- una
categoría que la verificación de conservación ya contempla). Motivo:
retirar a alguien de la cola borraba justo el dato que hace falta para
definir una garantía de tiempo con fundamento -- cuánto se tarda REALMENTE
en atender a alguien en el peor caso, sin censurar los casos malos con un
límite artificial. La recompensa (`recompensa.py`) sigue penalizando la
espera larga (con un techo por persona, para que no crezca sin límite),
pero ya no hay un evento de "pérdida" aparte ni un campo de paciencia en
`unidades.Unidad`.

Nada se recalcula aquí: los tiempos de viaje vienen de la matriz náutica de
`bergen-boats/02_ruteo_navegable`, y las unidades de demanda vienen de
`demand/src/llegadas.py` (vía `unidades.grupos_a_unidades`, ya generadas
antes de construir el entorno -- ver `notebooks/00_preparar_demanda_escalones.ipynb`).

**Registros para análisis (no los ve el agente):** `log_eventos` (una fila
por cada sube/baja/decision/movimiento), `log_recompensa` (desglose por
paso) y `historial_estados` (snapshot `estado.to_dict()` por paso) --
consumidos por `metricas.py` y `visualizacion.py`, nunca por la política ni
por el vector aplanado que ve el agente.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from estado import Barco, EstadoSimulacion, aplanar_estado, colas_vacias, dimension_vector
from recompensa import calcular_recompensa
from unidades import Unidad, grupos_a_unidades


class SimuladorBarcosBergen(gym.Env):
    """Entorno de despacho de barcos a demanda en Bergen.

    Espacios:
    - `action_space`: `MultiDiscrete([5] * num_barcos)`. Índices 0..3 = ir al
      nodo `nodos[i]`; índice 4 = esperar. Ignorado para barcos ocupados
      (siguen hasta llegar, la especificación no permite redirigir a media
      ruta).
    - `observation_space`: `Box` de tamaño fijo (`estado.dimension_vector`),
      el vector aplanado de `estado.aplanar_estado`.

    `info` (de `reset`/`step`) incluye `"estado_dict"` (JSON-friendly, para
    logs) y `"_estado_obj"` (el objeto `EstadoSimulacion` real, con las
    colas y el detalle por unidad) -- este último es lo que necesita
    `politica_base.politica_base()` para decidir; no es parte de la
    observación formal del agente, es un canal para el bucle de control de
    ESTE paso (política de referencia), tal como Gymnasium recomienda usar
    `info` para datos de depuración fuera de la observación.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grupos_df: pd.DataFrame,
        matriz_tiempos: pd.DataFrame,
        nodos: list[str],
        num_barcos: int,
        capacidad_barco: int,
        nodo_inicial: str,
        paso_tiempo_min: float,
        hora_inicio_min: float,
        hora_fin_min: float,
        cfg_recompensa: dict,
        unidad_demanda: str = "personas",
        dia_semana: int = 0,
    ):
        super().__init__()
        self.grupos_df = grupos_df
        self.matriz_tiempos = matriz_tiempos
        self.nodos = list(nodos)
        self.num_barcos = num_barcos
        self.capacidad_barco = capacidad_barco
        self.nodo_inicial = nodo_inicial
        self.paso_tiempo_min = paso_tiempo_min
        self.hora_inicio_min = hora_inicio_min
        self.hora_fin_min = hora_fin_min
        self.cfg_recompensa = cfg_recompensa
        self.unidad_demanda = unidad_demanda
        self.dia_semana = dia_semana

        self._min_para_llegar_norm = float(matriz_tiempos.values.max())
        # Índice 4 (None) = "esperar". Ver docstring de la clase.
        self.acciones_posibles: list[str | None] = list(self.nodos) + [None]

        self.action_space = gym.spaces.MultiDiscrete([len(self.acciones_posibles)] * num_barcos)
        dim = dimension_vector(num_barcos, len(self.nodos))
        self.observation_space = gym.spaces.Box(low=-10.0, high=100.0, shape=(dim,), dtype=np.float32)

    def codificar_accion_barco(self, nodo_o_none: str | None) -> int:
        """Convierte la decisión de una política (nodo destino, o None =
        esperar) al índice que espera `action_space`. Único lugar que sabe
        la codificación -- así el bucle de control (notebook) no la
        duplica.
        """
        return self.acciones_posibles.index(nodo_o_none)

    # -- ciclo de vida Gymnasium -------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t_actual_min = float(self.hora_inicio_min)
        self.unidades_pendientes: list[Unidad] = grupos_a_unidades(self.grupos_df, self.unidad_demanda)
        self.total_unidades_generadas = len(self.unidades_pendientes)
        self._idx_siguiente_unidad = 0
        self.colas: dict[tuple[str, str], list[Unidad]] = colas_vacias(self.nodos)
        self.barcos: list[Barco] = [
            Barco(id=f"barco_{i}", nodo_origen=self.nodo_inicial, nodo_destino=self.nodo_inicial, min_para_llegar=0.0)
            for i in range(self.num_barcos)
        ]
        self.atendidas_historico: list[Unidad] = []
        self.log_eventos: list[dict] = []
        self.log_recompensa: list[dict] = []
        self.historial_estados: list[dict] = []

        self._incorporar_unidades_hasta(self.t_actual_min)
        # Sin embarque aquí: en este modelo el embarque va SIEMPRE ligado a
        # una decisión de "ir a B" (ver step()), y todavía no se pidió
        # ninguna acción -- los barcos arrancan libres y vacíos en su nodo
        # inicial, esperando la primera decisión.

        estado = self._construir_estado()
        self.historial_estados.append(estado.to_dict(self.nodos))
        obs = self._observar(estado)
        info = {"estado_dict": estado.to_dict(self.nodos), "_estado_obj": estado}
        return obs, info

    def step(self, action):
        movimientos_iniciados = 0

        # 1) aplicar accion a los barcos libres: embarcar (solo del par exacto
        #    A->B elegido) y partir. Los ocupados ignoran su entrada, siguen.
        for i, barco in enumerate(self.barcos):
            if not barco.libre:
                continue
            A = barco.nodo_origen
            destino_elegido = self.acciones_posibles[int(action[i])]
            self.log_eventos.append({
                "tipo": "decision", "minuto": self.t_actual_min, "barco_id": barco.id,
                "decision": destino_elegido if destino_elegido is not None else "esperar",
            })
            if destino_elegido is None or destino_elegido == A:
                continue  # esperar, no-op

            B = destino_elegido
            self._embarcar_para_viaje(barco, A, B)
            barco.nodo_destino = B
            barco.min_para_llegar = float(self.matriz_tiempos.loc[A, B])
            movimientos_iniciados += 1
            self.log_eventos.append({
                "tipo": "movimiento", "minuto": self.t_actual_min, "barco_id": barco.id,
                "desde": A, "hasta": B, "ocupacion": barco.ocupacion,
            })

        # 2) avanzar paso_tiempo_min; el que llega baja a TODOS (por construccion,
        #    todos a bordo van al mismo destino) y queda libre
        self.t_actual_min += self.paso_tiempo_min
        for barco in self.barcos:
            if barco.nodo_origen != barco.nodo_destino:
                barco.min_para_llegar -= self.paso_tiempo_min
                if barco.min_para_llegar <= 1e-9:
                    barco.nodo_origen = barco.nodo_destino
                    barco.min_para_llegar = 0.0
                    self._desembarcar(barco)

        # 3) incorporar unidades nuevas del intervalo (a las colas, para el
        #    proximo paso -- no alcanzan a subir al barco que acaba de partir
        #    en el paso 1, orden exacto de la especificacion §6)
        self._incorporar_unidades_hasta(self.t_actual_min)

        # 4) recompensa + nuevo estado (ya no hay paso de "purgar perdidas" --
        #    nadie se va nunca, ver docstring del modulo)
        estado = self._construir_estado()
        r, desglose = calcular_recompensa(estado, self.cfg_recompensa)
        self.log_recompensa.append({"minuto": self.t_actual_min, **desglose, "total": r})

        terminated = False   # no hay estado terminal natural en despacho continuo
        truncated = self.t_actual_min >= self.hora_fin_min   # limite de tiempo del escalon configurado

        estado_dict = estado.to_dict(self.nodos)
        self.historial_estados.append(estado_dict)
        obs = self._observar(estado)
        info = {
            "estado_dict": estado_dict,
            "_estado_obj": estado,
            "recompensa_desglose": desglose,
            "movimientos_iniciados": movimientos_iniciados,
        }
        return obs, r, terminated, truncated, info

    # -- mecánica interna ---------------------------------------------------

    def _observar(self, estado: EstadoSimulacion) -> np.ndarray:
        return aplanar_estado(estado, self.nodos, self.capacidad_barco, self._min_para_llegar_norm)

    def _incorporar_unidades_hasta(self, t: float) -> None:
        pendientes = self.unidades_pendientes
        while self._idx_siguiente_unidad < len(pendientes) and pendientes[self._idx_siguiente_unidad].minuto_llegada <= t:
            u = pendientes[self._idx_siguiente_unidad]
            self.colas[(u.origen, u.destino)].append(u)
            self._idx_siguiente_unidad += 1

    def _embarcar_para_viaje(self, barco: Barco, origen: str, destino: str) -> None:
        """Embarca desde `colas[(origen,destino)]`, FIFO hasta capacidad --
        único punto de embarque del modelo (fundido con la decisión de
        moverse, ver `step()`). Si la siguiente unidad no cabe entera, se
        detiene ahí (con `unidad_demanda="personas"`, tamano=1, esto nunca
        se activa -- siempre cabe si queda espacio).
        """
        cola = self.colas[(origen, destino)]
        espacio = self.capacidad_barco - barco.ocupacion
        embarcados = []
        for u in cola:
            if u.tamano <= espacio:
                embarcados.append(u)
                espacio -= u.tamano
            else:
                break
        if not embarcados:
            return
        for u in embarcados:
            u.tiempo_espera_min = self.t_actual_min - u.minuto_llegada
            self.log_eventos.append({
                "tipo": "sube", "minuto": self.t_actual_min, "barco_id": barco.id,
                "unidad_id": u.id, "par": (origen, destino), "tiempo_espera": u.tiempo_espera_min,
            })
        ids_embarcados = {u.id for u in embarcados}
        self.colas[(origen, destino)] = [u for u in cola if u.id not in ids_embarcados]
        barco.a_bordo.extend(embarcados)

    def _desembarcar(self, barco: Barco) -> None:
        """Baja a TODOS los que lleva el barco -- por construcción (viajes
        directos punto a punto), todos a bordo comparten el mismo destino:
        el nodo al que el barco acaba de llegar.
        """
        if not barco.a_bordo:
            return
        for u in barco.a_bordo:
            minuto_embarque = u.minuto_llegada + (u.tiempo_espera_min or 0.0)
            tiempo_viaje = self.t_actual_min - minuto_embarque
            self.log_eventos.append({
                "tipo": "baja", "minuto": self.t_actual_min, "barco_id": barco.id,
                "unidad_id": u.id, "par": (u.origen, u.destino), "tiempo_viaje": tiempo_viaje,
            })
        self.atendidas_historico.extend(barco.a_bordo)
        barco.a_bordo = []

    def _construir_estado(self) -> EstadoSimulacion:
        return EstadoSimulacion(
            t_actual_min=self.t_actual_min,
            dia_semana=self.dia_semana,
            barcos=self.barcos,
            colas=self.colas,
            atendidas_historico=self.atendidas_historico,
        )
