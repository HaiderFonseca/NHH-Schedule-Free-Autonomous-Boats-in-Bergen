"""Estado del simulador (docs/especificacion_simulador_rl.md §4).

Dos representaciones separadas, a propósito:
- `EstadoSimulacion` / `Barco`: la representación RICA e interna. Vive el
  detalle por unidad (persona o grupo) en las colas y a bordo de cada barco
  -- necesario para manejar capacidad y recogida. `.to_dict()` reproduce el
  formato de ejemplo de la especificación, para loguear/verificar a ojo.
- `aplanar_estado()`: el vector NUMÉRICO de tamaño fijo que ve la red (fase
  siguiente). No está detallado en la especificación más allá de "se aplana
  a un vector de tamaño fijo" -- las decisiones de codificación (one-hot de
  nodos, seno/coseno para el tiempo, normalización) se explican en cada
  función y en simulacion/README.md.

**Colas por par origen-destino (no por nodo):** `colas` tiene una llave por
cada uno de los 12 pares (origen, destino) posibles entre los 4 nodos, no
una por nodo. Es la contraparte de que un barco haga siempre un viaje
DIRECTO punto a punto (ver `env.py`): la gente esperando para ir a B no se
mezcla con la que espera para ir a S, aunque estén paradas en el mismo
nodo -- son colas físicamente distintas. Con esto, un barco libre en A que
recibe la acción "ir a B" tiene una única cola de dónde embarcar: `colas[(A,B)]`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from unidades import Unidad

# Orden canónico de los 4 nodos -- fija el orden del one-hot y de los 12
# pares origen-destino. Debe coincidir con el orden de bergen-boats/config
# (kleppesto, laksevag, bryggen, sandviken); se pasa explícito a las
# funciones de este módulo en vez de hardcodearlo, pero este es el default.
NODOS_DEFAULT = ["kleppesto", "laksevag", "bryggen", "sandviken"]

# Códigos cortos para el to_dict() legible, igual que el ejemplo de la
# especificación (K/L/B/S), solo para mostrar/loguear -- no se usan en el
# vector aplanado ni en ninguna lógica interna.
CODIGO_CORTO = {"kleppesto": "K", "laksevag": "L", "bryggen": "B", "sandviken": "S"}


def pares_od(nodos: list[str]) -> list[tuple[str, str]]:
    """Los 12 pares (origen, destino) posibles entre los nodos dados, en un
    orden fijo -- única fuente de este orden, para no repetirlo suelto en
    cada módulo que lo necesite (env.py, metricas.py, visualizacion.py).
    """
    return list(permutations(nodos, 2))


def colas_vacias(nodos: list[str]) -> dict[tuple[str, str], list[Unidad]]:
    """Diccionario de colas inicializado con las 12 llaves (una por par),
    todas vacías -- usado en `env.reset()`."""
    return {par: [] for par in pares_od(nodos)}


@dataclass
class Barco:
    """Un barco de la flota. Si `nodo_origen == nodo_destino` y
    `min_para_llegar <= 0`, el barco está libre en ese nodo. Por
    construcción (viajes directos punto a punto, ver `env.py`), un barco
    libre siempre tiene `a_bordo == []` -- nunca queda gente a medio camino
    de un destino distinto al que el barco decide tomar después.
    """

    id: str
    nodo_origen: str
    nodo_destino: str
    min_para_llegar: float
    a_bordo: list[Unidad] = field(default_factory=list)

    @property
    def libre(self) -> bool:
        return self.nodo_origen == self.nodo_destino and self.min_para_llegar <= 0

    @property
    def ocupacion(self) -> int:
        return sum(u.tamano for u in self.a_bordo)


@dataclass
class EstadoSimulacion:
    """Foto completa del mundo en el paso actual (§4)."""

    t_actual_min: float          # minuto del día, 0-1439 (puede pasar de 1439 si la corrida cruza medianoche)
    dia_semana: int              # 0 (lunes) .. 6 (domingo)
    barcos: list[Barco]
    colas: dict[tuple[str, str], list[Unidad]]   # (origen,destino) -> unidades esperando, orden FIFO
    perdidas_historico: list[Unidad] = field(default_factory=list)   # para métricas al final de la corrida
    atendidas_historico: list[Unidad] = field(default_factory=list)  # idem

    def to_dict(self, nodos: list[str] = NODOS_DEFAULT) -> dict:
        """Reproduce el formato de ejemplo de la especificación (§4) --
        para logs de texto e inspección a ojo, no para la red.
        """
        demanda = {}
        for o, d in pares_od(nodos):
            clave = f"{CODIGO_CORTO[o]}->{CODIGO_CORTO[d]}"
            cola_od = self.colas.get((o, d), [])
            personas = sum(u.tamano for u in cola_od)
            espera_max = (self.t_actual_min - min(u.minuto_llegada for u in cola_od)) if cola_od else 0
            demanda[clave] = {"personas": personas, "espera_max": round(espera_max, 1)}

        return {
            "tiempo": {"minuto_del_dia": self.t_actual_min % 1440, "dia_semana": self.dia_semana},
            "barcos": [
                {
                    "origen": CODIGO_CORTO[b.nodo_origen],
                    "destino": CODIGO_CORTO[b.nodo_destino],
                    "min_para_llegar": round(b.min_para_llegar, 1),
                    "ocupacion": b.ocupacion,
                    "libre": b.libre,
                }
                for b in self.barcos
            ],
            "demanda": demanda,
        }


def dimension_vector(num_barcos: int, num_nodos: int = 4) -> int:
    """Tamaño fijo del vector aplanado: 11 valores/barco (one-hot origen +
    one-hot destino + min_para_llegar + ocupación + libre, con 4 nodos) +
    24 (12 pares O-D x 2) + 4 (tiempo cíclico). Ver `aplanar_estado`.
    """
    valores_por_barco = 2 * num_nodos + 3
    return num_barcos * valores_por_barco + num_nodos * (num_nodos - 1) * 2 + 4


def aplanar_estado(
    estado: EstadoSimulacion,
    nodos: list[str],
    capacidad_barco: int,
    min_para_llegar_norm: float,
    paciencia_norm: float = 35.0,
) -> np.ndarray:
    """Convierte `EstadoSimulacion` en un vector numérico de tamaño fijo
    (`dimension_vector(len(estado.barcos), len(nodos))`).

    Decisiones de codificación (no vienen de la especificación, que solo
    dice "se aplana a un vector de tamaño fijo" -- documentadas también en
    simulacion/README.md):
    - Nodo (origen/destino de cada barco): one-hot, no un índice entero --
      un entero 0-3 induciría una noción de "orden"/distancia entre nodos
      que no existe.
    - `min_para_llegar`: dividido por `min_para_llegar_norm` (el mayor
      tiempo de viaje de la matriz náutica) para quedar en ~[0,1].
    - `ocupación`: dividida por `capacidad_barco` -> fracción de barco lleno.
    - Demanda por par O-D: personas dividido por `capacidad_barco` (≈
      "cuántos barcos llenos están esperando ahí"), espera del más antiguo
      dividida por `paciencia_norm` (paciencia máxima plausible, 30+jitter).
    - Tiempo: minuto-del-día y día-de-semana como seno/coseno (2 valores
      cada uno) en vez de los enteros crudos -- un entero le haría creer a
      la red que el minuto 1439 y el minuto 0 están lejos, cuando en
      realidad son consecutivos (medianoche). Igual para el día de la
      semana (domingo -> lunes).
    """
    partes: list[float] = []

    for b in estado.barcos:
        partes.extend(1.0 if n == b.nodo_origen else 0.0 for n in nodos)
        partes.extend(1.0 if n == b.nodo_destino else 0.0 for n in nodos)
        partes.append(b.min_para_llegar / min_para_llegar_norm)
        partes.append(b.ocupacion / capacidad_barco)
        partes.append(1.0 if b.libre else 0.0)

    for par in pares_od(nodos):
        cola_od = estado.colas.get(par, [])
        personas = sum(u.tamano for u in cola_od)
        espera_max = (estado.t_actual_min - min(u.minuto_llegada for u in cola_od)) if cola_od else 0.0
        partes.append(personas / capacidad_barco)
        partes.append(espera_max / paciencia_norm)

    minuto = estado.t_actual_min % 1440
    partes.append(math.sin(2 * math.pi * minuto / 1440))
    partes.append(math.cos(2 * math.pi * minuto / 1440))
    partes.append(math.sin(2 * math.pi * estado.dia_semana / 7))
    partes.append(math.cos(2 * math.pi * estado.dia_semana / 7))

    return np.array(partes, dtype=np.float32)
