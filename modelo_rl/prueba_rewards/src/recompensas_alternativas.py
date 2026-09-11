"""Cuatro formas de recompensa, intercambiables, todas calculadas sobre el
MISMO `EstadoSimulacion` que ya usa `simulador/src/recompensa.py` -- ver
`modelo_rl/prueba_rewards/README.md` para la teoría detrás de cada una.

B y D son la fórmula YA EXISTENTE en `recompensa.calcular_recompensa` --
se importan y se llaman tal cual (con distinto `cfg_recompensa`), nunca se
reimplementan acá. A y C son fórmulas nuevas, genuinamente distintas de esa
fórmula base.
"""
from __future__ import annotations

from estado import EstadoSimulacion
from recompensa import calcular_recompensa, _unidades_activas

# Reexportadas para que quien importe este módulo tenga las 4 en un solo
# lugar -- B y D no necesitan wrapper propio, son la función original.
calcular_recompensa_B = calcular_recompensa
calcular_recompensa_D = calcular_recompensa


def calcular_recompensa_A(
    estado: EstadoSimulacion, cfg_recompensa: dict, paso_tiempo_min: float,
    entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """Objetivo puro, derivado de la identidad de Little (README, sección
    2.A): sin tolerancia, sin techo, sin cuadrado -- lineal en el tiempo que
    cada persona pasa activa. `entregas_este_paso` no se usa (no hay premio
    de entrega en A); queda en la firma solo para que las 4 funciones
    tengan la misma forma de llamado.
    """
    peso_movimiento = cfg_recompensa["peso_movimiento"]

    costo_tiempo = sum(u.tamano * paso_tiempo_min for u in _unidades_activas(estado))
    barcos_en_movimiento = sum(1 for b in estado.barcos if b.nodo_origen != b.nodo_destino)
    penalizacion_movimiento = peso_movimiento * barcos_en_movimiento

    desglose = {"tiempo_activo": costo_tiempo, "movimiento": penalizacion_movimiento, "entrega": 0.0}
    r = -(costo_tiempo + penalizacion_movimiento)
    return r, desglose


def potencial(estado: EstadoSimulacion, eta: float) -> float:
    """Phi(s) = -eta * N(s), N(s) = suma de `tamano` de toda unidad activa
    (esperando en cola o a bordo, sin entregar aún) -- ver README, sección
    2.C. Negativo: más gente sin entregar, "peor" potencial (más lejos de
    la meta), igual que un `-distancia_a_la_meta` en el ejemplo clásico de
    Ng et al. (1999).
    """
    n_activas = sum(u.tamano for u in _unidades_activas(estado))
    return -eta * n_activas


def calcular_recompensa_C(
    estado_antes: EstadoSimulacion, estado_despues: EstadoSimulacion,
    cfg_recompensa: dict, gamma: float, entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """B + potential-based shaping (Ng, Harada & Russell 1999) -- README,
    sección 2.C. `r = r_B(estado_despues) + gamma*Phi(estado_despues) -
    Phi(estado_antes)`. Necesita el estado ANTES (`s`) y DESPUÉS (`s'`) del
    paso -- lo da `EntornoRecompensaIntercambiable`, que los rastrea entre
    llamadas a `step()` (ver `entorno_recompensa_intercambiable.py`).
    """
    r_b, desglose_b = calcular_recompensa(estado_despues, cfg_recompensa, entregas_este_paso)
    eta = cfg_recompensa["eta_potencial"]
    termino_shaping = gamma * potencial(estado_despues, eta) - potencial(estado_antes, eta)

    desglose = dict(desglose_b, shaping=termino_shaping)
    r = r_b + termino_shaping
    return r, desglose


CALCULADORAS = {"A": calcular_recompensa_A, "B": calcular_recompensa_B, "C": calcular_recompensa_C, "D": calcular_recompensa_D}
