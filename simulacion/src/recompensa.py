"""Recompensa del simulador (docs/especificacion_simulador_rl.md §7).

Función aislada y pura: recibe el estado ya transicionado y qué unidades se
perdieron en este paso, devuelve un escalar (negativo) y el desglose por
término. Todos los pesos vienen de `cfg["recompensa"]`, nada hardcodeado --
así se puede sustituir/ajustar sin tocar el simulador (`env.py`).

r = -[ Σ_unidades_activas (sobrante/sobrante_max)²
       + peso_perdido * (unidades perdidas este paso)
       + peso_movimiento * (barcos en movimiento) ]

Notas de diseño:
- La penalización de incomodidad se pondera por `unidad.tamano` (así lo pide
  la especificación: "se suma sobre todos los PASAJEROS en el sistema") --
  en modo "personas" tamano=1 y no cambia nada; en modo "grupos" un grupo de
  8 personas pesa 8x, no 1x.
- La penalización de pérdida, en cambio, es POR UNIDAD (`len(...)`, no
  `sum(tamano)`) -- así colapsa exactamente a "por persona" en modo
  "personas" y a "por grupo" (como dice la especificación original, ≈1.3
  por grupo, sin escalar por tamaño) en modo "grupos", sin ninguna rama
  especial para cada caso.
- "Unidades activas" incluye las que van A BORDO de un barco en ruta, no
  solo las que esperan en la parada -- la especificación dice literalmente
  "cada pasajero que aún no llega a su destino". Por eso una `Unidad` no
  cambia su `minuto_llegada` al embarcar: el tiempo en el sistema sigue
  contando hasta la entrega real.

Bug detectado en la especificación (documentado también en
simulacion/README.md): `paciencia_min` en conexiones fuertes es 15 min con
jitter +-5 (`demand/config/instance.yaml`), así que puede caer hasta 10 min
-- por debajo de la tolerancia de 12 min. La fórmula literal
`sobrante_max = paciencia - tolerancia` daría un número negativo o cero para
esos casos (división inválida). Se corrige con un piso configurable
(`recompensa.sobrante_max_minimo`, default 1.0): para esos casos raros la
penalización satura en ~1.0 casi de inmediato, que es razonable (alguien a
punto de perderse). El generador de demanda y su jitter no se tocan.
"""
from __future__ import annotations

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def _unidades_activas(estado: EstadoSimulacion) -> list[Unidad]:
    activas = [u for cola in estado.colas.values() for u in cola]
    for b in estado.barcos:
        activas.extend(b.a_bordo)
    return activas


def calcular_recompensa(
    estado: EstadoSimulacion,
    unidades_perdidas_este_paso: list[Unidad],
    cfg_recompensa: dict,
) -> tuple[float, dict]:
    """Recompensa del paso (negativa) + desglose por término, para poder
    loguear/graficar cada componente por separado durante la verificación.
    """
    tolerancia = cfg_recompensa["tolerancia_incomodidad_min"]
    eps = cfg_recompensa["sobrante_max_minimo"]
    peso_perdido = cfg_recompensa["peso_perdido"]
    peso_movimiento = cfg_recompensa["peso_movimiento"]

    penalizacion_incomodidad = 0.0
    for u in _unidades_activas(estado):
        tiempo_en_sistema = estado.t_actual_min - u.minuto_llegada
        sobrante = max(0.0, tiempo_en_sistema - tolerancia)
        sobrante_max = max(eps, u.paciencia_min - tolerancia)
        penalizacion_incomodidad += u.tamano * (sobrante / sobrante_max) ** 2

    penalizacion_perdida = peso_perdido * len(unidades_perdidas_este_paso)

    barcos_en_movimiento = sum(1 for b in estado.barcos if b.nodo_origen != b.nodo_destino)
    penalizacion_movimiento = peso_movimiento * barcos_en_movimiento

    desglose = {
        "incomodidad": penalizacion_incomodidad,
        "perdida": penalizacion_perdida,
        "movimiento": penalizacion_movimiento,
    }
    r = -(penalizacion_incomodidad + penalizacion_perdida + penalizacion_movimiento)
    return r, desglose
