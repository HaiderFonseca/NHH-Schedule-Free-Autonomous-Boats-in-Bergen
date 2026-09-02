"""Recompensa del simulador (docs/especificacion_simulador_rl.md §7, muy
simplificada -- ver también `unidades.py` y `env.py` para el cambio de
diseño más grande: ya no existe el concepto de "paciencia"/"pérdida").

Función aislada y pura: recibe el estado ya transicionado, devuelve un
escalar (negativo) y el desglose por término. Todos los pesos vienen de
`cfg["recompensa"]`, nada hardcodeado -- así se puede sustituir/ajustar sin
tocar el simulador (`env.py`).

r = -[ Σ_unidades_activas min(techo, (sobrante / sobrante_normalizador)²)
       + peso_movimiento * (barcos en movimiento) ]

**Historia de los cambios sobre la especificación original** (por si hace
falta reconstruir el razonamiento):

1. Se quitó el término de pérdida (`peso_perdido * unidades_perdidas`) --
   se sentía redundante con la incomodidad, que ya llega a su techo justo
   antes de que alguien se perdiera.
2. `sobrante_max` pasó de calcularse por unidad (`paciencia - tolerancia`,
   que podía dar negativo con paciencia corta + jitter) a una constante
   fija (`sobrante_normalizador_min`) -- sin casos especiales.
3. **Se quitó la paciencia por completo, en todo el simulador (no solo
   aquí).** Antes, alguien que esperaba más de su paciencia se retiraba de
   la cola ("se pierde") y dejaba de aparecer en la recompensa y en las
   métricas -- eso escondía justo el dato que hace falta para definir una
   garantía de tiempo con fundamento (cuánto se tarda REALMENTE en
   atender a alguien, sin censurar los casos malos). Ahora nadie se va
   nunca: espera hasta ser atendido, o hasta que termina la corrida
   (`env.py` ya no tiene `_purgar_perdidas`, `unidades.py` ya no tiene un
   campo de paciencia).
4. Consecuencia de (3): sin paciencia, `sobrante` puede crecer sin límite
   para alguien que espera muchísimo, y `(sobrante/normalizador)²` también
   -- una sola persona muy retrasada podría dominar toda la recompensa del
   paso. Se le puso un **techo por persona** (`penalizacion_maxima_persona`,
   default 1.0): la penalización de una persona nunca pasa de ese valor,
   así que sigue presionando (mientras más tiempo pasa sin atenderla, más
   cerca del techo) pero no puede explotar sin límite. El techo por defecto
   (1.0) es el mismo valor que antes marcaba "a punto de perderse" -- la
   escala de la recompensa no cambió, solo se dejó de borrar a la gente al
   llegar ahí.
"""
from __future__ import annotations

from estado import Barco, EstadoSimulacion
from unidades import Unidad


def _unidades_activas(estado: EstadoSimulacion) -> list[Unidad]:
    """Todas las unidades que aún no llegaron a destino: esperando en
    cualquiera de las 12 colas, o a bordo de un barco en ruta (su tiempo en
    el sistema sigue contando hasta la entrega real, no se congela al
    embarcar).
    """
    activas = [u for cola in estado.colas.values() for u in cola]
    for b in estado.barcos:
        activas.extend(b.a_bordo)
    return activas


def calcular_recompensa(estado: EstadoSimulacion, cfg_recompensa: dict) -> tuple[float, dict]:
    """Recompensa del paso (negativa) + desglose por término, para poder
    loguear/graficar cada componente por separado durante la verificación.
    """
    tolerancia = cfg_recompensa["tolerancia_incomodidad_min"]
    sobrante_normalizador = cfg_recompensa["sobrante_normalizador_min"]
    techo_persona = cfg_recompensa["penalizacion_maxima_persona"]
    peso_movimiento = cfg_recompensa["peso_movimiento"]

    penalizacion_incomodidad = 0.0
    for u in _unidades_activas(estado):
        tiempo_en_sistema = estado.t_actual_min - u.minuto_llegada
        sobrante = max(0.0, tiempo_en_sistema - tolerancia)
        penalizacion_unidad = min(techo_persona, (sobrante / sobrante_normalizador) ** 2)
        penalizacion_incomodidad += u.tamano * penalizacion_unidad

    barcos_en_movimiento = sum(1 for b in estado.barcos if b.nodo_origen != b.nodo_destino)
    penalizacion_movimiento = peso_movimiento * barcos_en_movimiento

    desglose = {
        "incomodidad": penalizacion_incomodidad,
        "movimiento": penalizacion_movimiento,
    }
    r = -(penalizacion_incomodidad + penalizacion_movimiento)
    return r, desglose
