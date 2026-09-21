"""Formas de recompensa, intercambiables, todas calculadas sobre el MISMO
`EstadoSimulacion` que ya usa `simulador/src/recompensa.py` -- ver
`modelo_rl/prueba_rewards/README.md` para la teoría detrás de cada una.

B y D son la fórmula YA EXISTENTE en `recompensa.calcular_recompensa` --
se importan y se llaman tal cual (con distinto `cfg_recompensa`), nunca se
reimplementan acá. A, C, E, F, G, H son fórmulas nuevas, genuinamente
distintas de esa fórmula base y entre sí.
"""
from __future__ import annotations

from unidades import Unidad
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
    2.A): `r_t = -sum_i tam_i * paso_tiempo_min`, sin tolerancia, sin techo,
    sin cuadrado, **sin movimiento** -- lineal en el tiempo que cada persona
    pasa activa, nada más. `cfg_recompensa`/`entregas_este_paso` quedan en
    la firma solo para que todas las calculadoras tengan la misma forma de
    llamado (acá no se usa ningún campo de `cfg_recompensa`).

    **Corregido 2026-09-17:** esta función incluía hasta ahora un término de
    movimiento (`peso_movimiento * barcos_en_movimiento`, vía
    `cfg_recompensa["peso_movimiento"]`) que NO está en la fórmula
    documentada (README sección 2.A) ni en la que reporta el resumen del
    proyecto -- un desajuste real entre código y documentación, no una
    variante intencional. Se quitó para que A sea, en el código, exactamente
    lo que dice ser: el objetivo puro de tiempo, sin ningún otro término. El
    modelo A se reentrenó con este fix (ver metadata_entrenamiento.json,
    fecha de la corrida) -- no queda una versión "A con movimiento" aparte.
    """
    costo_tiempo = sum(u.tamano * paso_tiempo_min for u in _unidades_activas(estado))
    desglose = {"tiempo_activo": costo_tiempo, "movimiento": 0.0, "entrega": 0.0}
    return -costo_tiempo, desglose


def potencial(estado: EstadoSimulacion, eta: float) -> float:
    """Phi(s) = -eta * N(s), N(s) = suma de `tamano` de toda unidad activa
    (esperando en cola o a bordo, sin entregar aún) -- ver README, sección
    2.C. Negativo: más gente sin entregar, "peor" potencial (más lejos de
    la meta), igual que un `-distancia_a_la_meta` en el ejemplo clásico de
    Ng et al. (1999).
    """
    n_activas = sum(u.tamano for u in _unidades_activas(estado))
    return -eta * n_activas


def suma_tiempo_activo(estado: EstadoSimulacion) -> float:
    """Σ_i tamano_i * (t_actual - minuto_llegada_i), sobre las unidades
    ACTIVAS de `estado` (esperando o a bordo) -- pieza compartida por E, G
    (y el wrapper, que la usa para precomputar `s_tiempo_antes` -- ver
    README sección 2.G, "por qué no guardar el `EstadoSimulacion`
    completo").
    """
    t_actual = estado.t_actual_min
    return sum(u.tamano * (t_actual - u.minuto_llegada) for u in _unidades_activas(estado))


def calcular_recompensa_C(
    estado_despues: EstadoSimulacion, potencial_antes: float,
    cfg_recompensa: dict, gamma: float, entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """B + potential-based shaping (Ng, Harada & Russell 1999) -- README,
    sección 2.C. `r = r_B(estado_despues) + gamma*Phi(estado_despues) -
    Phi(estado_antes)`.

    **`potencial_antes` es un ESCALAR ya calculado, no el `EstadoSimulacion`
    de antes.** `EstadoSimulacion.colas`/`.barcos` (`estado.py`) se pasan
    por REFERENCIA desde `env.py` (`_construir_estado`, `colas=self.colas`),
    así que un `EstadoSimulacion` "viejo" guardado de un paso anterior NO
    queda congelado -- sus colas/barcos son literalmente los mismos objetos
    que el simulador sigue mutando paso a paso, así que leerlo más tarde
    devuelve el estado ACTUAL, no el histórico (confirmado con una prueba
    directa: un snapshot tomado en `reset()` mostró unidades activas
    "aparecidas" 8 pasos después, sin haber cambiado su `t_actual_min`
    guardado). La única forma segura de recordar "cómo estaba el mundo
    antes" es calcular el número que hace falta (acá, `Phi`) EN EL
    MOMENTO en que el estado todavía es fresco, y guardar ESE número (un
    float, inmutable) -- exactamente lo que hace
    `EntornoRecompensaIntercambiable` entre llamadas a `step()`.
    """
    r_b, desglose_b = calcular_recompensa(estado_despues, cfg_recompensa, entregas_este_paso)
    eta = cfg_recompensa["eta_potencial"]
    termino_shaping = gamma * potencial(estado_despues, eta) - potencial_antes

    desglose = dict(desglose_b, shaping=termino_shaping)
    r = r_b + termino_shaping
    return r, desglose


def calcular_recompensa_E(
    estado: EstadoSimulacion, cfg_recompensa: dict, entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """Tiempo lineal puro: `R_t = -sum_i T_{i,t}`, con `T_{i,t}` el tiempo
    TOTAL que la persona `i` lleva en el sistema en este instante (no un
    costo fijo por paso, como A -- ver README, sección 2.E, para la
    diferencia). Sin tolerancia, sin normalizador, sin techo, sin
    movimiento -- ningún término además del tiempo, tal cual se pidió.
    """
    costo_tiempo = suma_tiempo_activo(estado)
    desglose = {"tiempo_lineal": costo_tiempo, "movimiento": 0.0, "entrega": 0.0}
    return -costo_tiempo, desglose


def calcular_recompensa_F(
    estado: EstadoSimulacion, cfg_recompensa: dict, entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """Tiempo cuadrático puro, SIN normalizar: `R_t = -sum_i T_{i,t}^2` --
    a diferencia de D (`(T_i/12)^2`), acá no se divide por ningún
    normalizador. Ver README, sección 2.F, para el problema de escala que
    esto implica y por qué se prueba a propósito.
    """
    t_actual = estado.t_actual_min
    costo_tiempo = sum(u.tamano * (t_actual - u.minuto_llegada) ** 2 for u in _unidades_activas(estado))
    desglose = {"tiempo_cuadratico": costo_tiempo, "movimiento": 0.0, "entrega": 0.0}
    return -costo_tiempo, desglose


def calcular_recompensa_G(
    estado_despues: EstadoSimulacion, s_tiempo_antes: float,
    unidades_entregadas_este_paso: list[Unidad], cfg_recompensa: dict,
    entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """Tiempo incremental: `R_t = -sum_i deltaT_{i,t}`, el AUMENTO de tiempo
    en sistema ocurrido durante este paso (no el total acumulado, como E) --
    ver README, sección 2.G, para la derivación completa. En resumen:

    - Alguien que ya estaba activo y sigue activo (sin ser entregado):
      aporta exactamente `paso_tiempo_min` (2 min) -- estuvo todo el paso.
    - Alguien que llega a mitad del paso: aporta solo la fracción desde que
      llegó (`t_nuevo - minuto_llegada`), no los 2 min completos -- no
      estaba en el sistema antes de llegar.
    - Alguien que es entregado ESTE paso (`unidades_entregadas_este_paso`,
      calculado por `EntornoRecompensaIntercambiable` a partir de
      `env.atendidas_historico`): ya no aparece en `estado_despues` (bajó
      del barco), así que sumar solo sobre unidades activas lo perdería --
      se suma aparte, su tiempo total en el sistema completo.

    Se calcula como `(S_despues - S_antes) + correccion_entregados`, donde
    `S` es `suma_tiempo_activo` -- álgebra exacta (verificada en
    `00_verificar_formulas.ipynb`): la resta ya cuenta bien a quien sigue
    activo y a quien recién llega; falta sumar, aparte, el tiempo de quien
    se fue del conjunto activo por haber sido entregado.

    **`s_tiempo_antes` es un ESCALAR ya calculado (`suma_tiempo_activo` del
    paso anterior), no un `EstadoSimulacion`.** Ver la nota de
    `calcular_recompensa_C` -- mismo motivo exacto: un `EstadoSimulacion`
    guardado de un paso anterior no queda congelado, hay que guardar el
    número ya calculado mientras el estado todavía es fresco.
    """
    t_despues = estado_despues.t_actual_min
    s_despues = suma_tiempo_activo(estado_despues)
    correccion_entregados = sum(u.tamano * (t_despues - u.minuto_llegada) for u in unidades_entregadas_este_paso)

    delta_t_total = (s_despues - s_tiempo_antes) + correccion_entregados
    desglose = {"tiempo_incremental": delta_t_total, "movimiento": 0.0, "entrega": 0.0}
    return -delta_t_total, desglose


def calcular_recompensa_H(
    estado: EstadoSimulacion, cfg_recompensa: dict, entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """Multiobjetivo escalarizado: `R_t = -[w_T*J_T + w_M*J_M]`, con
    `J_T = sum_i T_{i,t}^2` (igual que F, sin normalizar) y `J_M` = número
    de barcos navegando este paso (la MISMA definición de costo de
    movimiento que usan A/B/C/D -- no una métrica nueva). `w_T`/`w_M` vienen
    de `cfg_recompensa["peso_tiempo"]`/`["peso_movimiento"]` (0.95/0.05 en
    el experimento pedido). Ver README, sección 2.H, para el problema de
    escala entre `J_T` y `J_M` (documentado, no oculto ni compensado acá).
    """
    peso_tiempo = cfg_recompensa["peso_tiempo"]
    peso_movimiento = cfg_recompensa["peso_movimiento"]

    t_actual = estado.t_actual_min
    j_tiempo = sum(u.tamano * (t_actual - u.minuto_llegada) ** 2 for u in _unidades_activas(estado))
    j_movimiento = sum(1 for b in estado.barcos if b.nodo_origen != b.nodo_destino)

    desglose = {
        "tiempo_cuadratico": j_tiempo, "movimiento": j_movimiento, "entrega": 0.0,
        "tiempo_cuadratico_ponderado": peso_tiempo * j_tiempo,
        "movimiento_ponderado": peso_movimiento * j_movimiento,
    }
    r = -(peso_tiempo * j_tiempo + peso_movimiento * j_movimiento)
    return r, desglose


def calcular_recompensa_H_normalizada(
    estado: EstadoSimulacion, cfg_recompensa: dict, entregas_este_paso: float = 0.0,
) -> tuple[float, dict]:
    """H con los dos componentes normalizados ANTES de aplicar los pesos --
    ver README, sección 2.H (actualizada 2026-09-17). Mismo `J_T`/`J_M` que
    H (sin normalizar): `J_T = sum_i tam_i*tau_i^2`, `J_M` = barcos
    navegando este paso.

    `r_t = -[w_T * (J_T/J_T_ref) + w_M * (J_M/J_M_ref)]`, con
    `J_T_ref`/`J_M_ref` leídos de `cfg_recompensa["j_tiempo_ref"]`/
    `["j_movimiento_ref"]` -- dos ESCALARES FIJOS, calculados una sola vez
    ANTES de este entrenamiento, corriendo la política BASE (nunca una
    política RL, y nunca la propia H) sobre las 5 semillas de evaluación
    estándar (1001-1005), en escalón 1 (misma instancia/demanda/flota que
    el entrenamiento) -- la MEDIANA por paso de J_T y J_M sobre esos 450
    pasos (no el promedio: J_T tiene varianza enorme entre semillas, una
    sola semilla atípica dominaría un promedio). Ver
    `output/comparacion/referencia_H_normalizada.json` para el detalle del
    cálculo. La referencia es completamente externa a H: no depende de
    cómo entrena ni de qué política resulte -- así se evita normalizar con
    información que todavía no existe cuando arranca el entrenamiento.

    Con w_T=0.95/w_M=0.05 y esta referencia, ambos términos normalizados
    valen ~1 en el caso típico de la política base -- los pesos sí
    representan, ahora, una preferencia relativa comparable entre los dos
    objetivos (no una garantía de que el agente entrenado termine
    repartiendo su comportamiento exactamente 95/5 -- eso depende de qué
    aprenda, no de la fórmula).
    """
    peso_tiempo = cfg_recompensa["peso_tiempo"]
    peso_movimiento = cfg_recompensa["peso_movimiento"]
    j_tiempo_ref = cfg_recompensa["j_tiempo_ref"]
    j_movimiento_ref = cfg_recompensa["j_movimiento_ref"]

    t_actual = estado.t_actual_min
    j_tiempo = sum(u.tamano * (t_actual - u.minuto_llegada) ** 2 for u in _unidades_activas(estado))
    j_movimiento = sum(1 for b in estado.barcos if b.nodo_origen != b.nodo_destino)

    j_tiempo_norm = j_tiempo / j_tiempo_ref
    j_movimiento_norm = j_movimiento / j_movimiento_ref

    desglose = {
        "tiempo_cuadratico": j_tiempo, "movimiento": j_movimiento, "entrega": 0.0,
        "tiempo_cuadratico_normalizado": j_tiempo_norm, "movimiento_normalizado": j_movimiento_norm,
        "tiempo_cuadratico_ponderado": peso_tiempo * j_tiempo_norm,
        "movimiento_ponderado": peso_movimiento * j_movimiento_norm,
    }
    r = -(peso_tiempo * j_tiempo_norm + peso_movimiento * j_movimiento_norm)
    return r, desglose


CALCULADORAS = {
    "A": calcular_recompensa_A, "B": calcular_recompensa_B, "C": calcular_recompensa_C, "D": calcular_recompensa_D,
    "E": calcular_recompensa_E, "F": calcular_recompensa_F, "G": calcular_recompensa_G, "H": calcular_recompensa_H,
    "H_normalizada": calcular_recompensa_H_normalizada,
}
