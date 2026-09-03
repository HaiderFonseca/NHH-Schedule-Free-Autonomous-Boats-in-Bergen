# Bitácora de experimentos -- diagnóstico y arreglo del agente PPO

Trazabilidad de cada cambio aplicado, en el orden pedido (`simulacion/README.md`,
sección de diagnóstico, tiene el resumen narrativo; esta tabla es el detalle
completo). Todas las filas B.1-B.5 corren sobre `escalon_toy` (1 barco, 2 nodos
-- `bryggen`/`kleppesto`, 90 pasos), para iterar rápido; la fila B.7 corre sobre
`escalon_1` (la instancia real de entrenamiento), con el combo validado.

**Punto de partida (antes de este ciclo, ver README sección 13 original):**
100k timesteps sobre `escalon_1`, sin nada de lo de abajo -- 27.6% atendidas
vs. 88.0% de la política base, ocupación 1.15 vs. 3.50, reward -17913.76 vs.
-2444.46 (agregado sobre 5 semillas de evaluación). Política degenerada.

## Experimentos sobre `escalon_toy`

| Experimento | Config | Timesteps | % atendidas agente | % atendidas base | Reward agente | Reward base | Tiempo |
|---|---|---|---|---|---|---|---|
| B.1 Receta original (sin fix) | vecnorm=False, premio=0.0, peso_mov=0.1, ent_coef=0.0 (default SB3) | 20000 | 0.0% | 67.1% | -13218.86 | -3694.02 | -- |
| B.2 + VecNormalize | vecnorm=True, premio=0.0, peso_mov=0.1, ent_coef=0.0 | 20000 | 67.1% | 67.1% | -3521.41 | -3694.02 | 107s |
| B.3 + premio de entrega | vecnorm=True, premio=0.5, peso_mov=0.1, ent_coef=0.0 | 20000 | 67.1% | 67.1% | -3416.41 | -3589.02 | 152s |
| B.4 + peso_movimiento=0 (temporal) | vecnorm=True, premio=0.5, peso_mov=0.0, ent_coef=0.0 | 20000 | 67.1% | 67.1% | -3408.91 | -3581.52 | 141s |
| B.5 + ent_coef=0.01, n_steps=512 | vecnorm=True, premio=0.5, peso_mov=0.0, ent_coef=0.01, n_steps=512 | 20000 | 67.1% | 67.1% | -3408.91 | -3581.52 | 123s |
| **B.7 (combo validado, `escalon_1` real)** | vecnorm=True, premio=0.5, peso_mov=0.0, ent_coef=0.01, n_steps=512 | 150000 | 77.18% | 88.0% | -4807.00 (5 ep.) | -2009.26 (5 ep.) | ~34 min |

**B.1 -> B.2, lectura:** el salto de 0% a 67.1% (empatando exactamente con la
política base, y con MEJOR reward: -3521 vs. -3694) confirma que la causa
principal era la falta de normalización -- sin ella, la red no lograba
aprender nada útil pese a que `check_env` y la mecánica de recompensa ya
estaban verificadas correctas (ver README, hipótesis descartadas). Con
`VecNormalize` sola, el agente YA IGUALA a la política base en esta instancia
de juguete.

**B.2 -> B.3 -> B.4, lectura:** el % atendidas se queda clavado en 67.1% en las
tres filas -- **con 1 solo barco, esta instancia de juguete parece tener un
techo de servicio cerca de 210/313 personas en 90 pasos** (la política base,
ya afinada, tampoco pasa de ahí), así que no hay margen para que el premio de
entrega o quitar la penalización de movimiento muestren una diferencia de
`% atendidas` en ESTA instancia -- sí se ve una mejora chica y consistente en
el reward crudo del agente (-3521 -> -3416 -> -3409) según se van agregando.
Conclusión: estos cambios no perjudican nada en el juguete, y el hallazgo real
(que la normalización era la causa principal) ya quedó confirmado en B.2 --
sigue B.5 (hiperparámetros) y después el combo completo se prueba en
`escalon_1` (B.7), la instancia real donde se detectó el problema original.

**Nota sobre `total_timesteps` de B.7:** el primer intento fue con 300000 (~3x el
original) -- se dejó correr 30 min y no alcanzó a terminar. Calibración directa en
esta máquina: `escalon_1` (2 barcos, 4 nodos) entrena a ~73 steps/seg, así que
300000 timesteps son ~68 min -- demasiado para iterar cómodo. Se bajó a 150000
(~1.5x el original, ~34 min esperados) como compromiso práctico; si el resultado
de B.7 no alcanza a la política base, subir `total_timesteps` de nuevo es la
primera palanca a probar (ya con el combo de fixes validado, no hace falta
repetir B.1-B.6).

**B.7, lectura final:** de 27.6% (punto de partida, sección de arriba) a **77.18%
atendidas** -- una mejora enorme, pero todavía por debajo del 88.0% de la
política base. La curva de entrenamiento durante esta corrida mejoró de forma
consistente (recompensa media: -2439.28 en los primeros 20 episodios, -691.15
en los últimos 20, de 1666 episodios entrenados) -- el agente seguía
mejorando activamente cuando el entrenamiento se detuvo, no se estancó. El
`reward_total` del agente (-4807 sobre 5 episodios) queda por debajo del de
la base (-2009.26) pese al `premio_por_persona_entregada` activo -- consistente
con que todavía deja ~23% de la demanda sin atender (cada persona sin atender
sigue acumulando penalización de incomodidad) y tiene una espera máxima notablemente peor (117.1 min vs. 69.7
min de la base) en al menos un par (`laksevag->sandviken`).

**Bug real encontrado y corregido durante esta última corrida:** la primera
versión de `correr_agente` (con VecNormalize) steppeaba el entorno A TRAVÉS del
`DummyVecEnv` (`venv.step(...)`) -- pero `DummyVecEnv.step_wait()` auto-resetea
el entorno interno en el mismo `step()` en que `done=True` (comportamiento
estándar de VecEnv), lo que borraba `atendidas_historico`/`log_eventos` ANTES
de poder leerlos. `verificar_conservacion` lo detectó de inmediato (todo en 0
salvo `generadas`) -- se corrigió steppeando el entorno CRUDO directamente
(mismo patrón ya usado y verificado en B.2-B.5) y usando `VecNormalize` solo
para normalizar la observación antes de `model.predict()`. Los números de
arriba son de la corrida ya corregida.

**Conclusión de esta fase de diagnóstico:** la meta era que el agente al menos
IGUALE a la política base -- con 150k timesteps se llegó a 77.18% (vs. 88.0%),
una mejora enorme sobre el punto de partida (27.6%) pero todavía no en
paridad. La curva de entrenamiento no se había aplanado -- **la palanca más
directa para la siguiente iteración es subir `total_timesteps`** (ya con el
combo de fixes validado, sin necesidad de repetir B.1-B.6). Superar a la
política base queda, como se esperaba desde el principio, para un siguiente
escalón del proyecto.
