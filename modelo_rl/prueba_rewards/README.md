# Prueba de recompensas -- cuatro formulaciones, comparadas sobre métricas reales

**Qué es esto, en una frase:** cuatro funciones de recompensa conceptualmente distintas (no cuatro pesos de la misma fórmula), cada una entrenada con PPO en las mismas condiciones, comparadas SOLO por lo que el simulador mide de verdad (tiempo en sistema, movimientos), nunca por el valor de cada recompensa propia.

Este experimento es autocontenido: no modifica `simulador/`, `politica_base/`, ni los notebooks/README ya cerrados de `modelo_rl/`. Reusa el motor (`simulador/src/`), la política base (`politica_base/src/politica_base.py`), el wrapper de demanda fresca (`modelo_rl/src/entrenamiento.py`) y las métricas/gráficas (`simulador/src/metricas.py`, `visualizacion.py`) tal cual -- nada de eso se reimplementa.

---

## 1. Cómo está organizado

```
modelo_rl/prueba_rewards/
├── README.md                                  # este archivo
├── config/instance.yaml                       # definición de A/B/C/D, timesteps, semilla
├── src/
│   ├── recompensas_alternativas.py            # calcular_recompensa_A, potencial(), calcular_recompensa_C
│   │                                            # (B y D: la función YA EXISTENTE de recompensa.py, reusada)
│   └── entorno_recompensa_intercambiable.py    # EntornoRecompensaIntercambiable -- enchufa cualquiera de las 4
├── notebooks/
│   ├── 00_verificar_formulas.ipynb             # Paso 1: las 4 fórmulas, un paso de ejemplo, sanity checks
│   ├── 01_entrenar_cuatro.ipynb                # Paso 2: 4 entrenamientos PPO cortos, mismos hiperparámetros
│   └── 02_comparar_resultados.ipynb            # Paso 3: evaluación contra base, métricas reales, gráficas
└── output/
    ├── modelos/{A,B,C,D}/                      # modelo_ppo.zip, vecnormalize.pkl, monitor.monitor.csv
    └── comparacion/                            # tablas y gráficas finales
```

---

## 2. Las cuatro recompensas

Las cuatro se calculan sobre el mismo `EstadoSimulacion` que ya construye `env.py` en cada paso -- ninguna reimplementa la mecánica de simulación, solo cambia cómo se traduce ese estado a un número.

### 2.A -- Objetivo puro, derivado de la identidad de Little

$$r_t = -\left[\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot \Delta t \;+\; w_{\text{move}}^A \cdot m_t\right], \qquad w_{\text{move}}^A \approx 0.003 \cdot \Delta t = 0.006$$

Sin tolerancia, sin techo, sin cuadrado: cada persona activa (esperando o a bordo, sin entregar aún) cuesta, por paso, su tamaño multiplicado por la duración del paso (Δt = 2 min).

**Por qué esto es "el objetivo real", no un proxy.** La identidad detrás de la Ley de Little (L = λW) es un argumento de conteo doble: el área bajo la curva de "cuánta gente hay en el sistema en cada instante", integrada en el tiempo, es exactamente igual a la suma de los tiempos que cada persona individual pasó en el sistema (cada persona "aporta" su propio tiempo en sistema al área, sin importar cuándo entró ni salió). En su versión discreta:

$$\sum_t N(t) \cdot \Delta t \;=\; \sum_{i} \text{tamaño}_i \cdot W_i$$

donde $N(t)$ es la gente activa en el paso $t$ y $W_i$ el tiempo que la persona $i$ pasó en el sistema. El término de la izquierda es, salvo el signo, exactamente lo que acumula la recompensa A a lo largo de un episodio. Maximizar la recompensa A acumulada es, por esta identidad, minimizar directamente $\sum_i \text{tamaño}_i \cdot W_i$ -- el tiempo total en sistema, la métrica que de verdad importa (`espera_media_min`/`sistema_medio_min` en `metricas.py`) -- sin pasar por ninguna tolerancia ni curva de castigo intermedia.

### 2.B -- Ponderada convexa con tolerancia, sin techo

La fórmula ya usada en el resto de `modelo_rl/` (ganadora de la secuencia A/B/C1/C2 ya cerrada -- ver `modelo_rl/README.md` sección 3), reusada tal cual vía `recompensa.calcular_recompensa`:

$$r_t = -\left[\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot \left(\frac{\max(0, s_{i,t}-12)}{18}\right)^2 \;+\; 0.003 \cdot m_t\right]$$

Sirve de referencia conocida: penaliza solo el exceso sobre la tolerancia (12 min), de forma cuadrática (crece más rápido cuanto más se tarda), sin techo por persona.

### 2.C -- B + *potential-based reward shaping* (Ng, Harada & Russell 1999)

$$r_t = r_t^B + \underbrace{\big(\gamma \cdot \Phi(s_{t+1}) - \Phi(s_t)\big)}_{\text{término de shaping}}, \qquad \Phi(s) = -\eta \cdot N(s)$$

con $N(s)$ = personas activas (ponderadas por tamaño) en el estado $s$, y $\eta$ (`eta_potencial`) configurable.

**Por qué esta forma preserva la política óptima -- el teorema de Ng, Harada & Russell (1999).** Dado un MDP $M$ con recompensa $R$, y un MDP modificado $M'$ con recompensa $R' = R + F$, los autores prueban que $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$ para cualquier función potencial $\Phi: S \to \mathbb{R}$ es, bajo condiciones leves, **necesaria y suficiente** para garantizar que toda política óptima de $M'$ sea también óptima en $M$ (y viceversa) -- para CUALQUIER recompensa base $R$, no solo la de este proyecto. La prueba muestra que $Q^*_{M'}(s,a) = Q^*_M(s,a) - \Phi(s)$: el shaping desplaza el valor de TODAS las acciones en un mismo estado por la MISMA constante $\Phi(s)$ (no depende de $a$), así que el orden entre acciones -- y por lo tanto la acción óptima -- no cambia en ningún estado.

**Qué se espera que aporte, entonces, si no cambia el óptimo.** Velocidad de aprendizaje, no el destino. Sin shaping, la única señal viene del castigo por incomodidad, que se acumula lento y de forma dispersa (recién se nota bastante después de que alguien lleva un rato esperando). Con shaping, cada paso que reduce la cola da un premio inmediato ($\Phi$ sube, menos negativo), y cada paso que la deja crecer da un castigo inmediato -- señal densa, correlacionada con el progreso, disponible desde el primer paso. La garantía teórica es sobre el óptimo con entrenamiento infinito; con el presupuesto corto de este experimento (30 000 timesteps, igual que las otras tres), la pregunta empírica es si esa señal más densa efectivamente ayuda a converger más rápido -- exactamente lo que este experimento mide.

### 2.D -- Actual del código (parámetros de producción)

$$r_t = -\left[\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot \min\!\left(1.0, \left(\frac{\max(0,s_{i,t}-12)}{18}\right)^2\right) \;+\; 0.1 \cdot m_t\right]$$

Exactamente `calcular_recompensa()` con los valores que hoy viven en `simulador/config/instance.yaml` → `recompensa:` (techo=1.0, `peso_movimiento`=0.1) -- los que gobiernan la política base y los escalones 1-3. **Coincidencia real, no fabricada:** la secuencia de experimentos ya cerrada (A/B/C1/C2, en `modelo_rl/notebooks/02_secuencia_experimentos_reward.ipynb`) nunca usó `peso_movimiento=0.1` ni dejó el techo activo -- esos valores de producción nunca se habían entrenado con RL hasta este experimento. Sirve de contraste "mal escalado" (techo bajo + movimiento caro) sin necesidad de inventar una quinta variante.

---

## 3. Entrenamiento -- mismas condiciones para las 4

| | Valor | De dónde sale |
|---|---|---|
| Instancia | `escalon_1` (2 barcos, franja mañana) | `simulador/config/instance.yaml` |
| `total_timesteps` | 30 000 (corto, para iterar) | `prueba_rewards/config/instance.yaml` |
| `semilla_entrenamiento` | 123 | `prueba_rewards/config/instance.yaml` |
| `VecNormalize` (obs y reward) | activo | `simulador/config/instance.yaml` → `agente.hiperparametros` |
| `ent_coef` | 0.01 | ídem |
| `learning_rate` | 0.0003 | ídem |
| `n_steps` | 512 | ídem (el validado -- ver `modelo_rl/README.md` sección 4.1) |

Lo ÚNICO que cambia entre las 4 corridas es la recompensa (`EntornoRecompensaIntercambiable`, `tipo_recompensa="A"|"B"|"C"|"D"`).

**Tiempo real medido, por tipo** (cada `output/modelos/{A,B,C,D}/metadata_entrenamiento.json` lo guarda aparte, con `time.time()` alrededor de `model.learn(...)` -- dato fijo, no una estimación):

| Tipo | Actualizaciones de PPO (30 000/512) | Tiempo medido |
|---|---|---|
| A | 58 | 2.6 min (153 s) |
| B | 58 | 2.1 min (127 s) |
| C | 58 | 3.1 min (187 s) |
| D | 58 | 3.8 min (227 s) |

El tiempo de reloj varía un poco entre corridas (carga de la máquina en el momento, no la recompensa en sí -- se confirmó reentrenando C aislado, sin nada más corriendo: tardó lo mismo del mismo orden que aquí). Lo que sí es fijo entre las 4: **58 actualizaciones de PPO cada una** -- una quinta parte de las 292 que hace el agente final (`modelo_rl/README.md` sección 4.2) -- por eso ninguna de las 4 se acerca a la política base en la sección 5: no es un problema de la fórmula de recompensa, es presupuesto de entrenamiento corto a propósito (esto es una comparación de FORMA de recompensa, no un intento de agente competitivo).

---

## 4. Validación anti-Goodhart

Las 4 recompensas tienen escalas distintas por diseño (A suma minutos crudos, B/C usan una curva cuadrática acotada entre 0 y ~7, D la misma curva con techo en 1.0) -- comparar sus valores absolutos entre sí no significa nada, y hacerlo sería exactamente el error que a veces se conoce como "reward hacking"/Ley de Goodhart aplicada mal (optimizar una métrica proxy hasta que deja de reflejar el objetivo real). Por eso la comparación del Paso 3 usa ÚNICAMENTE las métricas reales del simulador, reusando `metricas.reporte_completo` sin tocarlo:

- **Tiempo en sistema, medio y máximo** -- lo que las cuatro recompensas, en teoría, intentan reducir.
- **Espera media.**
- **Movimientos totales** -- el costo que se intenta no pagar de más.
- **Ocupación media.**
- **% atendidas** -- de referencia únicamente, nunca como criterio principal: el simulador no pierde a nadie (nadie se retira nunca), así que ese número depende en parte de dónde corta la ventana de evaluación, no solo de la política (mismo argumento que en `comparacion/README.md`).

---

## 5. Resultados

30 000 timesteps por tipo, 5 semillas de evaluación (1001-1005), conservación OK en las 5 corridas.

| Tipo | Tiempo sistema medio (min) | Tiempo sistema máximo (min) | Espera media (min) | Movimientos | Ocupación media | % atendidas (referencia) |
|---|---|---|---|---|---|---|
| A | 61.69 | 131.93 | 39.66 | 125 | 1.87 | 47.2% |
| B | 38.13 | 139.93 | 23.83 | 127 | 2.04 | 52.0% |
| C | 46.43 | **104.75** | 38.16 | **72** | 1.98 | 49.5% |
| D | **30.57** | 123.69 | **19.15** | 124 | 1.64 | 40.2% |
| base | 26.05 | 69.74 | 16.42 | 155 | 3.50 | 88.0% |

**Ninguna de las 4 se acerca a la base** -- esperado, 30 000 timesteps es una quinta parte del presupuesto que usa el agente final ya entrenado (`modelo_rl/notebooks/01`, 150 000 pasos) y acá el objetivo es comparar la FORMA de la recompensa entre sí, no producir un agente competitivo.

**Entre A/B/C/D, el ranking por suma de rangos (medio + máximo + movimientos) da un empate exacto entre C y D (5 puntos cada una, contra 10 de A y B).** No es una elección metodológica: `pandas.sort_values` rompe el empate por su propio criterio interno de ordenamiento, no por ningún argumento real -- así que en vez de forzar un único "ganador", se documentan los dos, porque representan compromisos genuinamente distintos:

- **D gana en el caso típico** (tiempo medio 30.57 min, espera media 19.15 min -- los mejores de los cuatro).
- **C gana en el peor caso y en eficiencia** (máximo 104.75 min, muy por debajo de los otros tres que superan 120 min; y 72 movimientos, contra 124-127 de las otras tres -- casi la mitad).

Gráficas: curvas de entrenamiento de las 4 (`output/comparacion/curvas_entrenamiento.html`), comparación de métricas en barras (`comparacion_metricas.html`), y detalle completo (heatmap por par + perfil de espera + ocupación de flota) para **las dos empatadas**, C y D (`heatmap_por_par_{C,D}.html`, `perfil_espera_{C,D}.html`, `ocupacion_flota_{C,D}.html`).

---

## 6. Conclusión

**Ningún resultado contradice la teoría -- pero ninguno confirma la lectura ingenua de "A es el objetivo real, así que debería ganar".** Los cuatro dan una lectura coherente, leída junto con lo que cada fórmula realmente premia:

**D (la de producción) no colapsó -- y eso, en sí, es un resultado.** La narrativa original de este proyecto (`modelo_rl/README.md`, diagnóstico histórico) es que estos mismos parámetros (techo bajo, `peso_movimiento` alto) producían una política degenerada, casi paralizada. Acá no pasó: D entrenó con `VecNormalize` activo (obligatorio para que la comparación entre las 4 fuera justa), y `VecNormalize` es precisamente el fix que en aquel diagnóstico corrigió el colapso. Con la normalización puesta, el desbalance crudo entre los pesos de incomodidad y movimiento pesa mucho menos de lo que pesaba sin ella -- D no solo no colapsa, termina con el mejor tiempo medio y la mejor espera media de las cuatro. Esto no invalida el diagnóstico original (el colapso fue real, en esas condiciones) -- lo que muestra es que la causa de fondo era la falta de normalización, no el peso en sí, algo que este experimento deja mucho más claro que antes.

**A (el objetivo "verdadero" por la identidad de Little) fue el peor de los cuatro en tiempo medio -- exactamente lo que en el límite debería minimizar mejor.** La explicación no contradice la teoría, la completa: la identidad de Little garantiza que minimizar A es minimizar el tiempo total en sistema *en el límite*, con entrenamiento suficiente -- no dice nada sobre qué tan fácil es aprender esa señal con un presupuesto corto. A cobra lo mismo por cada minuto que cualquier persona está activa, sin importar si ya está pasada de tolerancia o recién llegó -- no hay ninguna zona "barata" cerca del límite de tolerancia que el agente pueda explotar rápido, a diferencia de B/C/D, donde alguien servido dentro de los 12 minutos no cuesta nada. Con solo 30 000 pasos, esa falta de estructura aparentemente pesa más que la ventaja teórica de estar optimizando el objetivo correcto -- una distinción real entre "la función objetivo correcta" y "la función objetivo fácil de aprender rápido", y una razón concreta para no asumir que la formulación más directa es automáticamente la mejor elección práctica.

**C (B + shaping potencial) cumple exactamente lo que promete el teorema de Ng et al.: no cambia qué es óptimo, ayuda a encontrarlo mejor con presupuesto corto -- y acá se ve en el peor caso y en la eficiencia, no en el promedio.** El shaping le da al agente una señal inmediata por reducir la cola, en cada paso, en vez de solo el castigo tardío de la incomodidad -- con 30 000 pasos (un quinto del presupuesto final), esa señal densa parece haber ayudado a la política a encontrar un patrón de despacho más deliberado: bastantes menos movimientos (72, casi la mitad que A/B/D) sin sacrificar el peor caso (104.75 min, el mejor de los cuatro) -- protege exactamente lo que un techo *sin* shaping (como B) no logra proteger (B tiene el peor máximo de los cuatro, 139.93 min). Es consistente con la garantía teórica: la política óptima de C es la misma que la de B (mismo `r_B` de fondo), así que cualquier diferencia observada es una diferencia de qué tan rápido/bien se llega a una buena política con el mismo presupuesto de entrenamiento, no una diferencia de qué política es "correcta".

**Lectura general:** la elección de recompensa importa, y no siempre en la dirección que la intuición teórica sugeriría a primera vista -- exactamente la razón por la que este proyecto compara sobre métricas reales y no sobre el valor de cada recompensa (sección 4). Si el objetivo de un despliegue real fuera minimizar el caso típico, D es la elección defendible con esta evidencia; si el objetivo es proteger el peor caso y ser eficiente en movimientos, C lo es -- ninguna de las dos lecturas es "la respuesta", son dos objetivos de servicio legítimamente distintos.

---

## 7. Bibliografía

- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Little, J. D. C. -- la identidad L=λW. Notas de referencia: Columbia University, IEOR 4404, *Little's Law*. http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-LL.pdf ; MathWorld, *Little's Law*. https://mathworld.wolfram.com/LittlesLaw.html
- AdaPool (2021). *Adaptive Fleet Rebalancing via Reinforcement Learning.* arXiv:2104.00203. https://arxiv.org/abs/2104.00203
- RAST-MoE-RL (2025). arXiv:2512.13727 (reward hacking / anti-gaming en RL). https://arxiv.org/abs/2512.13727
- *Multi-Objective Rebalancing for Vehicle Sharing Systems* (2020). arXiv:2007.06801. https://arxiv.org/abs/2007.06801
- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd ed.). http://incompleteideas.net/book/the-book-2nd.html
