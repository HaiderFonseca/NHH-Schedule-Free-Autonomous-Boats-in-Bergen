# Prueba de recompensas -- ocho formulaciones, comparadas sobre métricas reales

**Qué es esto, en una frase:** ocho funciones de recompensa conceptualmente distintas (no ocho pesos de la misma fórmula), cada una entrenada con PPO en las mismas condiciones, comparadas SOLO por lo que el simulador mide de verdad (tiempo en sistema, movimientos), nunca por el valor de cada recompensa propia.

**A-D** son la primera secuencia (ya cerrada -- objetivo de Little, cuadrática con tolerancia, esa misma + potential shaping, y los parámetros de producción tal cual). **E-H** son cuatro variantes nuevas, agregadas después sin tocar A-D: tiempo lineal puro, tiempo cuadrático puro (sin normalizar), tiempo incremental por paso, y un multiobjetivo escalarizado 95%/5% (tiempo cuadrático + movimiento) -- ver sección 2 para las fórmulas completas y sección 7 para un hallazgo importante de correctitud descubierto al implementarlas (afecta a C también, ya entrenada).

Este experimento es autocontenido: no modifica `simulador/`, `politica_base/`, ni los notebooks/README ya cerrados de `modelo_rl/`. Reusa el motor (`simulador/src/`), la política base (`politica_base/src/politica_base.py`), el wrapper de demanda fresca (`modelo_rl/src/entrenamiento.py`) y las métricas/gráficas (`simulador/src/metricas.py`, `visualizacion.py`) tal cual -- nada de eso se reimplementa.

---

## 1. Cómo está organizado

```
modelo_rl/prueba_rewards/
├── README.md                                  # este archivo
├── config/instance.yaml                       # definición de A-H, timesteps, semilla
├── src/
│   ├── recompensas_alternativas.py            # calcular_recompensa_{A,C,E,F,G,H}, potencial(), suma_tiempo_activo()
│   │                                            # (B y D: la función YA EXISTENTE de recompensa.py, reusada)
│   └── entorno_recompensa_intercambiable.py    # EntornoRecompensaIntercambiable -- enchufa cualquiera de las 8
├── notebooks/
│   ├── 00_verificar_formulas.ipynb             # Paso 1: las 8 fórmulas, un paso de ejemplo, sanity checks
│   ├── 01_entrenar_cuatro.ipynb                # Paso 2: 8 entrenamientos PPO, mismos hiperparámetros
│   └── 02_comparar_resultados.ipynb            # Paso 3: evaluación contra base, métricas reales, gráficas
└── output/
    ├── modelos/{A,B,C,D,E,F,G,H}/              # modelo_ppo.zip, vecnormalize.pkl, monitor.monitor.csv
    └── comparacion/                            # tablas y gráficas finales
```

---

## 2. Las ocho recompensas

Las ocho se calculan sobre el mismo `EstadoSimulacion` que ya construye `env.py` en cada paso -- ninguna reimplementa la mecánica de simulación, solo cambia cómo se traduce ese estado a un número.

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

**Nota (2026-09-17):** la config de D fue actualizada por decisión del usuario a una formulación de tiempo puramente cuadrática y sin tolerancia (`tolerancia_incomodidad_min=0`, `sobrante_normalizador_min=12`, techo efectivamente infinito, `peso_movimiento=0`) -- $r_t = -\sum_i \text{tamaño}_i \cdot (s_{i,t}/12)^2$, sin término de movimiento. D fue reentrenada con esta config a 150 000 timesteps (frente a los 30 000 originales) -- los resultados de la sección 5 reflejan esta versión, no la descrita arriba en la fórmula matemática (que documenta la D *original* de este experimento, con techo=1.0 y `peso_movimiento`=0.1). Ver `config/instance.yaml` para los valores vigentes.

### 2.E -- Tiempo lineal puro (sin normalizar, sin movimiento)

$$r_t = -\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot T_{i,t}, \qquad T_{i,t} = t - \text{minuto\_llegada}_i$$

El tiempo TOTAL que cada persona activa lleva en el sistema en este instante (no un costo fijo por paso, como A -- A cobra `Δt` por persona activa cada paso; E cobra el tiempo acumulado completo, que crece con cada paso que esa persona sigue activa). Sin tolerancia, sin normalizador, sin techo, sin término de movimiento -- ningún parámetro además del tiempo mismo, a propósito, para aislar el efecto de la forma funcional (lineal vs. cuadrática) sin que ningún otro término la contamine.

### 2.F -- Tiempo cuadrático puro (sin normalizar, sin movimiento)

$$r_t = -\sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot T_{i,t}^2$$

Igual que D pero sin dividir por ningún normalizador (D usa $(T_i/12)^2$) -- versión cruda, para aislar si normalizar importa una vez que `VecNormalize` ya está activo en PPO (que normaliza la señal de recompensa completa, con sus propias estadísticas corridas). Sin tolerancia, sin techo, sin movimiento. Motivación: evaluar si penalizar de forma creciente los tiempos largos (cuadrático > lineal para $T>1$) mejora el comportamiento del agente frente a E.

### 2.G -- Tiempo incremental (delta T por paso)

$$r_t = -\sum_{i} \Delta T_{i,t}$$

El AUMENTO de tiempo en sistema ocurrido durante este paso de 2 min -- no el tiempo acumulado total (a diferencia de E). Se calcula como:

$$\Delta T_{\text{total}} = \big(S(s_{t+1}) - S(s_t)\big) + \sum_{i \in D_t} \text{tamaño}_i \cdot (t_{t+1} - \text{minuto\_llegada}_i), \qquad S(s) = \sum_{i \in \mathcal{A}(s)} \text{tamaño}_i \cdot (t(s) - \text{minuto\_llegada}_i)$$

donde $D_t$ son las unidades entregadas EN este paso (ya no están en $\mathcal{A}(s_{t+1})$, así que la resta $S(s_{t+1})-S(s_t)$ las pierde -- el segundo término las repone). Esta fórmula maneja correctamente los tres casos que puede haber en un paso:

- **Sigue activa, sin ser entregada:** su contribución es exactamente $\Delta t$ (2 min) -- ya estaba activa antes y sigue activa después, todo el paso cuenta.
- **Llega a mitad del paso:** su contribución es solo la fracción desde que llegó ($t_{t+1} - \text{minuto\_llegada}_i < \Delta t$) -- no estaba en el sistema antes de llegar, así que $S(s_t)$ no la incluye y $S(s_{t+1})$ sí, por su tiempo parcial.
- **Es entregada este paso:** su contribución es $t_{t+1} - \max(t_t, \text{minuto\_llegada}_i)$ -- solo el tiempo de ESTE paso, no su historia completa (el álgebra de arriba lo reduce exactamente a eso; ver verificación en `00_verificar_formulas.ipynb`).

Sin doble conteo ni fugas: la suma de $\Delta T_{\text{total}}$ a lo largo de un episodio completo es idéntica, bit a bit, al tiempo total en sistema de cada persona (atendida o activa al cierre) -- verificado exactamente en `00_verificar_formulas.ipynb` (diferencia 0.000000000 sobre un episodio de 90 pasos, semilla 1001).

### 2.H -- Multiobjetivo escalarizado, 95% tiempo cuadrático + 5% movimiento

$$r_t = -\big[w_T \cdot J_T + w_M \cdot J_M\big], \qquad w_T = 0.95,\ w_M = 0.05$$

$$J_T = \sum_{i \in \mathcal{A}_t} \text{tamaño}_i \cdot T_{i,t}^2 \ (\text{igual que F, sin normalizar}), \qquad J_M = m_t \ (\text{barcos navegando este paso, misma definición que A/B/C/D})$$

Combinación ponderada explícita de dos objetivos (tiempo y movimiento), pensada para pedirle al agente que priorice tiempo sobre movimiento en una proporción nominal 95/5. **Ver sección 8 para el problema de escala real, medido, entre $J_T$ y $J_M$ con estos pesos -- documentado, no corregido a mano.**

---

## 3. Entrenamiento -- mismas condiciones para las 8

| | Valor | De dónde sale |
|---|---|---|
| Instancia | `escalon_1` (2 barcos, franja mañana) | `simulador/config/instance.yaml` |
| `total_timesteps` | 150 000 | `prueba_rewards/config/instance.yaml` |
| `semilla_entrenamiento` | 123 | `prueba_rewards/config/instance.yaml` |
| `VecNormalize` (obs y reward) | activo | `simulador/config/instance.yaml` → `agente.hiperparametros` |
| `ent_coef` | 0.01 | ídem |
| `learning_rate` | 0.0003 | ídem |
| `n_steps` | 512 | ídem (el validado -- ver `modelo_rl/README.md` sección 4.1) |

Lo ÚNICO que cambia entre las 8 corridas es la recompensa (`EntornoRecompensaIntercambiable`, `tipo_recompensa="A".."H"`). **Nota histórica:** `total_timesteps` era originalmente 30 000 (para iterar rápido); se subió a 150 000 (292 actualizaciones de PPO, igual que el agente final -- `modelo_rl/README.md` sección 4.2) antes de agregar E-H, y A/B/C/D fueron reentrenadas a este presupuesto más largo -- los números de la sección 5 son de esa corrida, no de la de 30 000 (que ya no está documentada, para no confundir con la actual).

**Tiempo real medido, por tipo** (cada `output/modelos/{A..H}/metadata_entrenamiento.json` lo guarda aparte, con `time.time()` alrededor de `model.learn(...)` -- dato fijo, no una estimación):

| Tipo | Actualizaciones de PPO (150 000/512) | Tiempo medido |
|---|---|---|
| A | 292 | 14.7 min |
| B | 292 | 14.9 min |
| C | 292 | 12.9 min |
| D | 292 | 15.3 min |
| E | 292 | 9.3 min |
| F | 292 | 8.9 min |
| G | 292 | 9.2 min |
| H | 292 | 9.2 min |

E-H entrenan notablemente más rápido que A-D con el mismo número de actualizaciones -- consistente con que ninguna de las 4 nuevas tiene el término de shaping/potencial ni la lógica de umbral/techo de B/C/D (menos trabajo por paso en el cálculo de la recompensa, no en el simulador en sí, que es igual para las 8).

---

## 4. Validación anti-Goodhart

Las 8 recompensas tienen escalas distintas por diseño (A/E suman minutos crudos, B/C/D/F/H usan curvas cuadráticas -- algunas normalizadas y acotadas, otras no, G suma incrementos de tiempo por paso) -- comparar sus valores absolutos entre sí no significa nada, y hacerlo sería exactamente el error que a veces se conoce como "reward hacking"/Ley de Goodhart aplicada mal (optimizar una métrica proxy hasta que deja de reflejar el objetivo real). Por eso la comparación del Paso 3 usa ÚNICAMENTE las métricas reales del simulador, reusando `metricas.reporte_completo` sin tocarlo:

- **Tiempo en sistema, medio, máximo y percentiles (p50/p90/p95)** -- lo que las ocho recompensas, en teoría, intentan reducir.
- **Espera media.**
- **Movimientos totales** -- el costo que se intenta no pagar de más.
- **Ocupación media.**
- **% atendidas** -- de referencia únicamente, nunca como criterio principal: el simulador no pierde a nadie (nadie se retira nunca), así que ese número depende en parte de dónde corta la ventana de evaluación, no solo de la política (mismo argumento que en `comparacion/README.md`).

---

## 5. Resultados

150 000 timesteps por tipo, 5 semillas de evaluación (1001-1005), conservación OK en las 8 corridas + base.

**Nota (2026-09-17):** esta tabla reemplaza una versión anterior calculada a 30 000 timesteps (presupuesto exploratorio inicial) -- A/B/C/D fueron reentrenadas a 150 000 antes de agregar E-H, así que estos números de A-D ya no coinciden con los de la primera versión de este README. La lectura de la sección 6 (Conclusión) fue escrita sobre la corrida de 30 000 y **no se reescribió** para esta corrida más larga -- sus afirmaciones cualitativas (p.ej. "A es el peor en tiempo medio", "D no colapsó") siguen siendo ciertas con los números de abajo, pero sus valores numéricos exactos citados en el texto son los viejos. Se deja así (no se pidió rehacer esa sección) y se documenta la discrepancia en vez de dejarla implícita.

| Tipo | Sistema medio (min) | Sistema máx (min) | Sistema p95 (min) | Espera media (min) | Espera p95 (min) | Movimientos | Ocupación media | % atendidas (ref.) |
|---|---|---|---|---|---|---|---|---|
| A | 37.17 | 122.07 | 87.95 | 22.45 | 59.33 | 164 | 3.00 | 78.1% |
| B | 29.02 | 60.77 | 54.62 | 17.06 | 37.46 | 191 | 3.50 | 88.0% |
| C | 29.25 | 74.07 | 56.77 | 20.21 | 44.77 | 193 | 3.39 | 84.5% |
| D | **25.73** | **51.92** | **51.10** | **16.21** | **41.34** | 174 | 3.21 | 84.5% |
| E | 41.06 | 123.13 | 115.10 | 24.41 | 62.55 | 172 | 2.85 | 74.8% |
| F | 33.05 | 81.13 | 69.46 | 22.78 | 57.46 | 183 | 3.00 | 75.9% |
| G | 33.46 | 109.74 | 74.55 | 20.72 | 61.71 | 169 | 3.02 | 74.4% |
| H | 37.15 | 103.74 | 76.55 | 24.94 | 64.43 | **194** | 3.20 | 81.4% |
| base | 26.05 | 69.74 | 58.77 | 16.42 | 46.06 | 155 | 3.50 | 88.0% |

Tabla completa con más percentiles (p50/p90/p95 de sistema y espera): `output/comparacion/tabla_comparativa.csv`.

**Entre A/B/C/D (análisis ya cerrado, sección 6), el ranking por suma de rangos (medio + máximo + movimientos) sigue dando D como ganador** con esta corrida más larga -- ver `02_comparar_resultados.ipynb`, celda "¿Cuál ganó?".

**E-H, sin conclusión de cuál es mejor (pedido explícitamente así):** con esta única corrida de 150 000 timesteps y 5 semillas de evaluación, no se declara una recompensa ganadora entre las 8 -- los números de arriba quedan para un análisis conjunto posterior. Algunas observaciones puntuales, sin ser un ranking:

- **G (tiempo incremental) tiene el tiempo medio y p50 más bajos entre E-H** (33.46 min medio, 25.18 min p50 -- muy cerca del p50 de D, 23.36), pero un p95 bastante más alto que su propia mediana (74.55 min) -- sugiere una distribución con cola larga: la mayoría de los pasajeros se atienden rápido, pero un subconjunto espera mucho más.
- **E (tiempo lineal puro) tiene el peor tiempo máximo y p95 del grupo completo** (123.13 min máx, 115.10 min p95 -- comparable al peor caso de A). Consistente con la misma falta de estructura de castigo que ya se documentó para A en la sección 6: sin ninguna curva que penalice más fuerte a quien ya lleva mucho esperando, no hay incentivo adicional para priorizar los casos más atrasados.
- **F (tiempo cuadrático puro) mejora sobre E en casi todas las métricas** (33.05 vs 41.06 medio, 81.13 vs 123.13 máximo) -- consistente con la motivación de probar si castigar de forma creciente los tiempos largos ayuda frente a un castigo lineal.
- **H (multiobjetivo 95/5) se comporta de forma casi indistinguible de F** (37.15 vs 33.05 medio, 103.74 vs 81.13 máximo -- mismo orden de magnitud, ninguna mejora clara en movimientos pese al 5% de peso nominal ahí puesto: 194 movimientos, el valor más alto de las 8). Consistente con el hallazgo de la sección 8: con estos pesos, el término de movimiento es aritméticamente insignificante frente al de tiempo, así que H optimiza, en la práctica, casi lo mismo que F solo.

Gráficas: curvas de entrenamiento de las 8 (`output/comparacion/curvas_entrenamiento.html`), comparación de métricas A-D+base (`comparacion_metricas.html`), comparación completa A-H+base con percentiles (`comparacion_metricas_completa.html`), y detalle completo (heatmap por par + perfil de espera + ocupación de flota) para las dos empatadas de A-D, C y D (`heatmap_por_par_{C,D}.html`, `perfil_espera_{C,D}.html`, `ocupacion_flota_{C,D}.html`).

---

## 6. Conclusión

**Ningún resultado contradice la teoría -- pero ninguno confirma la lectura ingenua de "A es el objetivo real, así que debería ganar".** Los cuatro dan una lectura coherente, leída junto con lo que cada fórmula realmente premia:

**D (la de producción) no colapsó -- y eso, en sí, es un resultado.** La narrativa original de este proyecto (`modelo_rl/README.md`, diagnóstico histórico) es que estos mismos parámetros (techo bajo, `peso_movimiento` alto) producían una política degenerada, casi paralizada. Acá no pasó: D entrenó con `VecNormalize` activo (obligatorio para que la comparación entre las 4 fuera justa), y `VecNormalize` es precisamente el fix que en aquel diagnóstico corrigió el colapso. Con la normalización puesta, el desbalance crudo entre los pesos de incomodidad y movimiento pesa mucho menos de lo que pesaba sin ella -- D no solo no colapsa, termina con el mejor tiempo medio y la mejor espera media de las cuatro. Esto no invalida el diagnóstico original (el colapso fue real, en esas condiciones) -- lo que muestra es que la causa de fondo era la falta de normalización, no el peso en sí, algo que este experimento deja mucho más claro que antes.

**A (el objetivo "verdadero" por la identidad de Little) fue el peor de los cuatro en tiempo medio -- exactamente lo que en el límite debería minimizar mejor.** La explicación no contradice la teoría, la completa: la identidad de Little garantiza que minimizar A es minimizar el tiempo total en sistema *en el límite*, con entrenamiento suficiente -- no dice nada sobre qué tan fácil es aprender esa señal con un presupuesto corto. A cobra lo mismo por cada minuto que cualquier persona está activa, sin importar si ya está pasada de tolerancia o recién llegó -- no hay ninguna zona "barata" cerca del límite de tolerancia que el agente pueda explotar rápido, a diferencia de B/C/D, donde alguien servido dentro de los 12 minutos no cuesta nada. Con solo 30 000 pasos, esa falta de estructura aparentemente pesa más que la ventaja teórica de estar optimizando el objetivo correcto -- una distinción real entre "la función objetivo correcta" y "la función objetivo fácil de aprender rápido", y una razón concreta para no asumir que la formulación más directa es automáticamente la mejor elección práctica.

**C (B + shaping potencial) cumple exactamente lo que promete el teorema de Ng et al.: no cambia qué es óptimo, ayuda a encontrarlo mejor con presupuesto corto -- y acá se ve en el peor caso y en la eficiencia, no en el promedio.** El shaping le da al agente una señal inmediata por reducir la cola, en cada paso, en vez de solo el castigo tardío de la incomodidad -- con 30 000 pasos (un quinto del presupuesto final), esa señal densa parece haber ayudado a la política a encontrar un patrón de despacho más deliberado: bastantes menos movimientos (72, casi la mitad que A/B/D) sin sacrificar el peor caso (104.75 min, el mejor de los cuatro) -- protege exactamente lo que un techo *sin* shaping (como B) no logra proteger (B tiene el peor máximo de los cuatro, 139.93 min). Es consistente con la garantía teórica: la política óptima de C es la misma que la de B (mismo `r_B` de fondo), así que cualquier diferencia observada es una diferencia de qué tan rápido/bien se llega a una buena política con el mismo presupuesto de entrenamiento, no una diferencia de qué política es "correcta".

**Lectura general:** la elección de recompensa importa, y no siempre en la dirección que la intuición teórica sugeriría a primera vista -- exactamente la razón por la que este proyecto compara sobre métricas reales y no sobre el valor de cada recompensa (sección 4). Si el objetivo de un despliegue real fuera minimizar el caso típico, D es la elección defendible con esta evidencia; si el objetivo es proteger el peor caso y ser eficiente en movimientos, C lo es -- ninguna de las dos lecturas es "la respuesta", son dos objetivos de servicio legítimamente distintos.

---

## 7. Nota técnica: bug de aliasing descubierto al implementar G (afecta también a C, ya entrenada)

Al implementar G se necesitaba "cómo estaba el mundo antes de este paso" -- el mismo requisito que ya tenía C (para calcular $\Phi(s_t)$). La primera implementación de ambas guardaba el objeto `EstadoSimulacion` completo del paso anterior (`self._estado_previo = estado_despues`) para leerlo en el siguiente paso. **Esto tenía un bug real, silencioso, que llevaba entrenado desde que C existe:**

`EstadoSimulacion.colas`/`.barcos` (`simulador/src/estado.py`) se construyen en `env.py` → `_construir_estado()` pasando `self.colas`/`self.barcos` **por referencia**, no por copia:

```python
EstadoSimulacion(t_actual_min=self.t_actual_min, dia_semana=self.dia_semana,
                  barcos=self.barcos, colas=self.colas, atendidas_historico=self.atendidas_historico)
```

Un `EstadoSimulacion` guardado de un paso anterior **no queda congelado**: sus `colas`/`barcos` son literalmente los mismos objetos que el simulador sigue mutando en cada paso siguiente, así que leerlo más tarde devuelve el estado ACTUAL del simulador, no el histórico del momento en que se guardó. Solo `t_actual_min` (un float simple, copiado por valor) se mantenía correcto.

**Confirmado con una prueba directa:** se guardó `estado_0` en `env.reset()`, se corrieron 8 pasos más, y se volvió a leer `estado_0` -- tenía 19 unidades activas (debería tener 0, el estado inicial vacío) mientras `estado_0.t_actual_min` seguía correctamente en 360.0 (el minuto de inicio). `id(estado_0.colas) == id(env.colas)` daba `True` -- literalmente el mismo diccionario.

**Impacto real:** para G, esto producía valores de recompensa incorrectos en cada paso con llegadas nuevas (confirmado con una identidad de conservación que debía dar 0 de diferencia y no lo hacía). Para C, el término de shaping $\gamma\Phi(s_{t+1})-\Phi(s_t)$ se calculaba con un $\Phi(s_t)$ que en realidad era $\Phi$ del estado ACTUAL (post-mutación), no el de antes del paso -- **C ya fue entrenada y sus resultados ya están reportados en la sección 5 bajo este bug.** El teorema de invariancia de política de Ng et al. (sección 2.C) sigue garantizando que la política óptima de C sigue siendo la de B *si* el término fuera un shaping potencial válido -- con el bug, el término efectivamente inyectado no es $\gamma\Phi(s')-\Phi(s)$ sino una cantidad relacionada pero distinta (calculada con dos lecturas del mismo estado mutante), así que esa garantía no aplica estrictamente a la C ya entrenada. No se retan a C retroactivamente (no se pidió, y viola "no modifiques lo que ya existe") -- se deja este hallazgo documentado explícitamente en vez de ocultarlo, y queda para una decisión posterior si vale la pena reentrenar C con el fix.

**El fix (usado en C y G desde ahora, y en toda corrida nueva de ambas):** en vez de guardar el objeto `EstadoSimulacion`, se calcula y guarda un **escalar** (`potencial_antes` para C, `s_tiempo_antes` para G, vía el helper `suma_tiempo_activo()`) en el momento exacto en que el estado todavía es fresco -- un float es inmune a mutaciones futuras porque no comparte memoria con nada. Implementado en `EntornoRecompensaIntercambiable._actualizar_escalares_previos()`, llamado al final de cada `reset()`/`step()`. Verificado: con este fix, la identidad de conservación de G da diferencia exacta 0.000000000 sobre un episodio completo, y `C` con `eta_potencial=0` vuelve a coincidir bit a bit con `B` en los 90 pasos de un episodio de prueba (ver `00_verificar_formulas.ipynb`).

## 8. Problema de escala en H, medido

`H` pondera $J_T$ (tiempo cuadrático) y $J_M$ (movimiento) con pesos nominales 95%/5%. Se midió la magnitud real de cada término, sin ponderar, corriendo la política base un episodio completo (semilla 1001, `00_verificar_formulas.ipynb`):

| Término | Acumulado (episodio completo) | Promedio por paso |
|---|---|---|
| $J_T$ (tiempo cuadrático) | 229 567.00 | 2 550.74 |
| $J_M$ (movimiento) | 96.00 | 1.07 |

**Razón $J_T/J_M \approx 2391\times$ antes de ponderar.** Después de aplicar $w_T=0.95$/$w_M=0.05$, la contribución real a la suma ponderada es:

| Término | Contribución ponderada | % real de la señal |
|---|---|---|
| Tiempo (95% nominal) | 218 088.65 | **99.9978%** |
| Movimiento (5% nominal) | 4.80 | **0.0022%** |

**El 95/5 nominal no representa, en la práctica, una contribución de 95%/5% -- representa, efectivamente, 100%/0%.** El término de movimiento es aritméticamente insignificante frente al de tiempo con estos pesos: $J_T$ ya es ~2391 veces más grande que $J_M$ antes de ponderar, así que incluso con solo un 5% de peso nominal sigue dominando por completo. Por decisión explícita, este problema de escala **no se corrige a mano** (no se reescalan los pesos ni se normaliza $J_M$/$J_T$ para forzar un 95/5 real) -- se documenta tal cual, porque forma parte de lo que este experimento mide: si se define un multiobjetivo con pesos nominales sin verificar antes que ambos términos estén en escalas comparables, el resultado práctico puede ser radicalmente distinto de la intención nominal. H, tal como está configurada, es en la práctica casi indistinguible de F (tiempo cuadrático puro) en términos de qué optimiza -- una observación a tener en cuenta al leer sus resultados en la sección 5.

## 9. Bibliografía

- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Little, J. D. C. -- la identidad L=λW. Notas de referencia: Columbia University, IEOR 4404, *Little's Law*. http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-LL.pdf ; MathWorld, *Little's Law*. https://mathworld.wolfram.com/LittlesLaw.html
- AdaPool (2021). *Adaptive Fleet Rebalancing via Reinforcement Learning.* arXiv:2104.00203. https://arxiv.org/abs/2104.00203
- RAST-MoE-RL (2025). arXiv:2512.13727 (reward hacking / anti-gaming en RL). https://arxiv.org/abs/2512.13727
- *Multi-Objective Rebalancing for Vehicle Sharing Systems* (2020). arXiv:2007.06801. https://arxiv.org/abs/2007.06801
- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd ed.). http://incompleteideas.net/book/the-book-2nd.html
