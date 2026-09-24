# Barcos autónomos on-demand en Bergen: resumen del proyecto

*Documento de síntesis para presentar el trabajo. Secuencia: objetivo y caso de estudio, el sistema (estados y acciones), los escalones de prueba, la política de referencia, el modelo de refuerzo, y la experimentación con la función de recompensa. Referencias con enlaces al final.*

---

## 1. Objetivo y caso de estudio

**Pregunta del proyecto.** Si un servicio de transporte acuático **on-demand** (barcos autónomos que responden a la demanda, sin horario fijo) llegara a implementarse en una ciudad como Bergen, ¿qué metodología de operación conviene más? Para responderlo comparamos dos formas de decidir cómo se mueven los barcos:

- una **política de referencia**, una heurística fija y competente, y
- un **modelo de aprendizaje por refuerzo (RL)** que aprende la operación por ensayo y error.

**Caso de estudio.** Cuatro puntos reales sobre las vías de agua de Bergen: Kleppestø, Laksevåg, Bryggen y Sandviken. Los tiempos de viaje son náuticos reales (ruteo sobre agua que rodea la tierra, calibrado a 30 km/h; Kleppestø–Bryggen ≈ 11.5 min, cercano al ferry real de 14 min). La demanda es sintética pero anclada en datos abiertos de Statistics Norway (SSB) mediante un modelo de gravedad.

El servicio no existe hoy, así que el trabajo diseña **la lógica de la operación**, no optimiza un sistema existente.

---

## 2. El sistema: estados, acciones y transición

La operación se modela como un **Proceso de Decisión de Markov (MDP)**. El simulador avanza en pasos fijos de $\Delta t = 2$ min y es el que ejecuta la transición.

**Estado** (lo que ve quien decide). Un vector de tamaño fijo con tres bloques:
- por cada barco: posición (nodo origen y destino), minutos para llegar, ocupación;
- por cada par origen-destino: personas esperando y hace cuánto espera la más antigua;
- el tiempo: minuto del día y día de la semana.

*Ejemplo (2 barcos):* barco 0 quieto en Bryggen, vacío; barco 1 viajando de Kleppestø a Bryggen, 6 min para llegar, 12 personas a bordo; en la cola Kleppestø→Bryggen esperan 15 personas hace 8 min.

**Acciones.** Por cada barco libre, una de cinco: ir a uno de los cuatro nodos, o esperar. Un barco en ruta sigue hasta su destino (no se redirige a media ruta).

**Regla de embarque (del simulador, no de quien decide).** Las colas están separadas por par origen-destino. Un barco enviado de $A$ a $B$ embarca solo de la cola $A\to B$, en orden de llegada, hasta llenar su capacidad (20), y baja a todos en $B$. Son viajes directos punto a punto.

**Transición.** En cada paso: se aplican las órdenes, los barcos avanzan 2 min (los que llegan bajan y suben pasajeros), llegan las personas nuevas del intervalo, y se calcula la recompensa. El simulador es **determinista** dada una realización de demanda.

---

## 3. Los escalones de prueba

El sistema se verifica en tres tamaños crecientes, para poder revisar a fondo antes de escalar:

| Escalón | Ventana | Barcos | Demanda | Uso |
|---|---|---|---|---|
| 1 | mañana, 06–09 h (3 h) | 2 | ≈ 200 personas | verificación fina, entrenamiento del agente |
| 2 | día completo, 06–24 h | 3 | ≈ 1 100 personas | métricas agregadas |
| 3 | semana completa | 3 | 324–1 121 personas/día | comportamiento a largo plazo |

**El escalón 1 en detalle.** Concentra toda la demanda en las tres horas pico de la mañana, con flujo dominante hacia Bryggen (el centro), como se observa en cualquier ciudad. Al ser una única ventana pico sin valles para drenar el rezago, es el caso más exigente por persona y el más fácil de inspeccionar minuto a minuto. Por eso es el escalón sobre el que se entrena y compara el agente en esta primera fase.

---

## 4. Política de referencia (heurística)

Es una regla fija, sin aprendizaje. Decide el próximo destino de cada barco libre. Sea $A$ el nodo de un barco libre y $w_{(o,d)}$ el tiempo que lleva esperando la persona más antigua de la cola $(o,d)$.

**Regla de decisión (en dos niveles, prioridad estricta a lo local).**

1. **Demanda local primero.** Si hay colas que salen de $A$, el barco va al destino cuya persona más antigua lleva más esperando:

2. **Reposicionamiento.** Solo si no hay ninguna demanda local, el barco va a buscar la demanda remota más urgente (empate: el nodo más cercano en tiempo de viaje):


Regla 1

Primero mira si existe demanda en el nodo donde está el barco.

Regla 2

Si hay demanda local, la atiende antes que cualquier demanda remota.

Regla 3

Si hay varias demandas locales, escoge la más antigua/urgente.

Regla 4

Si no hay demanda local, busca demanda en otros nodos.

Regla 5

Entre las demandas remotas, prioriza la que lleva más tiempo esperando.

Regla 6

Si hay empate, utiliza el tiempo de viaje como desempate.

Regla 7

Cuando hay varios barcos libres simultáneamente, no los deja tomar decisiones completamente independientes: coordina sus decisiones mediante una copia temporal de las colas.


**Coordinación de flota.** La coordinación evita que varios barcos libres seleccionen de forma independiente la misma demanda. Para ello, las asignaciones se realizan secuencialmente sobre una copia temporal de las colas, descontando la demanda que ya sería atendida por las asignaciones anteriores.

**Límite conocido, declarado.** La prioridad estricta a lo local puede dejar esperando mucho a una persona urgente en otro nodo si el patrón nunca le da prioridad a ese par.


## 5. Modelo de aprendizaje por refuerzo

**Qué es.** Un agente decide, en cada paso, a qué nodo va cada barco. Recibe una recompensa que refleja la calidad del servicio y ajusta su política para maximizar la recompensa acumulada. La política vive en una red neuronal.

**Algoritmo: PPO.** La acción es una decisión por barco (espacio *MultiDiscrete*), que DQN no admite; PPO sí lo maneja de forma nativa. PPO es actor-crítico (una red elige la acción, otra estima el valor del estado) y actualiza la política en pasos pequeños, lo que lo hace estable.

**Configuración (real).**

| Parámetro | Valor |
|---|---|
| Descuento $\gamma$ | 0.99 |
| Tasa de aprendizaje | 3·10⁻⁴ |
| Coef. de entropía | 0.01 |
| Horizonte de actualización $n_{\text{steps}}$ | 512 |
| Normalización | VecNormalize (observación y recompensa) |
| Pasos de entrenamiento | 150 000 |
| Semilla entren. / evaluación | 123 / 1001–1005 |

Cada episodio de entrenamiento usa una realización de demanda distinta, de modo que el agente aprende a generalizar y no a memorizar una semana. La evaluación usa semillas que el agente nunca vio.

**Notación usada de aquí en adelante** (secciones 5 y 6 - misma notación en todas las fórmulas):

| Símbolo | Significado |
|---|---|
| $t$ | Índice de paso/instante de tiempo ($\Delta t=2$ min entre pasos). |
| $i$ | Índice de una persona (o grupo) de la demanda. |
| $\mathcal{P}$ | Conjunto de TODOS los pasajeros generados en el episodio. |
| $\mathcal{A}_t \subseteq \mathcal{P}$ | Personas activas en el instante $t$: ya llegaron, todavía no fueron entregadas. |
| $\text{tam}_i$ | Tamaño (peso) de la persona/grupo $i$ - 1 en la configuración actual (`unidad_demanda="personas"`). |
| $T_i$ | Tiempo TOTAL en el sistema de la persona $i$, una vez entregada ($T_i = t_i^d - t_i^a$) - la métrica principal (5.1). |
| $\tau_{i,t}$ | Tiempo que la persona $i$ lleva en el sistema en el instante $t$, mientras sigue activa (el valor "corriendo"; al entregarse, $\tau_{i,t}$ coincide con $T_i$). |
| $v$ | Índice de un barco. |
| $m_{v,t}\in\{0,1\}$ | Indica si el barco $v$ se está moviendo (navegando) en el paso $t$. |
| $m_t=\sum_v m_{v,t}$ | Número de barcos moviéndose en el paso $t$ - el término de "costo de movimiento" que usan las recompensas de la sección 6. |
| $N_t=\sum_{i\in\mathcal{A}_t}\text{tam}_i$ | Número (ponderado por tamaño) de personas activas en el instante $t$. |
| $r_t$ | Recompensa que recibe el agente en el paso $t$. |
| $s_t$ | Estado del sistema en el paso $t$ (lo que ve el agente, sección 2). |
| $\gamma$ | Descuento de PPO (0.99, tabla de arriba). |

### 5.1 La métrica principal: tiempo en el sistema

El objetivo operacional es minimizar el tiempo que los pasajeros permanecen en el sistema, desde su llegada hasta su entrega en el destino $T_i = t_i^{d}-t_i^{a}$ (espera más viaje), reportada como media y máximo. El porcentaje atendido y los movimientos son secundarios.


Principal: Queremos una política que minimice el tiempo total en el sistema de todos los pasajeros

$$
\boxed{\;\min_{}\;\;\sum_{i\in\mathcal{P}} T_i}
\tag{1}
$$


Secundario minimizar movimientos ineficientes:

$$
\boxed{\;\min_{}\;\; \sum_{t}\sum_{v} m_{v,t}\;}
\tag{2}
$$

Para entrenar PPO, este objetivo debe transformarse en una señal de recompensa que pueda evaluarse en cada transición de 2 minutos. Por esto se estudian diferentes formulaciones de reward.

### 5.2 Función de recompensa y alternativas evaluadas

La mejor de las variantes probadas (sección 6) fue **D**, la configuración de producción. Para cada persona activa (aún no entregada), con la notación de la tabla de arriba:

$$r_t = -\sum_{i \in \mathcal{A}_t}\text{tam}_i\left(\frac{\tau_{i,t}}{12}\right)^2$$

- $12$: tiempo de referencia (minutos).
- El exponente $2$: hace que la penalización aumente cuadráticamente con el tiempo.


### 5.3 Resultados (evaluación, escalón 1, semillas 1001–1005)

| Política | Tiempo medio (min) | Tiempo máx (min) | Espera media (min) | Movimientos | Atendidas |
|---|---|---|---|---|---|
| Base (heurística) | 26.1 | 69.7 | 16.4 | 155 | 88.0 % |
| **RL (D)** | **25.7** | **51.9** | **16.2** | 174 | 84.5 % |

En la métrica principal, el agente supera a la heurística: menor tiempo medio (25.7 vs 26.1) y, sobre todo, mejor peor caso (51.9 vs 69.7 min). El precio es atender un 3.5 % menos de personas. En otras palabras, la política aprendida entrega más rápido y con un peor caso más controlado, a cambio de dejar una fracción algo mayor sin atender dentro de la ventana.



## 6. Experimentación con la función de recompensa

La recompensa es la pieza más delicada de un modelo de RL: es lo único que el agente ve para saber qué queremos, y diseñarla mal produce políticas que maximizan la recompensa sin cumplir el objetivo (fenómeno conocido como *reward hacking* o mala especificación, ligado a la ley de Goodhart [3][4]). Por eso se compararon **siete** funciones, todas bajo las mismas condiciones (PPO, mismos hiperparámetros, mismo presupuesto de 150 000 pasos, misma semilla de entrenamiento, mismas cinco semillas de evaluación) - lo único que cambia entre ellas es la fórmula de recompensa. Siempre se comparan contra las métricas reales, nunca contra el valor de cada recompensa (que no son comparables entre sí: cada una vive en su propia escala).

**Dos correcciones respecto a una ronda anterior de este experimento, ambas verificadas contra el código real antes de tocar nada:**

- **A tenía un término de movimiento que no debía estar ahí.** La fórmula documentada de A siempre fue puramente de tiempo (ver abajo), pero el código incluía además `peso_movimiento * barcos_en_movimiento` - un desajuste real entre lo documentado y lo implementado, no una variante intencional. Se quitó el término y se reentrenó A; los resultados de la tabla 6.2 ya son los corregidos.
- **Una variante de tiempo incremental por paso, evaluada en la misma ronda, quedó fuera de esta comparación** (no aporta una lectura adicional sobre las preguntas que este documento quiere responder). Sus archivos se conservan en el proyecto por si hace falta revisarlos más adelante, pero no se reporta aquí.

### 6.1 Las siete variantes (fundamento, fórmula y valores usados)

**A - objetivo puro (derivado del objetivo verdadero).** Penaliza, por paso, las personas activas por la duración del paso, sin tolerancia ni curva:
$$
r_t = -\sum_{i\in\mathcal{A}_t}\text{tam}_i\cdot\Delta t.
$$
Su fundamento es una identidad exacta (la que sustenta la **Ley de Little**, $L=\lambda W$ [2], con $W\equiv T$ en nuestra notación): el área bajo el conteo de gente en el sistema es igual a la suma de los tiempos individuales,
$$
\sum_t N_t\,\Delta t \;=\; \sum_{i\in\mathcal{P}} \text{tam}_i\,T_i.
$$
Por tanto, **maximizar la recompensa A acumulada es exactamente minimizar el tiempo total en el sistema** (la suma de la ecuación (1)). No es un proxy: es el objetivo mismo, repartido por paso. Valores usados: ninguno - a propósito, A no tiene ningún parámetro (ni tolerancia, ni normalizador, ni techo, ni peso de movimiento).

**B - ponderada convexa con tolerancia, sin techo.** Penaliza solo el exceso sobre una tolerancia, al cuadrado, con un costo chico de movimiento:
$$
r_t = -\Big[\sum_{i\in\mathcal{A}_t}\text{tam}_i\big(\tfrac{\max(0,\tau_{i,t}-12)}{18}\big)^2 + 0.003\, m_t\Big].
$$
Valores usados: tolerancia de 12 min (el viaje directo más largo de la red), normalizador de 18 (ancho de la curva de castigo) y peso de movimiento 0.003 (del mismo orden que la penalización de alguien con ~13 min de espera, para que ninguno de los dos términos domine al otro de entrada).

**C - B más *shaping* potencial (Ng, Harada & Russell 1999 [1]).** Añade un término que da señal densa por reducir la cola, sin cambiar cuál es la política óptima (el teorema de invariancia de PBRS lo garantiza para cualquier función potencial):
$$
r_t = r_t^{B} + \big(\gamma\,\Phi_{t+1}-\Phi_t\big),\qquad \Phi_t = -\eta\,N_t,
$$
usando el mismo $N_t$ (personas activas, ponderadas por tamaño) de la tabla de notación. Valores usados: $\gamma=0.99$ (el mismo descuento que ya usa PPO, no un parámetro nuevo) y $\eta=0.05$. Este último es deliberadamente chico y no se calibró finamente en esta ronda: la idea es que el término de shaping actúe como un empujón adicional suave sobre B (premia reducir la cola en cada paso, antes de que el castigo por incomodidad lo note), no que compita en magnitud con B.

**D - configuración de producción (sin tolerancia, cuadrática pura normalizada).** La fórmula que hoy gobierna la política base y los escalones 1–3, llevada a RL:
$$
r_t = -\sum_{i \in \mathcal{A}_t}\text{tam}_i\left(\frac{\tau_{i,t}}{12}\right)^2.
$$
Valores usados: sin tolerancia (tolerancia = 0, el castigo empieza desde el minuto 1), normalizador 12, sin techo por persona (fijado en un valor enorme, prácticamente infinito) y sin costo de movimiento (peso = 0).

**E - tiempo lineal puro.** Igual que A pero cobrando el tiempo TOTAL acumulado de cada persona activa en vez de un costo fijo por paso, sin ningún otro término:
$$
r_t = -\sum_{i\in\mathcal{A}_t}\text{tam}_i\cdot\tau_{i,t}.
$$

**F - tiempo cuadrático puro, sin normalizar.** Como D pero sin dividir por el normalizador de 12 - la versión "cruda" de esa misma idea:
$$
r_t = -\sum_{i\in\mathcal{A}_t}\text{tam}_i\cdot\tau_{i,t}^2.
$$
Se probó para ver si normalizar realmente importa, dado que PPO ya normaliza la recompensa completa por su cuenta (VecNormalize).

**H - multiobjetivo escalarizado, tiempo + movimiento, con los componentes normalizados antes de ponderar.** Combina un costo de tiempo (igual a F, cuadrático sin normalizar) con el costo de movimiento ya usado en A–D, con pesos nominales 95 %/5 %:
$$
r_t = -\Big[0.95\cdot\tilde{J}_T + 0.05\cdot\tilde{J}_M\Big],\qquad \tilde{J}_T=\frac{J_T}{J_T^{ref}},\quad \tilde{J}_M=\frac{J_M}{J_M^{ref}},
$$
con $J_T=\sum_{i\in\mathcal{A}_t}\text{tam}_i\,\tau_{i,t}^2$ y $J_M=m_t$ (igual definición que en las demás variantes).

Los componentes se normalizaron antes de aplicar los pesos porque el costo temporal cuadrático y el costo de movimiento están en escalas completamente distintas: medido sobre la política base (referencia externa, nunca sobre una política RL ni sobre H misma, para no normalizar con información que todavía no existe al empezar a entrenar), $J_T$ resultó del orden de miles por paso, mientras que $J_M$ está acotado entre 0 y 2 (el número de barcos). Aplicar 0.95/0.05 directamente sobre esas magnitudes no habría significado una influencia real de 95 %/5 % - el término de tiempo habría dominado por completo, casi sin importar el peso nominal del movimiento.

Valores usados: $J_T^{ref}=3473.2$ y $J_M^{ref}=2.0$, la mediana por paso de cada término bajo la política base, sobre las 5 semillas de evaluación estándar, en la misma instancia (escalón 1) y condiciones que el entrenamiento (se usó la mediana y no el promedio porque $J_T$ varía mucho entre semillas -- una sola semilla con demanda atípica domina el promedio). Esta normalización hace que los pesos 0.95/0.05 tengan una interpretación más clara como preferencia relativa entre los dos objetivos - no es una garantía de que el agente entrenado vaya a repartir su comportamiento exactamente 95 %/5 %, eso depende de qué aprenda, no de la fórmula.

### 6.2 Resultados de la comparación

Misma instancia (escalón 1), mismo presupuesto (150 000 pasos) y mismas cinco semillas de evaluación (1001–1005) para las siete.

| Variante | Tiempo medio (min) | Tiempo máx (min) | Movimientos | Atendidas |
|---|---|---|---|---|
| A (objetivo puro, corregida) | 29.2 | 70.2 | 152 | 74.7 % |
| B (convexa, sin techo) | 29.0 | 60.8 | 191 | 88.0 % |
| C (B + shaping) | 29.2 | 74.1 | 193 | 84.5 % |
| D (producción) | 25.7 | 51.9 | 174 | 84.5 % |
| E (lineal puro) | 41.1 | 123.1 | 172 | 74.8 % |
| F (cuadrático puro) | 33.0 | 81.1 | 183 | 75.9 % |
| H (multiobjetivo, normalizada) | 33.6 | 78.3 | 180 | 79.6 % |


### 6.3 Lecturas

**Fundamento teórico vs. comportamiento observado - no son lo mismo, y ninguna variante "gana" solo por un número.** Solo dos de las siete tienen una garantía teórica detrás: A, por la identidad de Little (minimiza el tiempo total *en el límite*, con entrenamiento suficiente); y C, por el teorema de invariancia de Ng et al. [1] (el shaping no cambia cuál es la política óptima de B, solo puede ayudar a encontrarla más rápido). Las demás (B, D, E, F, H) son formulaciones razonables pero sin esa garantía formal - su desempeño es enteramente empírico, en este presupuesto y estas semillas.

- **A, ya corregida (sin el término de movimiento que no debía tener), rinde muy distinto de lo que se había reportado antes.** Antes de corregirla parecía de las peores; corregida, su tiempo medio (29.2 min) queda prácticamente empatado con B y C, y su tiempo máximo (70.2 min) mejora bastante frente a su versión con movimiento. Es un recordatorio concreto de por qué vale la pena revisar el código contra la fórmula documentada antes de sacar conclusiones sobre una variante.
- **E, la otra formulación lineal "pura", sigue siendo de las más débiles** (41.1 min medio, el peor tiempo máximo de las siete). A diferencia de A, E no tiene la garantía teórica de Little detrás (cobra tiempo acumulado, no un costo fijo por paso) - su desempeño más débil no tiene la misma lectura que la de A antes del fix.
- **F mejora sobre E en casi todas las métricas** (33.0 vs 41.1 min medio, 81.1 vs 123.1 min máximo) - consistente con que castigar de forma creciente (cuadrático) los tiempos largos ayuda frente a un castigo lineal, la misma idea que ya separaba a B de A.
- **C cumple lo que promete el teorema de Ng et al. [1]:** no cambia el óptimo de B, y en la práctica ayuda a llegar mejor con presupuesto corto (mejor peor caso y bastantes menos movimientos que B).
- **H, ya con sus dos componentes en escalas comparables, se ubica entre F y D** (33.6 min medio, 78.3 min máximo) -- más cerca de F (tiempo puro) que de un punto medio exacto entre tiempo y movimiento, algo esperable con un peso nominal de 95 % para el tiempo. La normalización resuelve el problema de escala (ahora los pesos sí comparan magnitudes del mismo orden), pero no obliga a que el comportamiento aprendido se reparta 95/5 -- eso depende de qué le convenga al agente, no de la fórmula.
- **D sigue teniendo el mejor tiempo medio y máximo de las siete en esta ronda**, incluso comparada contra la A corregida. No colapsó pese a no tener tolerancia ni techo - el diagnóstico histórico de que estos parámetros paralizaban al agente se explica por la falta de normalización de aquel momento, no por los parámetros en sí (con VecNormalize activo, el desbalance crudo pesa mucho menos).

**Nota metodológica.** Durante el desarrollo de esta ronda de experimentos se encontró y corrigió un error real en cómo C guardaba el estado del paso anterior para calcular su término de shaping (una referencia compartida que terminaba leyendo el estado ya mutado, no el histórico). El resultado de C en la tabla es el de la corrida original, previa al fix - no se reentrenó retroactivamente porque no cambia la lectura cualitativa (C sigue sin tener por qué ganar en tiempo medio, su aporte es el peor caso y los movimientos), pero es una corrección real, documentada en el repositorio, y a tener en cuenta si se decide reentrenar C.

**Elección para continuar (no es una conclusión definitiva).** Con la evidencia de esta ronda, D sigue siendo la formulación con la que se sigue esta fase del proyecto - una decisión práctica sobre este presupuesto y estas semillas, no una afirmación de que D sea teóricamente "la" recompensa correcta (esa garantía, hasta ahora, solo la tienen A y C, y ninguna de las dos ganó en la métrica principal, aunque A corregida quedó mucho más cerca). Una ronda con más presupuesto de entrenamiento, más semillas de entrenamiento (no solo de evaluación), o una recalibración de C, podría cambiar esta lectura - queda como trabajo futuro, no como algo ya descartado.

**Principio metodológico.** Todo se compara sobre el tiempo en el sistema y el porcentaje atendido, no sobre la recompensa propia de cada variante. Es la defensa contra optimizar un proxy hasta que deja de reflejar el objetivo [3][4].



## Referencias

1. Ng, A., Harada, D., Russell, S. (1999). *Policy Invariance Under Reward Transformations.* ICML. - shaping potencial y garantía de invariancia. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
2. Little's Law ($L=\lambda W$). Notas de Columbia (IEOR): http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-LL.pdf · MathWorld: https://mathworld.wolfram.com/LittlesLaw.html
3. Amodei, D. et al. (2016). *Concrete Problems in AI Safety* - reward hacking. https://arxiv.org/abs/1606.06565
4. RAST-MoE-RL (2025). arXiv:2512.13727 - diseño de recompensa anti-gaming en despacho. https://arxiv.org/abs/2512.13727
5. Gu, Y., Wallace, S. W. (2021). *Operational benefits of autonomous vessels - water-taxis in Bergen.* TR-E 154:102456. https://doi.org/10.1016/j.tre.2021.102456
6. Braathen, C., Goez, J. C., Guajardo, M. (2024). *Autonomous ferries in light of labor regulations.* Maritime Transport Research. https://doi.org/10.1016/j.martra.2024.100115
