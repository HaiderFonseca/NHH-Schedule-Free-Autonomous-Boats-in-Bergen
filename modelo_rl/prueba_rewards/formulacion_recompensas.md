# Formulación matemática de las funciones de recompensa

*Derivación paso a paso, con demostraciones y ejemplos numéricos reales del proyecto (4 nodos de Bergen, capacidad 20, matriz de tiempos náuticos). Objetivo: tener varias recompensas bien fundamentadas para compararlas experimentalmente.*

**Enlaces de todas las referencias al final (sección 8), numerados [1]–[9].**

---

## 0. Notación

| Símbolo | Significado |
|---|---|
| $\mathcal{P}$ | conjunto de pasajeros (o unidades) del episodio |
| $t_i^{a}$ | minuto en que el pasajero $i$ **llega** a la parada (aparece) |
| $t_i^{d}$ | minuto en que el pasajero $i$ es **entregado** en su destino |
| $T_i = t_i^{d}-t_i^{a}$ | **tiempo total en el sistema** del pasajero $i$ |
| $\Delta t$ | duración del paso (aquí $\Delta t = 2$ min) |
| $N_t$ | número de pasajeros aún no entregados en el paso $t$ (en cola o a bordo) |
| $\tau_{i,t}=t\cdot\Delta t - t_i^{a}$ | tiempo que lleva el pasajero $i$ en el sistema al paso $t$ |
| $m_{v,t}\in\{0,1\}$ | 1 si el barco $v$ se está moviendo en el paso $t$ |
| $\gamma$ | factor de descuento del RL |
| $s_i$ | tamaño del grupo/unidad $i$ (personas); con personas sueltas $s_i=1$ |

---

## 1. El objetivo verdadero (lo que queremos), formal

Queremos una política $\pi$ que minimice el tiempo total en el sistema de todos los pasajeros más una penalización por movimientos ineficientes:

$$
\boxed{\;\min_{\pi}\;\; J(\pi)=\sum_{i\in\mathcal{P}} T_i \;+\; \lambda \sum_{t}\sum_{v} m_{v,t}\;}
\tag{1}
$$

$\lambda \ge 0$ pondera cuánto pesa ahorrar movimientos frente a servir rápido. Este es **el** objetivo. Las secciones siguientes construyen recompensas que lo persiguen.

---

## 2. Cómo el RL convierte "minimizar" en "maximizar"

Un agente de RL maximiza la recompensa descontada esperada [1] (Sutton & Barto):

$$
\max_{\pi}\; \mathbb{E}\!\left[\sum_{t=0}^{H}\gamma^{t} r_t\right].
\tag{2}
$$

Para alinear (2) con (1), definimos la recompensa como el **negativo del costo por paso**:

$$
r_t = -\,c_t,\qquad\text{con } \sum_t c_t \text{ que reconstruya } J(\pi).
\tag{3}
$$

Todo el diseño se reduce a elegir bien $c_t$.

---

## 3. RECOMPENSA A — la que sigue el objetivo (derivada de Little)

### 3.1 La identidad (demostración)

**Afirmación.** El tiempo total en el sistema es igual a la suma, sobre los pasos, del número de pasajeros presentes:

$$
\sum_{i\in\mathcal{P}} T_i \;=\; \int_0^{H\Delta t} N(t)\,dt \;\approx\; \sum_{t} N_t\,\Delta t.
\tag{4}
$$

**Demostración (argumento geométrico de Little [2][3][4]).** Cada pasajero $i$ está en el sistema exactamente durante el intervalo $(t_i^{a}, t_i^{d})$. Su tiempo en el sistema se puede escribir como una integral de su función indicadora:

$$
T_i = t_i^{d}-t_i^{a} = \int_0^{H\Delta t}\mathbf{1}\{t_i^{a}<t<t_i^{d}\}\,dt.
$$

Sumamos sobre todos los pasajeros e intercambiamos la suma con la integral:

$$
\sum_{i}T_i=\sum_i\int_0^{H\Delta t}\mathbf{1}\{t_i^{a}<t<t_i^{d}\}\,dt
=\int_0^{H\Delta t}\underbrace{\sum_i\mathbf{1}\{t_i^{a}<t<t_i^{d}\}}_{=\,N(t)}\,dt=\int_0^{H\Delta t}N(t)\,dt.
$$

La suma interna cuenta cuántos pasajeros están en el sistema en el instante $t$: eso es exactamente $N(t)$. Discretizando en pasos de $\Delta t$ se obtiene $\sum_t N_t\,\Delta t$. $\blacksquare$

Esta es la misma identidad que sustenta la Ley de Little $L=\lambda W$ [2][3][4][5]: el área bajo las líneas de vida de los clientes se puede medir de dos maneras (sumando por cliente, o integrando la altura de la pila en el tiempo), y ambas dan lo mismo.

### 3.2 La recompensa A

Elegimos el costo por paso como el número de personas presentes por la duración del paso, más el término de movimiento:

$$
\boxed{\;c_t^{A}=\sum_{i\text{ activo en }t} s_i\,\Delta t \;+\; \lambda\sum_v m_{v,t},\qquad r_t^{A}=-\,c_t^{A}.\;}
\tag{5}
$$

Por (4), $\sum_t \big(\sum_i s_i\,\Delta t\big)=\sum_i s_i T_i$ = tiempo total en el sistema (ponderado por tamaño). Por tanto:

$$
\sum_t r_t^{A}=-\Big(\sum_i s_iT_i+\lambda\sum_{t,v}m_{v,t}\Big)=-J(\pi).
$$

**Maximizar $\sum_t r_t^A$ es exactamente minimizar el objetivo (1).** No hay tolerancia, ni cuadrado, ni techo: es el objetivo puro, repartido por paso. Esta es la recompensa **más defendible**, porque *es* el objetivo.

### 3.3 Ejemplo numérico (datos reales)

Paso $t$ con 3 pasajeros esperando ($s=1$ cada uno) y 1 barco moviéndose, $\Delta t=2$, $\lambda=0.5$:

$$
c_t^{A}=(1+1+1)\cdot 2 + 0.5\cdot 1 = 6.5,\qquad r_t^{A}=-6.5.
$$

Si esos 3 esperan durante 5 pasos (10 min) antes de ser recogidos, acumulan $3\cdot 10=30$ de penalización de tiempo — que es justo la suma de sus tiempos en el sistema. Nada arbitrario.

---

## 4. RECOMPENSA B — suma ponderada bien escalada (estándar de despacho)

La recompensa A es pura pero "plana" (cada minuto de espera pesa igual). La práctica en despacho de flotas usa una **suma ponderada de objetivos** [6] (AdaPool) y a veo penaliza de forma **no lineal** para dar urgencia. Formalizamos y, sobre todo, **escalamos con cuidado** [7] (anti-hacking).

### 4.1 Forma general

$$
r_t^{B}=-\Big[\underbrace{\beta_{\text{time}}\sum_{i\text{ activo}} s_i\,g(\tau_{i,t})}_{\text{tiempo}} \;+\; \underbrace{\beta_{\text{mov}}\sum_v m_{v,t}}_{\text{movimiento}}\Big]
\tag{6}
$$

donde $g(\tau)$ es cómo penalizamos el tiempo de un pasajero. Dos opciones:
- **Lineal:** $g(\tau)=\tau$ → equivale a la recompensa A (con $\beta_{\text{time}}=1$).
- **Con tolerancia + convexa (cuadrática):** solo penaliza el exceso sobre una tolerancia $\theta$, y crece de forma acelerada:

$$
g(\tau)=\Big(\frac{\max(0,\ \tau-\theta)}{\kappa}\Big)^2
\tag{7}
$$

con $\theta$ = tolerancia (min) y $\kappa$ = normalizador (min). Interpretación: hasta $\theta$ minutos, gratis (viaje "normal"); pasado eso, la molestia crece al cuadrado.

### 4.2 Escalado de los pesos (con números reales)

Aquí está el análisis que evita el reward hacking por pesos [7]. Fijamos $\theta=12$ (la ruta directa más larga, Kleppestø–Bryggen ≈ 11.5 min) y $\kappa=18$.

**Penalización de una persona según cuánto lleva:**

| $\tau$ (min) | $\max(0,\tau-12)$ | $g(\tau)=(\cdot/18)^2$ |
|---|---|---|
| 12 | 0 | 0.000 |
| 13 | 1 | **0.0031** |
| 18 | 6 | 0.111 |
| 24 | 12 | 0.444 |
| 30 | 18 | 1.000 |
| 40 | 28 | 2.42 |

**El costo de un movimiento debe ser menor que la penalización marginal de una persona**, o el agente aprende a no moverse [7]. Una persona apenas pasada de la tolerancia (13 min) pesa $\approx 0.003$. Por tanto:

$$
\boxed{\;\beta_{\text{mov}} \lesssim 0.003\;}\quad(\text{probar } 0.001 \text{ y } 0.003),\qquad \beta_{\text{time}}=1.
$$

**Contraste con el error original:** el peso de movimiento estaba en $\beta_{\text{mov}}=0.1$. Un viaje Kleppestø–Bryggen (11.5 min ≈ 6 pasos de 2 min) costaba $6\times0.1=0.6$ — casi lo mismo que una persona al máximo de molestia (1.0). Eso hacía que **mover un barco costara como abandonar a una persona**, y el agente aprendió a quedarse quieto (la política degenerada observada). Con $\beta_{\text{mov}}=0.003$, ese viaje cuesta $\approx0.018$: un empujón suave, no una barrera.

### 4.3 Sobre el techo

El techo por persona $\min(1,\,g(\tau))$ hace que $\tau=30$ y $\tau=50$ penalicen igual (ambos = 1), quitando la urgencia de atender a los más rezagados. **Recomendación: quitarlo** (o subirlo mucho) en los experimentos, para que el agente sienta que dejar a alguien 50 min es peor que 30. Es un parámetro a barrer.

---

## 5. RECOMPENSA C — shaping potencial que preserva el óptimo (Ng, Harada & Russell)

### 5.1 La teoría

Para **guiar** al agente (que aprenda más rápido) **sin cambiar la política óptima**, la única forma con garantía es el *potential-based reward shaping* (PBRS) [8]. Se elige una función potencial $\Phi(s)$ sobre los estados y se añade a la recompensa el término:

$$
F(s,s')=\gamma\,\Phi(s')-\Phi(s).
\tag{8}
$$

**Teorema (Ng et al. 1999) [8]:** con esta forma, y solo con esta forma aditiva, el conjunto de políticas óptimas del problema **no cambia**. Es decir, puedes inyectar la ayuda que quieras vía $\Phi$ sin riesgo de que el agente termine optimizando algo distinto (sin reward hacking por el shaping).

### 5.2 Cómo se haría con NUESTRO estado

Nuestro estado $s$ contiene, entre otras cosas, cuánta gente hay esperando en cada cola. Un potencial natural: **cuanta menos gente pendiente, mejor el estado.** Definimos

$$
\Phi(s) = -\,\eta\, N(s),
$$

donde $N(s)$ = número total de pasajeros aún no entregados en el estado $s$, y $\eta>0$ una escala. Entonces el término de shaping entre el estado $s_t$ y el siguiente $s_{t+1}$ es:

$$
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t)=-\eta\big(\gamma N_{t+1}-N_t\big).
$$

**Interpretación con ejemplo:** supón $\gamma\approx1$, $\eta=1$. Si en un paso el agente **entrega** a 8 personas (baja de $N_t=20$ a $N_{t+1}=12$):

$$
F=-\big(12-20\big)=+8.
$$

Recibe una **señal positiva de +8** por reducir la cola en 8 — exactamente el "premio por entrega" que ya intuíamos, pero ahora **con la garantía teórica de que no cambia el óptimo**, solo acelera el aprendizaje. Si en cambio la cola crece (llega gente y no la atiende, $N$ sube), $F$ es negativo. Es una guía densa y correcta.

### 5.3 La recompensa C

$$
\boxed{\;r_t^{C}=r_t^{A}\;+\;F(s_t,s_{t+1})=r_t^{A}-\eta\big(\gamma N_{t+1}-N_t\big).\;}
\tag{9}
$$

Como $F$ es potencial, **C tiene la misma política óptima que A**, pero aprende más rápido gracias a la señal densa de entrega. Es la forma técnicamente correcta de añadir el "premio por entrega".

---

## 6. Nuestra recompensa actual, formalizada

Para comparar, así está hoy la recompensa del código (`simulador/src/recompensa.py`):

$$
r_t^{\text{actual}}=-\Big[\sum_{i\text{ activo}} s_i\,\min\!\Big(1,\ \big(\tfrac{\max(0,\tau_{i,t}-12)}{18}\big)^2\Big)\;+\;0.1\sum_v m_{v,t}\Big]\;+\;w_{\text{ent}}\,E_t
\tag{10}
$$

con $\tau$ = tiempo en sistema, tolerancia $\theta=12$, normalizador $\kappa=18$, techo $=1$, peso movimiento $=0.1$, y $E_t$ = personas entregadas en el paso (premio $w_{\text{ent}}$, hoy 0 salvo override).

**Diagnóstico frente a lo anterior:** es un caso particular de la recompensa B (ecuación 6–7) **con dos problemas de diseño ya identificados**: (i) $\beta_{\text{mov}}=0.1$ está ~30× por encima de lo recomendado ($\lesssim0.003$), lo que induce inacción [7]; (ii) el techo aplana la urgencia (sección 4.3). El premio de entrega $w_{\text{ent}}E_t$ es un shaping aditivo que **funciona pero no está en forma potencial**, así que no tiene la garantía de invariancia de la recompensa C.

---

## 7. Las cuatro recompensas a comparar experimentalmente

| | Fórmula | Qué prueba | Escala clave |
|---|---|---|---|
| **A** — objetivo puro | ec. (5): $-\sum_i s_i\Delta t-\lambda\sum m$ | la derivada exacta del objetivo (Little) | $\lambda\lesssim 0.003\cdot\Delta t$ |
| **B** — ponderada convexa | ec. (6)–(7) | tolerancia + urgencia cuadrática, sin techo | $\beta_{\text{mov}}\lesssim0.003$ |
| **C** — A + shaping potencial | ec. (9) | aprender más rápido sin cambiar el óptimo | $\eta$ moderado |
| **actual** | ec. (10) | línea de partida (con techo y $\beta_{\text{mov}}=0.1$) | — |

**Regla de validación (anti-Goodhart) [7][9]:** todas se comparan **contra las métricas reales** (tiempo total en sistema, % atendidos), NO contra su propia recompensa. Un agente con "buena recompensa" pero peor tiempo real está haciendo reward hacking.

---

## 8. Referencias (con enlaces verificables)

1. **Sutton, R. & Barto, A. (2018).** *Reinforcement Learning: An Introduction.* — El agente maximiza la recompensa descontada esperada. http://incompleteideas.net/book/the-book-2nd.html
2. **Little, J. D. C. & Graves, S. C. (2008).** *Little's Law.* (capítulo, prueba por área) https://web.eng.ucsd.edu/~massimo/ECE158A/Handouts_files/Little.pdf
3. **Sigman, K. (2009).** *Notes on Little's Law (L = λW).* Columbia. http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-LL.pdf
4. **Whitt / Columbia (2015).** *Notes on Little's Law* (demostración detallada). https://www.columbia.edu/~ww2040/4615S15/LittlesLawNotes012715.pdf
5. **Little's Law — Wolfram MathWorld.** https://mathworld.wolfram.com/LittlesLaw.html
6. **AdaPool (2021).** *A Diurnal-Adaptive Fleet Management Framework using Model-Free Deep RL.* — recompensa multi-objetivo de suma ponderada en despacho. https://arxiv.org/pdf/2104.00203
7. **RAST-MoE-RL (2025).** *Regime-Aware Spatio-Temporal MoE for Deep RL in Ride-Hailing* — "Anti-Hacking Reward Design": costos incrementales y modos de falla (sobre-penalizar → inacción/miopía). https://arxiv.org/pdf/2512.13727
8. **Ng, A., Harada, D. & Russell, S. (1999).** *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML. — PBRS y la garantía de invariancia. Texto y teorema reproducidos en: https://arxiv.org/pdf/2501.00989 (Teorema 1) y original: http://luthuli.cs.uiuc.edu/~daf/courses/games/AIpapers/ng99policy.pdf
9. **Multi-Objective Vehicle Rebalancing for Ridehailing (2020).** — combinación convexa de espera + millas vacías, marco SMDP; validación contra métricas reales. https://arxiv.org/pdf/2007.06801
