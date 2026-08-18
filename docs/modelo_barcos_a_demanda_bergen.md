# Sistema de barcos a demanda en Bergen — planteamiento del modelo

*Documento de trabajo para la reunión con Julio y Stein. Resume la lógica de la operación tal como la plantearon en la reunión.*

---

## 0. El espíritu del proyecto (léelo antes que nada)

No se trata de tomar un servicio existente y mejorarlo. **No existe** una flota de barcos pequeños a demanda en Bergen. Lo que se pide es **diseñar la lógica de cómo funcionaría esa operación**, para cuando alguien decida montarla. Es crear la operación desde cero.

Tres cosas que quedaron claras:

1. **Lo que importa es la lógica, no la velocidad del algoritmo.** Con ~25 puertos no hay problema computacional serio. Si el modelo queda lento, eso se arregla después con mejor computador o mejor algoritmo. Lo valioso es tener un modelo bien planteado que de verdad quieras resolver.
2. **No se busca rentabilidad.** Ningún transporte público es rentable por sí solo. La justificación es ambiental, ahorrar inversión en carreteras, y liberar el centro de la ciudad de autos. **No modelar ingresos ni ganancia.**
3. **La demanda te la inventas.** No existe el servicio, así que no hay datos reales de demanda. Se inventan patrones realistas. Esto está permitido y esperado.

---

## 1. El problema núcleo (una sola frase)

> Dado quién apretó el botón, dónde están los barcos ahora, y quién espera en qué puertos y hacia dónde va, **¿cómo asigno y programo los barcos para los próximos minutos**, usando el patrón de demanda para no quedar mal posicionado después?

Todo lo demás son refinamientos de esto.

---

## 2. Los datos de la instancia

- **Puertos:** un mapa con las posiciones aproximadas de 4 puertos (nodos de demanda) para empezar (crece a ~25 después). Bergen sirve porque la gente reconoce el mapa, aunque los puertos sean inventados. (Hegreneset se guarda aparte como waypoint de ruteo, no como puerto de demanda.)
- **Distancias / tiempos de viaje:** entre cada par de puertos. Se pueden sacar de posiciones en el mapa + estimación de distancia tipo Google.
- **Flota:** un número fijo de barcos. Al inicio, **todos del mismo tamaño / misma capacidad**.
- **Patrón de demanda:** inventado, pero con estructura realista (ver sección 5).

---

## 3. Las decisiones (las palancas del modelo)

En cada instante de decisión (por ejemplo cada pocos minutos), para cada barco disponible:

- **¿Se mueve o se queda?**
- **Si se mueve, ¿a qué puerto va y qué pasajeros recoge?**
- **¿Qué ruta/secuencia de paradas sigue** (directo, o con conexión/parada intermedia para juntar pasajeros de baja demanda)?
- **Reposicionamiento:** cuando un barco queda libre, ¿dónde lo dejo esperando para estar bien parado para la demanda que viene?

---

## 4. El objetivo (OJO: no es ganancia)

El objetivo es **dar buen servicio de forma eficiente**. Se plantea de dos formas equivalentes, elige una como principal:

- **Forma A — fijar la garantía, minimizar recursos:** garantizo que nadie espere más de *X* minutos (ej. 15) y minimizo el número de barcos / el costo de operación necesario para lograrlo.
- **Forma B — fijar la flota, medir el servicio:** dado un número fijo de barcos, ¿qué garantías de espera/llegada puedo ofrecer en cada conexión?

Métricas de servicio que entran en el objetivo: tiempo de espera del pasajero, tiempo total de viaje, y **cumplimiento de las garantías** (sobre todo en las conexiones importantes, las de ida/vuelta al trabajo).

Las conexiones importantes (mucha demanda, típicamente laborales) llevan **garantía fuerte** (viaje directo). Las de poca demanda pueden llevar **garantía más débil** (aceptan conexión intermedia).

---

## 5. El rol de la demanda y el "efecto cola" (el concepto clave)

La demanda **no es un pronóstico clásico**; es un **patrón espacio-temporal** que describe quién viaja, cuándo y hacia dónde:

- En la mañana la gente va **hacia** la ciudad; en la tarde **sale** de la ciudad → muy desbalanceado (~90/10 en una dirección).
- Otras conexiones son más balanceadas (~50/50).
- Ese patrón se puede **aprender y actualizar** con el tiempo (aquí entra el ML), pero eso es una fase posterior.

**Por qué la demanda es imprescindible — el efecto cola (*tail effect*):**

Si optimizas solo para la gente que espera **ahora**, como si el mundo se acabara en este instante, mandas todos los barcos a atender esa demanda inmediata. Cinco minutos después todos los barcos quedaron en el lugar equivocado y no puedes atender a los nuevos pasajeros. Es como un problema de inventario que se vacía al final porque "ya no importa".

**Conclusión:** hay que mirar un poco **hacia el futuro** (usando el patrón de demanda) al decidir el movimiento de los barcos ahora. No para adivinar el futuro exacto, sino para no tomar decisiones miopes que te dejen mal posicionado.

---

## 6. La estructura de dos capas (híbrido)

Lo que les interesa no es el "schedule-free" puro (ese término ya casi no se usa; ahora se habla de *demand-driven*). Lo valioso es tener **las dos cosas juntas**:

| Capa | Qué es | Cómo opera |
|------|--------|-----------|
| **Fija** | Ferries / express boats grandes existentes | Mantienen **horario**. Se les puede aplicar optimización de velocidad. |
| **Flexible** | Barcos pequeños a demanda (nuevo, se simula) | **Sin horario**: el pasajero pide con el botón y el barco va. Aquí vive el modelo de este documento. |

El valor del híbrido: puede **ahorrar inversión en infraestructura** y quitar autos del centro.

---

## 7. Qué simplificar en la versión 1 (y qué dejar para después)

**Versión 1 (arranca aquí):**
- 4 puertos (nodos de demanda) alrededor de la ciudad (evitar los de larga distancia tipo Knarvik; sí incluir tipo Kleppestø). Hegreneset no cuenta como puerto: es solo un waypoint intermedio para el ruteo, sin demanda propia.
- Todos los barcos **del mismo tamaño**.
- Patrones de demanda **inventados** pero razonables (picos direccionales por hora del día).
- Foco: la lógica de asignar/programar barcos cada pocos minutos con mirada al futuro (evitar el efecto cola).

**Refinamientos posteriores:**
- Barcos de **distinto tamaño** (hay que elegir qué barco va a dónde) y **capacidad** (ej. llega una clase de colegio de 25 personas).
- **Conexiones indirectas** para demanda baja, con garantías diferenciadas.
- Operación **nocturna** (pocos barcos pequeños que de día hacen rutas menores y de noche funcionan como taxis).
- **Aprendizaje** del patrón de demanda que se actualiza con el tiempo (ML).
- Eventos especiales (partidos de fútbol, un colegio entero que se mueve de A a B).
- Una **simulación gráfica** sencilla para mostrar cómo se mueven los barcos y llega la gente.

---

## 8. El ángulo de machine learning / IA

Dos lugares naturales, ambos para fase posterior a la v1:

1. **Aprender el patrón de demanda** en el espacio y el tiempo, y actualizarlo continuamente (no día a día, sino en ventanas más largas, para no dejarse engañar por un evento puntual).
2. **La política de decisión en tiempo real** (qué barco hace qué en los próximos minutos) se puede ver como un problema de decisión secuencial → aprendizaje por refuerzo, con la demanda futura y el reposicionamiento incorporados.

Ellos lo describieron como "un problema lindo, mezcla de optimización y big data".

---

## 9. El primer paso concreto que te pidieron

No esperan que tengas la respuesta ahora. El primer paso es:

1. Tomar el **mapa de Bergen** y ubicar 4 puertos (nodos de demanda).
2. Estimar **distancias/tiempos** entre ellos.
3. **Inventar patrones de demanda** simples (picos direccionales por hora).
4. Con eso, plantear la **instancia base** y empezar a jugar con la lógica de asignación de barcos.

Con eso ya llegas con algo empezado.

---

## Preguntas abiertas para llevar a la reunión

- ¿El horizonte de decisión es "rolling" (cada X minutos re-optimizo) o de otro tipo?
- ¿Cómo formalizamos la "garantía" — como restricción dura (nadie espera más de 15 min) o penalización blanda en el objetivo?
- ¿La mirada al futuro entra como un modelo de dos etapas / estocástico (Stein), o como una penalización de reposicionamiento?
- Para la métrica de servicio, ¿priorizamos espera del pasajero, tiempo total, o cumplimiento de garantías por conexión?
- ¿Vale la pena la reformulación cónica de Julio en alguna parte (ej. velocidad/consumo), o eso es más de la capa fija?
