## 1. Introducción y descripción del problema

Los modelos de lenguaje de gran escala (LLMs) modernos atraviesan típicamente cuatro fases de entrenamiento antes de su despliegue. La primera es el preentrenamiento masivo sobre corpus generales —billones de tokens de texto extraído de la web, libros y código—, donde el modelo aprende lenguaje y conocimiento del mundo. La segunda, opcional, es el preentrenamiento continuado sobre un corpus de dominio específico (literatura médica, papers científicos, notas clínicas), donde el vocabulario y los patrones del modelo se sesgan hacia el área de interés. La tercera es el ajuste mediante instrucciones supervisadas, donde el modelo aprende a seguir consignas y a producir respuestas útiles a partir de ejemplos curados de pares pregunta-respuesta. La cuarta es el alineamiento por preferencias humanas o sintéticas —implementado mediante técnicas como RLHF—, donde el modelo refina su comportamiento para responder de manera útil, honesta y segura. El conjunto de las fases tres y cuatro suele agruparse bajo el término post-training.

La adaptación de un LLM a un dominio especializado como la medicina típicamente combina las fases segunda y tercera, manteniendo el alineamiento de la cuarta heredado del modelo base. Esa combinación, sin embargo, conlleva costos computacionales del orden de decenas o cientos de miles de dólares en infraestructura especializada — TPU pods, clusters multi-GPU, semanas de entrenamiento — recursos disponibles únicamente para laboratorios industriales como Google, OpenAI o Anthropic. Esa realidad económica concentra la capacidad de producir modelos de dominio en un puñado de actores y limita la adaptación reproducible por parte de instituciones académicas, sistemas de salud regionales o equipos independientes.

El presente proyecto explora la siguiente pregunta con implicaciones prácticas directas: ¿en qué medida es posible recuperar el desempeño de un modelo de dominio especializado prescindiendo de la fase de preentrenamiento continuado y aplicando únicamente ajuste fino eficiente en parámetros sobre la fase de instrucciones? Concretamente, se toma como modelo base Gemma-3-4B-it, un LLM de propósito general ya pasado por preentrenamiento e instrucciones genéricas por parte de Google, y se compara contra MedGemma-4B-it, un modelo construido por Google sobre exactamente la misma base mediante el pipeline industrial completo (preentrenamiento continuado sobre corpus médico de gran escala más ajuste por instrucciones de dominio). La adaptación se realiza mediante una técnica de fine-tuning eficiente que opera únicamente sobre la fase final del pipeline que es instrucciones de dominio, manteniendo el costo computacional dentro del rango de una GPU de consumidor.

El dominio de aplicación es QA médico tipo examen, formulado como preguntas con respuestas de opción múltiple y además con razonamiento explícito. El modelo recibe una viñeta clínica con un caso de paciente y un conjunto de opciones cerradas, y debe generar una cadena de razonamiento textual (análisis del cuadro clínico, descarte de diferenciales) seguida de la opción correcta. 

La tarea es generativa: el output del modelo es texto libre producido token a token, no una clasificación cerrada sobre cuatro o cinco etiquetas. La cadena de razonamiento se evalúa cualitativamente, mientras que la opción final permite cuantificar desempeño mediante una métrica objetiva.

Este dominio es apropiado para evaluar adaptación al dominio en LLMs ya que exige conocimiento factual extenso —anatomía, fisiopatología, farmacología, guías clínicas— combinado con razonamiento estructurado, lo que pone a prueba simultáneamente la memoria del modelo y sus capacidades de inferencia. Además, las viñetas clínicas reproducen fielmente el tipo de razonamiento que ejerce un profesional médico en formación, dotando al benchmark de validez de constructo respecto a la práctica real.

La utilidad del trabajo en un plano práctico es demostrar una vía accesible para adaptar modelos de lenguaje al dominio médico sin requerir infraestructura industrial. Esto es directamente relevante para sistemas de salud regionales, instituciones académicas, equipos de investigación clínica y desarrolladores independientes que necesitan modelos especializados pero carecen del presupuesto de cómputo de un laboratorio puntero. La diferencia entre "es posible únicamente con acceso a TPU pods" y "es posible en una GPU de consumidor durante una sesión de entrenamiento" es la diferencia entre una capacidad concentrada en pocos actores y una capacidad efectivamente distribuida y reproducible.

En el plano metodológico, el experimento aporta evidencia empírica sobre la contribución específica de cada fase del pipeline de adaptación al dominio. Al comparar el modelo resultante contra MedGemma — que comparte modelo base pero recibió preentrenamiento continuado industrial — y contra el modelo base sin adaptar, se aísla y cuantifica cuánto del beneficio total de la adaptación al dominio puede capturarse o rescatarse mediante ajuste eficiente en la fase de instrucciones. Esa cuantificación tiene valor más allá del proyecto puntual: ofrece una referencia para futuras decisiones de costo-beneficio en otros dominios donde la pregunta "¿conviene invertir en preentrenamiento continuado o basta con ajuste por instrucciones?" se plantea de forma análoga.

Lo que hace al problema interesante es el reto e impacto en la adaptación de un dominio en donde la trazabilidad de un razonamiento explícito es indispensable para la confianza profesional, así como la pugna entre modelos generalistas adaptados con técnicas ligeras y modelos especialistas entrenados extensamente desde fases tempranas. Cuantificar hasta dónde llega la adaptación eficiente y dónde permanece la brecha respecto al preentrenamiento especializado contribuye a un debate técnico y económico dentro del campo.

# 2. Objetivos

## 2.1 Objetivo general

Cuantificar empíricamente en qué medida un fine-tune eficiente en parámetros (QLoRA) aplicado sobre Gemma-3-4B-it permite recuperar el desempeño de MedGemma-4B-it en tareas de QA médico con razonamiento explícito, prescindiendo de la fase de preentrenamiento continuado a escala industrial y operando dentro del presupuesto computacional de una GPU de consumidor.

## 2.2 Objetivos específicos

1. **Implementar el pipeline de adaptación**: aplicar QLoRA fine-tuning sobre Gemma-3-4B-it usando MedMCQA como corpus de entrenamiento, aprovechando su campo de explicación nativa como señal de cadena de razonamiento.

2. **Evaluar el desempeño in-distribution**: comparar el modelo resultante con dos referencias —Gemma-3-4B-it sin adaptar y MedGemma-4B-it— sobre MedMCQA test, bajo protocolo idéntico para los tres.

3. **Medir transferencia entre estilos de examen médico**: evaluar los tres modelos sobre MedQA-USMLE-4options test, un conjunto del examen estadounidense USMLE que el modelo no vio durante el entrenamiento, para verificar si la adaptación generaliza más allá del estilo del corpus de entrenamiento.

4. **Cuantificar la fracción de la brecha recuperada**: reportar qué porcentaje de la diferencia de desempeño entre Gemma-3-4B-it y MedGemma-4B-it logra cerrar el fine-tune QLoRA, respondiendo directamente a la pregunta central del proyecto.

5. **Realizar análisis cualitativo del razonamiento generado**: inspeccionar muestras de cadenas de razonamiento producidas por cada modelo para identificar tipos de error como alucinaciones de entidades médicas, fallos de razonamiento clínico y errores de formato.

## 3. Diseño de Arquitectura

## 3.1 Modelo base

El modelo base elegido es **Gemma-3-4B-it**, una variante instruction-tuned de la familia Gemma 3 de Google, con aproximadamente 4 mil millones de parámetros. La versión "-it" indica que ya pasó por instruction tuning y alineamiento generales por parte de Google, por lo que es capaz de seguir instrucciones desde el primer prompt con una ventana de contexto de hasta 128k tokens.

Tres razones soportan esta elección:

**(1) Permite una comparación limpia con MedGemma.** MedGemma-4B-it fue construido por Google a partir de Gemma-3-4B-it, añadiendo preentrenamiento continuado sobre corpus médico. Al partir también de Gemma-3-4B-it, las diferencias observadas entre nuestro modelo y MedGemma se atribuyen exclusivamente al método de adaptación de dominio. Cualquier otro modelo base (Llama, Qwen, Mistral) introduce variables confundidas — distinto tokenizador, distinto corpus de preentrenamiento, distinto tamaño — que estarían fuera de la comparación dado el alcance del presente trabajo.

**(2) Limitaciones sobre hardware disponible.** Un modelo de 4B parámetros, cuantizado a 4 bits, junto los adaptadores entrenables y los gradientes caben dentro de una GPU de consumidor. Modelos más grandes (8B, 27B) requieren hardware industrial; modelos más pequeños (1B-2B) tienen capacidad insuficiente para razonamiento clínico.

**(3) Capacidad multilingüe nativa.** Gemma 3 fue preentrenado sobre un corpus que incluye texto en más de 140 idiomas. Esto permite que el modelo, tras un fine-tune en inglés, conserve la capacidad de operar sobre preguntas médicas en otros idiomas (como el español), sin colapso del idioma y habilitando el eje de evaluación translingüe.

## 3.2 Por qué la arquitectura Transformer es adecuada

Gemma-3-4B-it es un *transformer decoder-only autoregresivo*, la familia arquitectónica que domina actualmente la generación de texto (GPT, Claude, Llama, Gemma, etc.). Conviene revisar por qué esta arquitectura es particularmente apta para la tarea.

### Atención: el corazón del modelo

El componente central del Transformer es el *mecanismo de atención*. En cada capa, para cada posición de la secuencia, el modelo calcula qué tan relevantes son las posiciones anteriores y combina sus representaciones de forma ponderada. En términos prácticos: cuando el modelo va a predecir el siguiente token, tiene acceso completo y selectivo a todo lo dicho antes.

Esto es exactamente lo que se necesita para razonamiento clínico sobre viñetas largas: un caso puede tener un dato crítico (por ejemplo, "elevación del ST en II, III, aVF") al inicio de la viñeta y la pregunta clave varios cientos de tokens después. La atención permite "regresar" a ese dato con precisión cuando es necesario, capacidad ausente en arquitecturas recurrentes anteriores (LSTM, GRU).

Gemma 3 usa una variante eficiente llamada *Grouped-Query Attention (GQA)*, que reduce el consumo de memoria durante la generación sin pérdida significativa de calidad — útil cuando se generan cadenas de razonamiento extensas.

### Otros componentes

Otros componentes adecuados dentro de la arquitectura seleccionada:

- **RoPE (Rotary Position Embeddings)** para codificar la posición de cada token de forma que el modelo entienda el orden secuencial.
- **GeGLU**, una función de activación con compuerta usada en el MLP de cada bloque, que da más capacidad expresiva que activaciones simples como ReLU.
- **RMSNorm** para estabilizar el entrenamiento, una variante computacionalmente barata de la normalización tradicional.

### Capacidad generativa y espacio latente

El modelo termina con una proyección lineal sobre el vocabulario completo del tokenizador (~256Ki tokens en Gemma 3), produciendo logits que se convierten en una distribución de probabilidad sobre el siguiente token mediante softmax. La generación es autorregresiva: en cada paso se muestrea (según la temperatura, temperatura de cero para selección greedy) un token, se concatena al contexto y se repite.

Las representaciones ocultas del modelo constituyen un espacio latente. Este espacio fue moldeado durante el preentrenamiento masivo para codificar relaciones semánticas, sintácticas, factuales y multilingües del corpus de entrenamiento. Es en este espacio donde residirá el conocimiento médico latente del modelo: las asociaciones entre síntomas y diagnósticos, los mecanismos farmacológicos, los patrones de razonamiento clínico observados en los textos de preentrenamiento. El fine-tuning de dominio actúa precisamente sobre este espacio, reorganizando localmente las regiones relevantes sin reaprender la estructura general del lenguaje. Es por esta misma razón que el conocimiento aprendido en inglés puede transferirse a otros idiomas, donde el conocimiento sin importar el idioma se proyecta a regiones cercanas del mismo espacio conceptual.

## 3.3 Configuración del fine-tuning eficiente: QLoRA

Sobre el modelo base se aplica **_QLoRA_** (Dettmers et al. 2023), un método de fine-tuning eficiente en parámetros que combina dos técnicas independientes:

### Cuantización del modelo base a 4 bits

Los pesos del modelo base se almacenan en formato *NF4 (4-bit NormalFloat)*, una codificación de precisión reducida diseñada específicamente para distribuciones de pesos cercanas a la normal estándar (la distribución empírica de pesos de un transformer preentrenado). Cada peso original en bf16 (16 bits) se mapea al cuantil más cercano de una grilla de 16 valores derivada de la distribución normal. Adicionalmente se aplica doble cuantización: las constantes de escala de la cuantización se cuantizan a su vez, ahorrando ~0.4 bits por parámetro.

Para el forward pass, los pesos cuantizados se descomprimen sobre la marcha al formato de cómputo de adaptadores (Compute Dtype) para ejecutar las operaciones matriciales.

### Adaptadores LoRA

Los pesos cuantizados del modelo base se mantienen congelados durante todo el entrenamiento. Sobre las matrices de proyección lineal seleccionadas se inserta una descomposición de bajo rango: para una matriz original $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, se añaden dos matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ y $B \in \mathbb{R}^{d_{\text{out}} \times r}$ con $r \ll \min(d_{\text{in}}, d_{\text{out}})$. La salida modificada se calcula como:

$$h = Wx + \frac{\alpha}{r} \cdot BAx$$

Solo $A$ y $B$ son entrenables y están en bf16. El producto $BA$ representa una actualización de bajo rango al peso original, suficiente para capturar la adaptación de dominio sin necesidad de modificar todos los parámetros.

### Configuración propuesta para este proyecto

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Rango LoRA (`r`) | 16 | Punto medio en el rango de literatura (8–64); 16 ofrece capacidad suficiente para adaptación de dominio sin sobreparametrizar |
| Alpha (`α`) | 32 | Convención estándar `α = 2r` que escala las actualizaciones LoRA proporcionalmente al rango |
| Dropout LoRA | 0.05 | Regularización suave; la literatura reporta poca sensibilidad a este valor en el rango 0.0–0.1 |
| Cuantización | NF4 + double quant | Configuración recomendada por el paper original de QLoRA |
| Cómputo de adaptadores | bf16 | Coincide con la precisión nativa de Gemma 3 en TPU/GPU modernas |
| Módulos objetivo | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Aplicar LoRA a la atención completa (Q,K,V,O) y al MLP completo (gate, up, down) maximiza la cobertura sin entrenar embeddings ni LM head |

Aplicar LoRA a los siete módulos objetivo es la práctica actual recomendada por trabajos como el reporte oficial de PEFT y el paper de QLoRA.

---

## 3.4 Entradas, salidas y flujo del sistema

### Tokenizador

Gemma 3 utiliza un tokenizador *SentencePiece* con vocabulario de aproximadamente 256Ki tokens, entrenado sobre el corpus multilingüe de preentrenamiento. Es idéntico al tokenizador de MedGemma, lo que garantiza que los dos modelos comparados procesan la entrada con el mismo nivel de granularidad léxica.

### Chat template

Gemma 3 espera un formato de conversación delimitado por tokens especiales de turno:

```text
<start_of_turn>user
{contenido del mensaje del usuario}<end_of_turn>

<start_of_turn>model
{respuesta generada del modelo}
<end_of_turn>
```

Este es el formato que el modelo aprendió a seguir durante su instruction tuning original y el que se va a utilizar.

### Entrada del modelo

Para una pregunta MCQA, la entrada se construye así:

```text
<start_of_turn>user
You are a medical expert. Answer the following multiple-choice question
by reasoning step by step, then giving your final answer as a single letter.

Question: A 65-year-old man presents with crushing substernal chest pain
radiating to the left arm, diaphoresis, and nausea. ECG shows ST-segment
elevation in leads II, III, and aVF.

A) Pericarditis
B) Inferior wall myocardial infarction
C) Aortic dissection
D) Pulmonary embolism

Reason step by step, then provide your answer in the format "Answer: <letter>".
<end_of_turn>
```

Este texto se tokeniza, produciendo una secuencia de identificadores enteros que ingresan al modelo. La longitud típica está entre 200 y 500 tokens.

### Salida del modelo

A partir del último token de la entrada, el modelo genera autorregresivamente: en cada paso produce una distribución de probabilidad sobre los 256Ki tokens del vocabulario, se selecciona el token con la mayor probabilidad mediante *Greedy Decoding* (garantizando así determinismo y reproducibilidad, comportamiento requerido en evaluación médica), se añade al contexto, y se repite hasta encontrar el token `<end_of_turn>` o alcanzar `max_new_tokens`. La salida típica:

```text
<start_of_turn>model
Reasoning: Leads II, III, and aVF correspond to the inferior wall of the heart.
ST-segment elevation in these leads, combined with the clinical picture of
crushing chest pain, diaphoresis, and nausea, is the classic presentation of
an acute inferior wall myocardial infarction. Pericarditis would show diffuse
ST elevation across multiple leads. Aortic dissection typically presents with
tearing pain radiating to the back. Pulmonary embolism would show right heart
strain (S1Q3T3), not inferior ST elevation.

Answer: B
<end_of_turn>
```

### Flujo end-to-end del sistema

```text
┌───────────────────────────────────────────────────────────────┐
│  1. Pregunta cruda del dataset (MedQA / MedMCQA / HEAD-QA)    │
│     {question, options, correct_letter}                       │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│  2. Construcción del prompt aplicando el chat template        │
│     de Gemma con system instruction + question + options      │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│  3. Tokenización: SentencePiece → secuencia de input_ids      │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│  4. Forward pass autorregresivo                               │
│     ┌─────────────────────────────────────────────────────┐   │
│     │  Embedding lookup                                   │   │
│     │       │                                             │   │
│     │       ▼                                             │   │
│     │  N capas transformer con:                           │   │
│     │    - GQA + RoPE                                     │   │
│     │    - GeGLU MLP                                      │   │
│     │    - RMSNorm                                        │   │
│     │    - LoRA adapters en bf16 (entrenables)            │   │
│     │    - Pesos base en NF4 4-bit (congelados)           │   │
│     │       │                                             │   │
│     │       ▼                                             │   │
│     │  LM head → logits sobre vocabulario                 │   │
│     │       │                                             │   │
│     │       ▼                                             │   │
│     │  Greedy Decoding → token siguiente                  │   │
│     └─────────────────────────────────────────────────────┘   │
│     Predecir siguiente token (autoregresivo) hasta            │    │       <end_of_turn> o max_new_tokens                          │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│  5. Decodificación de tokens → texto                          │
│     "Reasoning: ... Answer: B"                                │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│  6. Extracción de respuesta vía regex sobre patrones          │
│     "Answer: ([A-D])" → letra predicha                        │
└───────────────────────────────────────────────────────────────┘
        

```

## 3.5 Alternativas consideradas y descartadas

| Alternativa | Razón de descarte |
|---|---|
| **Full fine-tuning de Gemma-3-4B** | Requiere >30 GB de VRAM solo para gradientes y estados del optimizador. Infactible en hardware disponible. |
| **LoRA sin cuantización (base en bf16)** | Solo el modelo base en bf16 podría ocupar 8 GB; sumando activaciones y demás, no es factible. |
| **Modelo base Llama-3.1-8B u otro de familia distinta** | Rompe la comparación causal contra MedGemma; las diferencias observadas no podrían ser atribuibles a la adaptación de dominio sino al modelo base. |
| **Modelo más pequeño (Gemma-3-1B)** | Capacidad insuficiente para razonamiento clínico complejo según evidencia consistente en la literatura. |
| **Modelo más grande (Gemma-3-27B)** | No cabe en hardware disponible ni con QLoRA agresivo. Requeriría hardware industrial. |

# 4. Estrategia de Datos

## 4.1 Datasets utilizados

El corpus de datos del proyecto se compone de dos datasets con roles diferenciados: uno para entrenamiento y uno exclusivamente para evaluación. Esta separación responde al objetivo 3 establecido en la sección anterior: medir transferencia entre estilos de examen médico (MedMCQA → MedQA).

| Dataset | Idioma | Origen | Cita | Licencia | Rol |
|---|---|---|---|---|---|
| **MedMCQA** | Inglés | Examen de admisión médica indio (AIIMS, NEET-PG) | Pal et al., CHIL 2022 | MIT | Entrenamiento + validación + evaluación in-distribution |
| **MedQA-USMLE-4options** | Inglés | Examen USMLE (Estados Unidos) | Jin et al., arXiv 2020 | MIT | Evaluación out-of-distribution (solo test) |

### MedMCQA

Dataset de preguntas tipo opción múltiple del examen de admisión a residencia médica en India. Contiene preguntas de 21 especialidades médicas (anatomía, fisiología, farmacología, patología, microbiología, medicina interna, cirugía, ginecología, pediatría, etc.) con cuatro opciones de respuesta. Una fracción mayoritaria de las preguntas incluye un campo `explanation` con la justificación de la respuesta correcta extraída del libro de texto de referencia, lo que permite usarlo como señal de cadena de razonamiento (CoT) sin necesidad de generación sintética.

**Razones de elección como dataset de entrenamiento:**
1. Es uno de los datasets MCQA médico más grande disponible públicamente, y es utilizado como benchmark de referencia citado en el reporte técnido de MedGemma.
2. Incluye explicaciones nativas escritas por humanos, evitando depender de un teacher LLM externo para generar CoT.
3. Cobertura amplia y balanceada de especialidades.
4. Licencia MIT, permite uso académico sin restricciones.

### MedQA-USMLE-4options

Dataset de preguntas tipo USMLE, el examen de licenciatura médica estadounidense. La variante "4-options" reduce las preguntas originales de 5 a 4 opciones, alineando el formato con MedMCQA y permitiendo evaluar con un parser unificado.

**Razones de elección como evaluación out-of-distribution y no como entrenamiento:**
1. Mantiene su rol como benchmark de referencia citado en el reporte técnico de MedGemma.
2. Permite medir transferencia entre estilos de examen: las viñetas USMLE tienden a ser más largas y con razonamiento clínico más elaborado que las preguntas indias de admisión.
3. Su tamaño compacto cabe íntegro dentro del cap de evaluación.

A diferencia de MedMCQA, el dataset original de MedQA no provee desglose por especialidad médica como campo estructurado. Por consiguiente, la evaluación sobre MedQA se realiza únicamente como accuracy global sobre la respuesta correcta, sin desglose por especialidad. El análisis por categoría se reporta exclusivamente sobre MedMCQA.

---

## 4.2 Preprocesamiento

El preprocesamiento se aplica a los datasets completos antes de definir cualquier subconjunto del experimento. El objetivo es producir, para cada dataset, un corpus limpio, libre de duplicados internos, libre de leakage cruzado y sin tokens fuera de vocabulario. Los subconjuntos de entrenamiento y evaluación se construyen posteriormente en §4.5 sobre el resultado de este preprocesamiento.

Las operaciones se ejecutan en el orden: filtrado → deduplicación → verificación de leakage → sanity checks. El resumen consolidado se reporta en §4.2.5.

### 4.2.1 Filtrado de calidad

Los siguientes filtros se aplican a los datasets completos:

| Filtro | Acción | Motivo | Aplica a MedMCQA | Aplica a MedQA |
|---|---|---|---|---|
| Pregunta o opciones vacías/null | Descartar | Datos malformados | ✓ | ✓ |
| Campo de respuesta correcta fuera del rango {0,1,2,3} | Descartar | Etiqueta corrupta | ✓ | ✓ |
| Pregunta con menos de 5 tokens | Descartar | Probable error de extracción | ✓ | ✓ |
| Sin campo `exp` (sin explicación) | Descartar | Sin señal de CoT, inconsistente con el resto del corpus de entrenamiento | ✓ | ✗ |
| Idioma detectado distinto a inglés (vía `langdetect`) | Descartar | Posible contaminación de otros idiomas | ✓ | ✓ |

El campo `exp` aplica únicamente a MedMCQA porque solo se usa para entrenamiento, donde la explicación es requerida como señal de CoT. MedQA se usa exclusivamente para evaluación (la cadena de razonamiento la genera el modelo en inferencia, no la consume del dataset), por lo que la ausencia de `exp` en MedQA es esperada y no constituye motivo de descarte.

### 4.2.2 Deduplicación

Se eliminan los ejemplos con preguntas exactamente duplicadas dentro de cada dataset, conservando una sola copia de cada pregunta única. La detección se realiza mediante hash MD5 del campo `question` normalizado (lowercased, sin puntuación, espacios colapsados). La deduplicación se aplica una vez sobre la totalidad de cada dataset (MedMCQA y MedQA test) tras el filtrado de §4.2.1.

### 4.2.3 Verificación de no-leakage

Tras la deduplicación interna, se verifica que no haya overlap entre los dos datasets utilizados en el experimento:

- **MedMCQA ∩ MedQA test**: comparación de hashes entre todas las preguntas limpias de MedMCQA y todas las preguntas limpias de MedQA test, para detectar duplicados cruzados (los datasets provienen de exámenes distintos pero podrían existir preguntas reusadas de bancos comunes).

Cualquier overlap encontrado se elimina del lado de MedQA (no de MedMCQA), priorizando la integridad del conjunto de evaluación.

### 4.2.4 Sanity checks automatizados

Sobre los corpus ya filtrados, deduplicados y libres de leakage se ejecutan controles automatizados que informan las decisiones de §4.4 (formateo) y verifican la salud del corpus.

#### Distribución de longitudes (tokens)

Se computa el histograma de longitudes en tokens (post-tokenización con el tokenizador de Gemma 3) de los ejemplos formateados completos. Esta distribución informa la elección de `max_length` en §4.4.4.

> ![Distribución de longitudes](../reports/figures/eda/length_distribution.png)
> *Figura 4.1 — Distribución de longitudes (tokens) de ejemplos formateados en MedMCQA y MedQA tras filtrado.*

| Percentil | MedMCQA (tokens) | MedQA (tokens) |
|---|---|---|
| p50 (mediana) | _201_ | _256_ |
| p90 | _380_ | _376_ |
| p95 | _477_ | _425_ |
| p99 | _768_ | _503_ |
| máximo | _4,901_ | _979_ |

#### Distribución de letras correctas

Verifica que la distribución A/B/C/D no esté sesgada. 

> ![Balance A/B/C/D en MedMCQA y MedQA](../reports/figures/eda/letter_balance.png)
> *Figura 4.2 — Distribución de letras correctas en MedMCQA y MedQA limpios.*

#### Distribución de especialidades

Se reporta la frecuencia de cada una de las 21 especialidades médicas en el corpus limpio de MedMCQA. Este desglose informa el muestreo estratificado en §4.5 y el análisis de accuracy por especialidad en Evaluación. MedQA no dispone de esta información (ver §4.1) y por tanto no aparece en este desglose.

> ![Distribución de especialidades en MedMCQA](../reports/figures/eda/medmcqa_specialty_distribution.png)
> *Figura 4.3 — Distribución de las 21 especialidades en MedMCQA limpio.*

#### Tokens fuera de vocabulario

Se verifica que el tokenizador de Gemma 3 no produce tokens desconocidos (`<unk>`) sobre los corpus. Dado que Gemma 3 usa un vocabulario SentencePiece de ~256k tokens entrenado sobre corpus multilingüe masivo, la tasa esperada de OOV es efectivamente cero. En caso de corrupción confirmada, se descartan.

| Métrica | MedMCQA | MedQA |
|---|---|---|
| Tokens totales en el corpus | _29,579,163_ | _342,494_ |
| Tokens `<unk>` detectados | _0_ | _0_ |
| Porcentaje de OOV | _0%_ | _0%_ |
| Ejemplos descartados por corrupción | _0_ | _0_ |

### 4.2.5 Resumen del corpus consolidado

Tabla consolidada que reporta la reducción del corpus a través de las operaciones del preprocesamiento (filtrado, deduplicación, leakage, OOV).

| Etapa | MedMCQA | MedQA `test` |
|---|---|---|
| Corpus crudo | _187,005_ | _1,273_ |
| Tras filtrado de calidad | _145,206_ | _1,273_ |
| Tras deduplicación | _124,388_ | _1,273_ |
| Tras eliminación de leakage cruzado | _124,388_ | _1,273_ |
| Tras eliminación de OOV (si aplica) | _124,388_ | _1,273_ |
| **Corpus final** | **_124,388_** | **_1,273_** |
| Pérdida total respecto al crudo (%) | _33.48%_ | _0.0%_ |

 Para MedMCQA el corpus combina las particiones `train` y `validation` provistas originalmente por los autores del dataset; las etiquetas de partición se preservan internamente en cada ejemplo y se utilizan en §4.5 para construir los subconjuntos del experimento.

---

## 4.3 Estadísticas descriptivas

Las estadísticas siguientes describen el corpus tras el preprocesamiento de §4.2. Son las cifras que efectivamente alimentan los splits de entrenamiento y evaluación.

| Estadística | MedMCQA (post-preprocesamiento) | MedQA-4opt (test post-preprocesamiento) |
|---|---|---|
| Preguntas totales | _124,388_ | _1,273_ |
| Longitud media de la pregunta (tokens) | _19.11_ | _174.03_ |
| Longitud media de cada opción (tokens) | _4.29_ | _5.66_ |
| Longitud media de la explicación (tokens) | _120.01_ | N/A |
| Cobertura de categorías | 21 especialidades médicas | No disponible |
| Balance de letras correctas (A/B/C/D) | _28.79%, 26.05%, 23.33%, 21.84%_ | _27.73%, 24.27%, 27.18%, 20.81%_ |

**Notas:**

- Las longitudes se reportan en tokens del tokenizador SentencePiece de Gemma 3, no en palabras, para consistencia con la unidad usada durante entrenamiento e inferencia.
- "Post-preprocesamiento" significa después de filtrado, deduplicación, eliminación de leakage y eliminación de OOV (§4.2). Es el corpus efectivo del que se sortean los splits en §4.5.
- Sobre el balance de opciones correctas: El ligero desbalance observado en la distribución de respuestas (con una leve preponderancia hacia la opción 'A' y menor frecuencia en la 'D') no representa un problema metodológico crítico. Dado que el fine-tuning penaliza la entropía cruzada de toda la cadena de razonamiento (CoT) y no solo la clasificación final, el modelo es forzado a aprender la inferencia clínica subyacente en lugar de explotar heurísticas o sesgos estadísticos de posición

---

## 4.4 Formateo y tokenización

Una vez producido el corpus limpio (post-§4.2), cada ejemplo se transforma en un par prompt-respuesta listo para entrenar. Esta sección define las decisiones de formato y tokenización que constituyen el "lado del input/output" del modelo.

### 4.4.1 Construcción del prompt (turno del usuario)

```text
<bos><start_of_turn>user
You are a medical expert. Answer the following multiple-choice question
by reasoning step by step, then giving your final answer as a single letter.

Question: {question}

A) {opa}
B) {opb}
C) {opc}
D) {opd}

Reason step by step, then provide your answer in the format "Answer: <letter>".
<end_of_turn>
```

### 4.4.2 Construcción de la respuesta esperada (turno del modelo)

```text
<start_of_turn>model
Reasoning: {exp}

Answer: {letter_from_cop}
<end_of_turn>
```

Donde `{letter_from_cop}` se obtiene mapeando el índice numérico de la respuesta correcta (0,1,2,3) a la letra correspondiente (A,B,C,D).

### 4.4.3 Decisiones de formateo

- **Instrucción inline en el turno de usuario**: la consigna se incluye en el turno del usuario en lugar de un turno de sistema separado, porque Gemma no tiene un rol "system" formal en su chat template.
- **Formato de respuesta uniforme**: todos los ejemplos terminan con la línea `Answer: <letter>` para facilitar el parseo durante evaluación con un único regex.
- **Solo letras como identificador de opción**: no se usa numeración adicional, alineando con el formato de MedQA.

### 4.4.4 Política de truncamiento

La distribución de longitudes reportada en §4.2.4 informa el techo de truncamiento. Con un percentil 95 medido en _477_ tokens y un percentil 99 en _768_ tokens sobre el corpus de entrenamiento (MedMCQA), se establece:

- **`max_length = 1024` tokens**: techo aplicado durante la tokenización.
- **Estrategia**: `truncation="longest_first"`, aplicado preferentemente a la explicación (no a la pregunta ni a las opciones), porque la pregunta y las opciones son contenido crítico que el modelo debe ver íntegro para aprender la asociación correcta.
- **Ejemplos que excederían 1024 tokens**: se truncan según la política anterior (no se descartan), preservando pregunta + opciones + respuesta final.

**Justificación de 1024 como techo:**
- Cubre el ~99% de los ejemplos sin truncamiento después de realizar el análisis exploratorio.
- Es potencia de 2, lo que aprovecha alineamiento con kernels GPU.
- Subir a 2048 doblaría el costo computacional de la atención sin beneficio para la inmensa mayoría de los ejemplos.

**Tasa de truncamiento aplicada:**

| Métrica | Valor |
|---|---|
| Ejemplos truncados | _435_ |
| Porcentaje del corpus | _0.35%_ |

### 4.4.5 Padding

Las secuencias formateadas tienen longitud variable. Durante el entrenamiento se agrupan en batches, lo que obliga a igualar longitudes dentro de cada batch mediante tokens de relleno (*padding*). La política aplicada es padding dinámico al máximo de cada batch en lugar de padding estático al techo global `max_length = 1024`. La razón es eficiencia: cuando un batch contiene secuencias substancialmente más cortas que el techo, el padding estático desperdicia cómputo en posiciones que no aportan señal de aprendizaje, mientras que el padding dinámico solo rellena hasta lo necesario para alinear las secuencias del batch en cuestión.

Esta operación es propia del *runtime* de entrenamiento, no del preprocesamiento: el corpus que sale de la presente sección contiene secuencias sin padding. La implementación concreta del *data collator* responsable del padding, junto con la configuración fina se detalla en §5.

### 4.4.6 Etiquetado de labels (`-100`)

Durante el entrenamiento, la pérdida de cross-entropy debe calcularse únicamente sobre los tokens de la respuesta del modelo —cadena de razonamiento, letra final y token de fin de turno—, no sobre los tokens del prompt del usuario. La razón conceptual es que el prompt es contexto fijo dado al modelo; obligarlo a "predecir" su propio prompt no aporta señal útil y diluiría el gradiente entre miles de tokens de pregunta y opciones. Para lograr ese filtrado, los tokens del prompt se etiquetan con el valor especial `-100`, que la implementación de cross-entropy de PyTorch interpreta como "ignorar en el cálculo del loss".

Esta mecánica es propia del _loop_ de entrenamiento. El corpus que sale del preprocesamiento contiene los pares prompt-respuesta sin etiquetado de loss; el etiquetado con `-100` se aplica al construir cada batch en tiempo de entrenamiento detallado en §5.

### 4.4.7 Inspección manual

Se revisa una muestra aleatoria de 10 ejemplos del corpus formateado (post-tokenización, decodificados de vuelta a texto) para verificar:

- Coherencia entre pregunta, opciones y respuesta correcta.
- Calidad de la explicación como cadena de razonamiento.
- Formato consistente del prompt y la respuesta.
- Ausencia de artefactos de extracción (HTML residual, caracteres mal codificados).
- Aplicación correcta del chat template y tokens especiales.

La inspección se realiza en `notebooks/01_data_exploration.ipynb`. Los hallazgos cualitativos no se incluyen aquí — el reporte simplemente confirma que la inspección fue ejecutada y no se detectaron problemas mayores.

---

## 4.5 Splits del experimento

A partir del corpus limpio producido en §4.2, se construyen los subconjuntos definitivos que se usan para monitoreo durante el entrenamiento y validación, y la evaluación final. La construcción se realiza por muestreo estratificado sobre las particiones originales del dataset (`train`, `validation`, `test`).

### Pools disponibles tras preprocesamiento

| Pool | Origen | Tamaño disponible |
|---|---|---|
| MedMCQA `train` limpio | partición original `train` de MedMCQA tras §4.2 | _122,315_ |
| MedMCQA `validation` limpio | partición original `validation` de MedMCQA tras §4.2 | _2,073_ |
| MedQA `test` limpio | partición `test` de MedQA tras §4.2 | _1,273_ |

### Regla de muestreo

Para cada subconjunto se define un objetivo de tamaño. La regla aplicada es:

- Si el pool disponible excede el objetivo: se realiza muestreo estratificado al tamaño objetivo, conservando la distribución original de especialidades (en MedMCQA) o sin estratificación (en MedQA, que no dispone de etiqueta de categoría).
- Si el pool disponible no alcanza el objetivo: se utilizan todos los ejemplos disponibles, dejando registrado el tamaño efectivo en el reporte.


### Construcción de los subconjuntos

#### Conjunto de entrenamiento

Del pool MedMCQA `train` limpio (_122,315_ ejemplos disponibles) se extrae un subconjunto estratificado por especialidad con objetivo 50,000 ejemplos.

#### Conjunto de validación durante el entrenamiento

Del pool MedMCQA `validation` limpio (_2,073_ ejemplos disponibles) se extrae un primer subconjunto estratificado con objetivo 500 ejemplos (relación a 10% del conjunto de entrenamiento), destinado al monitoreo de val accuracy intermedia durante el entrenamiento.

#### Conjunto de evaluación in-distribution

Del mismo pool MedMCQA `validation` limpio, excluyendo los 500 del subconjunto anterior para evitar leakage entre monitoreo y evaluación final, se extrae el segundo subconjunto estratificado de 1,573 ejemplos (_2,073_ - _500_ de validación). Esta es la métrica final in-distribution comparable entre los tres modelos del experimento.

#### Conjunto de evaluación out-of-distribution

Del pool MedQA `test` limpio (_1,273_ ejemplos disponibles) se utilizan todos los ejemplos sin muestreo adicional. 

### Resumen final

| Conjunto  | Tamaño efectivo | Origen | Uso |
|---|---|---|---|
| Entrenamiento  | _5000_ | MedMCQA `train` limpio, muestreo estratificado | Fine-tuning QLoRA |
| Validación durante entrenamiento | _500_ | MedMCQA `validation` limpio, muestreo estratificado | Eval intermedia (cada N pasos) |
| Evaluación in-distribution | _1,573_ | MedMCQA `validation` limpio, muestreo estratificado disjunto | Métrica final in-distribution |
| Evaluación out-of-distribution | todos los disponibles | _1,273_ | MedQA `test` limpio íntegro | Métrica final out-of-distribution |


# 5. Entrenamiento