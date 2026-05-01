# TFG
TFG about LLM and minoritary spanish languages

## Herramientas
https://github.com/TALP-UPC/FreeLing/tree/master

## Pasos a realizar
### Memoria
- [x] Cambiar color de descripciones en gráfico
- [x] Como se genera, limpian los datos en 4.2 Materiales
- [x] De donde vienen los datos en 4.2 Materiales
- [x] Añadir pequeña explicacion de Bleue y chrF
- [ ] Añadir estadísticas de datasets elegidos y generados
- [x] Citar de donde vienen los lexicon -> FreeLing
- [x] Añadir figura en 4.2.2 (diagrama...)
- [x] Identificar posibles soluciones (PEFT, Full fine tuning, algo más?) => explicar olvido catastrófico
L A revisar
- [x] Actualizar en la memoria los modelos elegidos y por qué (después de elegir los modelos)
L A revisar
- [x] Escribir resultados en memoria
- [ ] En future work, añadir que para cortar bien igual sería mejor cortar con un llm (no me da el cómputo)
- [x] Revisar aspectos legales y eticos (8) y metodología (4.1)
- [ ] Redactar sección 1
- [ ] Poner estadísticas de los datos
- [ ] Revisar 6.2 y 6.3
- [ ] Continuar con 6.3
- [ ] Escribir 7.2 y 7.3
- [ ] Comentar pad token
- [x] Explicar bien a que capas vamos a aplicarlo
L Revisarlo
- [ ] Explicar las capas de los LLM en el Marco teórico 
- [x] Contar primer entrenamiento que devolvía siempre EOS al principio
- [ ] Poner que vamos a reocrtar en el percentil 95, para evitar tanto EOS

### Datos
- [ ] Crear dataset para Gallego, Asturiano y  Aranés/Occitano [Train]
- [x] Crear lexicons
- [x] Crear datasets anotados

### Benchmark
- [x] Elegir métricas (Benchmark debajo) => No he encontrado papers sobre esto
- [x] Implementar Benchmark
- [x] Probar Benchmark con Dataset Anotado

### Entrenamiento
- [x] Elegir modelos más grandes y ver que se pueden cuantizar y cargar
- [x] Crear esqueleto QLoRA entrenamiento
- [ ] Entrenar con LoRA sobre el mejor (o los 2 mejores?)

### Evaluación
- [x] Evaluar los modelos base Asturiano
- [x] Evaluar los modelos base Gallego
- [x] Evaluar los modelos base Aranés
- [ ] Evaluar resultados QLoRA Asturiano
- [ ] Evaluar resultados QLoRA Gallego
- [ ] Evaluar resultados QLoRA Aranés

## Benchmark para evaluar LLMs en gallego, asturiano y aranés

### 1. Evaluación mediante LLM como juez

Se emplea un modelo grande como evaluador externo. El procedimiento consiste en:

1. Generar una tarea (pregunta, resumen, traducción, clasificación, corrección, etc.).
2. Obtener la respuesta del modelo evaluado.
3. Pedir a un LLM más potente que valore la respuesta según criterios como adecuación, fidelidad, corrección lingüística y coherencia.

Aunque el evaluador no sea perfecto en estas lenguas, su consistencia permite comparar modelos de forma relativa.

### 2. Traducción como tarea de evaluación

La traducción asturiano ↔ español y aranés ↔ español permite medir comprensión y generación. Para el aranés puede ser útil incluir también comparaciones con francés, dado su parentesco occitano. La evaluación puede realizarse mediante:

- LLM como juez. (lo prefiero, para evitar sesgos de que haya generado el LLM grande)
- Métricas automáticas basadas en similitud semántica entre la traducción del modelo y una traducción de referencia generada por un LLM grande.

#### 2.1. Evaluación de consistencia intra‑modelo

Consiste en pedir al modelo que genere un texto en la lengua objetivo, lo traduzca al español, vuelva a traducirlo al asturiano/gallego/aranés, y comparar la versión final con la inicial.

### 3. Corrección gramatical mediante introducción de errores

Para evaluar la capacidad de corrección, se parte de textos correctos en asturiano o aranés (por ejemplo, de Wikipedia). Un LLM grande introduce errores controlados de ortografía, morfología o sintaxis. El modelo evaluado debe corregirlos. La comparación con el texto original permite medir la calidad de la corrección.

### 4. Medición de castellanización o interferencia lingüística

Para detectar si el modelo mezcla castellano con asturiano o aranés, se pueden aplicar métricas cuantitativas:

#### 4.1. Índice tipo‑token (TTR)

Se calcula como número de palabras únicas dividido entre el total de palabras. Un TTR bajo puede indicar uso excesivo de vocabulario castellano básico.

#### 4.2. Entropía léxica

Se calcula la distribución de frecuencias de los tokens y su entropía. Una entropía baja sugiere un vocabulario poco variado y potencial castellanización.

#### 4.3. Frecuencia relativa de formas propias

Se construye un lexicón asturiano o aranés a partir de corpus públicos. Se compara la proporción de tokens generados por el modelo que pertenecen al lexicón propio frente a un lexicón castellano o francés (en el caso del aranés). Esto permite medir interferencia.

#### 4.4. N‑gram overlap con corpus de referencia

Se toma un corpus real en asturiano o aranés (por ejemplo, Wikipedia). Se extraen sus n‑gramas y se comparan con los n‑gramas generados por el modelo. El modelo no recibe nada en esta fase; simplemente se analizan sus salidas. Un solapamiento bajo indica que el modelo no reproduce patrones característicos de la lengua.

