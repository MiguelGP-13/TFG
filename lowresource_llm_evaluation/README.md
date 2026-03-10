Librería para TFG de Miguel Gómez, conteniendo métricas para evaluar LLMs sin datasets de Q&A

### 2. Traducción como tarea de evaluación

La traducción gallego ↔ español, asturiano ↔ español y aranés ↔ español permite medir comprensión y generación. Para el aranés puede ser útil incluir también comparaciones con francés, dado su parentesco occitano. 

### 3. Corrección gramatical mediante introducción de errores

Para evaluar la capacidad de corrección, se parte de textos correctos en asturiano o aranés (por ejemplo, de Wikipedia). Un LLM grande introduce errores controlados de ortografía, morfología o sintaxis. El modelo evaluado debe corregirlos. La comparación con el texto original permite medir la calidad de la corrección.

- ```"llama-3.1-8b-instant"``` Mucho más rápido, pruebas
- ```"openai/gpt-oss-120b"``` Para dataset final
- ```"qwen/qwen3-32b"``` Probar

### 4. Medición de castellanización o interferencia lingüística

Para detectar si el modelo mezcla castellano con asturiano o aranés
#### 4.1. Índice tipo‑token (TTR)

Se calcula como número de palabras únicas dividido entre el total de palabras. Un TTR bajo puede indicar uso excesivo de vocabulario castellano básico.
#### 4.2. Entropía léxica

Se calcula la distribución de frecuencias de los tokens y su entropía. Una entropía baja sugiere un vocabulario poco variado y potencial castellanización.
#### 4.3. Frecuencia relativa de formas propias

Se construye un lexicón asturiano o aranés a partir de corpus públicos. Se compara la proporción de tokens generados por el modelo que pertenecen al lexicón propio frente a un lexicón castellano o francés (en el caso del aranés). Esto permite medir interferencia.
#### 4.4. N‑gram overlap con corpus de referencia

Se toma un corpus real en asturiano o aranés (por ejemplo, Wikipedia). Se extraen sus n‑gramas y se comparan con los n‑gramas generados por el modelo. El modelo no recibe nada en esta fase; simplemente se analizan sus salidas. Un solapamiento bajo indica que el modelo no reproduce patrones característicos de la lengua.
