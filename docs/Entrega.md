Los modelos entrenados y los datos en todas las lenguas y con todas las configuraciones se encuentran disponibles en HuggingFace agrupados en una colección:  
https://hf.co/collections/MiguelGP-13/tfg

# Datos
A continuación se listan los conjuntos de datos generados y publicados en HuggingFace, junto con sus DOI.

## Disponibilidad de datos

### Lexicons
- Lexicons (todas las lenguas): https://doi.org/10.57967/hf/9171

### Instructivos
- Gallego: https://doi.org/10.57967/hf/9172  
- Asturiano: https://doi.org/10.57967/hf/9173  
- Aranés: https://doi.org/10.57967/hf/9174  

### Anotados
- Gallego: https://doi.org/10.57967/hf/9176  
- Asturiano: https://doi.org/10.57967/hf/9175  
- Aranés: https://doi.org/10.57967/hf/9177  

### Huecos
- Gallego: https://doi.org/10.57967/hf/9179  
- Asturiano: https://doi.org/10.57967/hf/9178  
- Aranés: https://doi.org/10.57967/hf/9180  

---

# Código

El código desarrollado en este Trabajo de Fin de Grado se organiza en dos partes principales:  
(i) notebooks de experimentación y pipeline, y  
(ii) un paquete Python reutilizable para la evaluación de modelos en lenguas minoritarias.

Todo el código está disponible en el repositorio de GitHub del TFG:  
https://github.com/MiguelGP-13/tfg

## Notebooks principales

- `1-datasetGeneration.ipynb`: generación de los distintos datasets utilizados en la evaluación (ortografía, huecos, instructivo, etc.), a partir de los corpus limpios y de los lexicones por lengua.  
- `2-LLM_Evaluation.ipynb`: evaluación de modelos base (sin afinado) sobre las tareas definidas en el TFG, utilizando el paquete `lowresource_llm_evaluation`.  
- `3-Train.ipynb`: entrenamiento de modelos mediante QLoRA para las distintas lenguas y configuraciones (concatenado, instructivo, etc.).  
- `3.1-GenerateTrainDatasets.ipynb`: generación de datos específicos para entrenamiento (pares instructivos, ejemplos concatenados, etc.).  
- `3.2-Trained_LLM_Evaluation.ipynb`: evaluación de los modelos ya entrenados (QLoRA) sobre los mismos conjuntos de evaluación que los modelos base.  
- `4-exploreResults.ipynb`: análisis y exploración de resultados, generación de tablas y figuras utilizadas en la memoria.  
- `5-UploadModels.ipynb`: subida de los modelos entrenados y artefactos asociados a HuggingFace.  
- `Utilidades.ipynb`: utilidades para descarga y preparación de corpus, así como tareas auxiliares de preprocesado.

Además, se han utilizado dos notebooks adicionales para análisis preliminares de audio:

- `initial_audio_analysis.ipynb`: análisis con el dataset inicial.  
- `expanded_audio_analysis.ipynb`: análisis con el dataset extendido.

## Paquete Python: lowresource_llm_evaluation

Dentro del repositorio se incluye el paquete Python `lowresource_llm_evaluation`, que implementa la lógica reutilizable de limpieza de corpus, generación de datasets sintéticos y evaluación de modelos:

- `LanguageDatasets.py`: módulo central de limpieza y gestión de corpus por lengua (carga, filtrado, normalización y particionado).  
- `generateDataset.py`: generación de datasets sintéticos para las distintas tareas (ortografía, huecos, instructivo, etc.).  
- `traduccion.py`: métricas y pipeline de evaluación de traducción.  
- `vocabulario.py`: evaluación de tareas de huecos (cloze) y vocabulario.  
- `ortografica.py`: evaluación ortográfica y detección de errores.  
- `interferenciaLinguistica.py`: métricas de interferencia lingüística entre lenguas cercanas.  
- `exploreResults.py`: funciones para explorar, agregar y visualizar resultados de evaluación.  
- `utils.py`: utilidades generales (carga de configuraciones, manejo de rutas, helpers de evaluación).  
- `constants/`: constantes compartidas (códigos de idioma, prompts, estilos de salida, etc.).

El paquete incluye además:

- `README.md`: descripción del paquete, instalación y ejemplos de uso.  
- `pyproject.toml`: metadatos del paquete, dependencias y configuración de construcción.

La documentación adicional relacionada con el código (instrucciones de ejecución, requisitos y descripción de resultados) se encuentra en el directorio `docs/` del repositorio, que incluye:

- `docs/README.md`  
- `docs/RESULTS.md`  
- `docs/requirements.txt`

---

# Modelos

Los modelos entrenados en todas las lenguas y configuraciones se encuentran disponibles en HuggingFace:

## Modelos publicados

- Asturiano: https://doi.org/10.57967/hf/9185  
- Asturiano concatenado: https://doi.org/10.57967/hf/9183  
- Asturiano concatenado instructivo: https://doi.org/10.57967/hf/9186  

- Gallego: https://doi.org/10.57967/hf/9181  

- Aranés: https://doi.org/10.57967/hf/9182  
- Aranés instructivo: https://doi.org/10.57967/hf/9184  


---
