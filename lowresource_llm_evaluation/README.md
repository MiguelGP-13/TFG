# lowresource-llm-evaluation

Librería para trabajar con **lenguas minoritarias / low‑resource**, proporcionando:

1. Un módulo robusto para **preparar y limpiar corpus** (`LanguageDataset`).
2. Un **benchmark multicomponente** para evaluar modelos en tareas lingüísticas reales.
3. Herramientas para **generar datasets sintéticos** (ortografía, QA, huecos, instructivo).
4. Módulos de **exploración y visualización de resultados** (consola, HTML, LaTeX, gráficas).

El objetivo es ofrecer un marco reproducible para investigación y evaluación de LLMs en lenguas con pocos recursos.

---

## 1. LanguageDataset

`LanguageDataset` es el núcleo de la parte de **preprocesamiento**.

### Funcionalidades principales

- Carga de datos desde:
  - carpetas locales (`.txt`, `.csv`, `.json`)
  - Tatoeba (descarga automática)
  - OPUS monolingüe (streaming `.gz`)
  - listas de strings
  - dataframes de pandas
  - datasets de HuggingFace

- Limpieza avanzada:
  - normalización Unicode
  - eliminación de URLs, HTML, hashtags, emojis
  - control de repeticiones
  - filtrado por longitud mínima/máxima
  - anonimización (emails, teléfonos)
  - preservación de etiquetas especiales

- Filtrado por idioma con FastText LID‑176:
  - batch prediction
  - top‑k para lenguas minoritarias
  - reconstrucción de índices por dataset

- Preparación para entrenamiento:
  - tokenización en batch
  - división train/test
  - concatenación de líneas respetando el origen del dataset

- Estadísticas:
  - distribución de tokens
  - media, mediana, percentiles, moda aproximada

---

## 2. Benchmark

El benchmark evalúa un modelo en **cinco tareas lingüísticas**:

### 2.1. Calidad de lengua (generación)
Evalúa naturalidad y adecuación al idioma objetivo.

Métricas:
- TTR
- entropía léxica
- solapamiento de n‑gramas
- frecuencia de vocabulario objetivo
- comparación con lexicones de otras lenguas

### 2.2. Traducción directa
Traducción desde otra lengua hacia la lengua objetivo.

Métricas:
- BLEU
- chrF

### 2.3. Round‑trip translation
Traducción ida‑y‑vuelta pasando por lenguas intermedias.

Métricas:
- BLEU
- chrF

### 2.4. Vocabulario (huecos)
Predicción de palabras faltantes en frases enmascaradas.

Métricas:
- accuracy
- accuracy_lower
- Levenshtein

### 2.5. Ortografía (corrección)
Corrección de errores anotados.

Métricas:
- BLEU
- chrF
- Levenshtein
- precisión, recall, F1
- errores corregidos, no corregidos y nuevos

Cada tarea devuelve:
- ejemplos concretos
- medias agregadas
- métricas específicas

---

## 3. Generación de datasets sintéticos

La librería incluye generadores de datasets para entrenamiento supervisado:

### 3.1. Dataset ortográfico
`generateDatasetOrtografico`  
Genera frases con errores ortográficos, léxicos o gramaticales.

### 3.2. Dataset ortográfico anotado
`generateDatasetOrtograficoAnotado`  
Genera errores marcados con etiquetas XML `<err t=...>...</err>`.

### 3.3. Dataset de huecos
`generateDatasetHuecos`  
Crea frases con `<mask>` y la palabra faltante correspondiente.

### 3.4. Dataset instructivo
`generateInstructivoDataset`  
Genera pares `<|user|> ... <|assistant|> ...` siguiendo plantillas por idioma.

### 3.5. Dataset QA instructivo
`generateInstructivoQADataset`  
Genera pares pregunta‑respuesta en formato instructivo.

Todos los generadores permiten:
- control de número de ejemplos
- reintentos automáticos
- guardado opcional en CSV

---

## 4. Exploración y visualización de resultados

La librería incluye herramientas para **analizar y presentar resultados** del benchmark.

### 4.1. Consola
`pretty_print_results`  
Informe formateado con colores ANSI, tablas y ejemplos.

### 4.2. HTML
`generate_html_report`  
Informe interactivo con:
- tablas comparativas
- ejemplos paralelos por modelo
- leyendas de errores
- estilos integrados

### 4.3. LaTeX
`generate_latex_snippet_completo`  
`generate_latex_snippet_compacto`  
Generación de fragmentos LaTeX para papers.

### 4.4. Gráficas
`generar_grafico_barras`  
Comparación de métricas entre modelos y lenguas usando:
- Seaborn
- Plotly

### 4.5. Exploración estructural
`explorar_estructuras`  
Muestra un árbol de tareas y métricas disponibles en un JSON de resultados.

---

## Instalación

```
pip install lowresource-llm-evaluation
```

---

## Ejemplo mínimo

### Preparar un dataset

```python
from lowresource_llm_evaluation import LanguageDataset
from transformers import AutoTokenizer

ds = LanguageDataset(language="asturiano", initializeTatoeba=True)
ds.filter_by_language(0.7)
ds.summary()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
train, test = ds.split(tokenizer=tokenizer, max_length=512)
```

### Ejecutar el benchmark

```python
from lowresource_llm_evaluation import benchmark
resultados = benchmark(
    model=model,
    tokenizer=tokenizer,
    lang_eval="asturiano",
    df_textos=df_pareado,
    list_textos_round=list_textos,
    df_huecos=df_huecos,
    df_anotado=df_errores,
    lexicon_target=lex_ast,
    lexicons_comparison=lex_otros
)
```

### Explorar resultados

```python
from lowresource_llm_evaluation import pretty_print_results
pretty_print_results(resultados)
```

### Generar un dataset sintético

```python
from lowresource_llm_evaluation import generateDatasetHuecos
df_huecos = generateDatasetHuecos(ds)
```

---

## Licencia

MIT (o la que corresponda en tu repositorio).
