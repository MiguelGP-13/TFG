# TFG — Evaluación y Mejora de Modelos de Lenguaje en Lenguas Minoritarias

Este repositorio contiene el código, experimentos, datasets y resultados del Trabajo de Fin de Grado dedicado a **evaluar y mejorar modelos de lenguaje (LLMs) en lenguas minoritarias**, con un enfoque aplicado y reproducible.  
El proyecto combina:

- preparación y limpieza de corpus reales  
- generación de datasets sintéticos  
- evaluación multicomponente de modelos  
- análisis cuantitativo y cualitativo  
- visualización avanzada de resultados  
- experimentos reproducibles en notebooks  

El objetivo es estudiar el comportamiento de modelos modernos en lenguas con pocos recursos y proponer herramientas prácticas para su mejora.

---

## 1. Estructura del repositorio

```
TFG/
│
├── notebooks/                         # Notebooks principales del TFG
├── 1-datasetGeneration.ipynb          # Generación de datasets (ortografía, huecos, instructivo…)
├── 2-LLM_Evaluation.ipynb             # Evaluación de modelos base
├── 3-Train.ipynb                      # Entrenamiento QLoRA
├── 3.1-GenerateTrainDatasets.ipynb    # Generación de datos para entrenamiento
├── 3.2-Trained_LLM_Evaluation.ipynb
├── 4-exploreResults.ipynb             # Análisis de resultados
├── 5-UploadModels.ipynb               # Subida de modelos a HF
├── Utilidades.ipynb                   # Descarga de corpus
│
├── lowresource_llm_evaluation/        # Paquete Python desarrollado en el TFG
│   ├── lowresource_llm_evaluation/
│   │   ├── LanguageDatasets.py        # Módulo central de limpieza y gestión de corpus
│   │   ├── generateDataset.py         # Generación de datasets sintéticos
│   │   ├── traduccion.py              # Evaluación de traducción
│   │   ├── vocabulario.py             # Evaluación de huecos
│   │   ├── ortografica.py             # Evaluación ortográfica
│   │   ├── interferenciaLinguistica.py# Métricas de interferencia
│   │   ├── exploreResults.py          # Exploración de resultados
│   │   ├── utils.py                   # Utilidades
│   │   └── constants/                 # Códigos de idioma, prompts, estética
│   ├── README.md
│   └── pyproject.toml
│
│
├── results/                           # Resultados de modelos base
│   ├── Raw/                           # JSON originales
│   ├── Processed/                     # HTML/LaTeX procesados
│   └── Figures/                       # Figuras del TFG
│
├── results_lora/                      # Resultados de modelos QLoRA
│   ├── Raw/
│   ├── Processed/
│   └── Figures/
│
├── Memoria/                           # Figuras utilizadas en el documento del TFG
│
├── docs/                              # Documentación auxiliar
│   ├── README.md
│   ├── RESULTS.md
│   └── requirements.txt
│
└── lexicons/                          # Lexicones por lengua para métricas
```

## 2. Objetivo del proyecto

El TFG aborda un problema central en PLN:  
**la falta de herramientas, corpus y benchmarks para lenguas minoritarias**.

El trabajo propone:

1. Un pipeline reproducible para preparar corpus reales.  
2. Un conjunto de tareas de evaluación adaptadas a lenguas low‑resource.  
3. Métodos para generar datos sintéticos útiles para entrenamiento.  
4. Un análisis comparativo del rendimiento de varios LLMs.  

El proyecto se centra en lenguas como asturiano, aragonés o gallego, pero es extensible a cualquier lengua con escasez de recursos.

---

## 3. Preparación de corpus: `LanguageDataset`

El módulo `LanguageDataset` implementa un sistema completo para:

- cargar datos desde Tatoeba, OPUS, carpetas locales, listas o dataframes  
- limpiar texto con reglas avanzadas  
- anonimizar datos sensibles  
- filtrar por idioma usando FastText LID‑176  
- concatenar líneas para evitar fragmentación  
- tokenizar y dividir en train/test  
- obtener estadísticas de tokens para elegir `max_length`  

Este módulo es la base del pipeline de datos del TFG y se usa en todas las fases posteriores.

---

## 4. Generación de datasets sintéticos

El proyecto incluye herramientas para generar datasets útiles para entrenar o evaluar modelos:

### 4.1. Dataset ortográfico  
Frases con errores sintéticos (ortográficos, léxicos, reordenación, etc.).

### 4.2. Dataset ortográfico anotado  
Errores marcados con etiquetas XML `<err t=...>...</err>`.

### 4.3. Dataset de huecos  
Frases con `<mask>` y palabra objetivo.

### 4.4. Dataset instructivo  
Pares `<|user|> ... <|assistant|>` generados a partir de plantillas por idioma.

### 4.5. Dataset QA instructivo  
Generación automática de preguntas y respuestas.

Todos los generadores incluyen reintentos automáticos y control de errores.

---

## 5. Benchmark de evaluación

El benchmark evalúa modelos en **cinco tareas lingüísticas**, diseñadas para capturar distintos aspectos del rendimiento en lenguas minoritarias:

### 5.1. Calidad de lengua  
Generación libre evaluada con:
- TTR  
- entropía  
- solapamiento de n‑gramas  
- frecuencia de vocabulario objetivo  
- comparación con otras lenguas  

### 5.2. Traducción directa  
Métricas:
- BLEU  
- chrF  

### 5.3. Round‑trip translation  
Traducción ida‑y‑vuelta pasando por lenguas intermedias.

### 5.4. Vocabulario (huecos)  
Predicción de palabras faltantes:
- accuracy  
- accuracy_lower  
- Levenshtein  

### 5.5. Ortografía  
Corrección de errores anotados:
- BLEU  
- chrF  
- Levenshtein  
- precisión, recall, F1  
- errores corregidos / no corregidos / nuevos  

El benchmark devuelve un JSON estructurado con métricas globales y ejemplos por tarea.

---

## 6. Visualización y análisis de resultados

El repositorio incluye herramientas para generar informes en varios formatos:

### Consola  
`pretty_print_results`  
Informe formateado con colores ANSI y tablas.

### HTML  
`generate_html_report`  
Informe interactivo con tablas comparativas y ejemplos paralelos.

### LaTeX  
`generate_latex_snippet_completo`  
`generate_latex_snippet_compacto`  
Fragmentos para papers y memoria del TFG.

### Gráficas  
`generar_grafico_barras`  
Comparación de modelos por lengua y métrica usando Seaborn o Plotly.

---

## 7. Reproducibilidad

El repositorio incluye:

- notebooks con todos los experimentos  
- scripts de evaluación  
- datasets generados  
- resultados guardados  
- figuras y comparativas  
- memoria del TFG  

Todo el pipeline puede ejecutarse de forma reproducible siguiendo los notebooks.

---

## 8. Instalación

```
git clone https://github.com/MiguelGP-13/TFG
cd TFG
pip install -r requirements.txt
```

---

## 9. Documento del TFG

La memoria completa del proyecto está disponible en:

```
[docs/Memoria.pdf](docs/Memoria.pdf)
```

Incluye:
- motivación  
- objetivos  
- metodología  
- experimentos  
- resultados  
- conclusiones  
- trabajo futuro  

---

## 10. Autor

Miguel Gómez Prieto  
Grado en Ciencia de Datos e Inteligencia Artificial
Universidad Politécnica de Madrid

