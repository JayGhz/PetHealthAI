# Proyecto: Detección Multimodal de Enfermedades en Mascotas (Perros y Gatos)

[cite_start]Informe del Trabajo Parcial para el curso de **Machine Learning (CC57)** de la **Universidad Peruana de Ciencias Aplicadas (UPC)**. [cite: 1, 2, 3, 4]

## 1. Integrantes

* [cite_start]Jhair Armando Quispe Marca (u20211c699) [cite: 8]
* [cite_start]Andre Angel Chipana Rios (u202220230) [cite: 8]
* [cite_start]Maria Ximena Chavarria Barrios (u202323166) [cite: 8]

## 2. Objetivo del Proyecto

[cite_start]El sector veterinario enfrenta desafíos en la detección temprana y precisa de enfermedades en animales de compañía. [cite: 17] [cite_start]Este proyecto busca desarrollar un sistema multimodal para la predicción temprana de enfermedades en perros y gatos. [cite: 484]

[cite_start]El sistema integra **información clínica estructurada** (datos tabulares) con **evidencia visual** (imágenes de síntomas o lesiones) para asistir a los profesionales veterinarios en la toma de decisiones, optimizar recursos y mejorar el cuidado de los pacientes. [cite: 24, 25, 484, 485]

## 3. Datasets

[cite_start]Se utilizó un enfoque multimodal empleando dos conjuntos de datos complementarios de Kaggle: [cite: 31, 42]

1.  [cite_start]**Dataset Clínico:** Proporciona registros clínicos de mascotas (principalmente perros) con variables como `species`, `breed`, `age`, `weight`, `symptoms` y `disease`. [cite: 33, 36]
    * [cite_start]**Fuente:** Kaggle [cite: 36] (Ref: Animal Disease Prediction Dataset) [cite_start][cite: 547]

2.  [cite_start]**Dataset de Imágenes:** Consiste en imágenes de perros y gatos etiquetadas con distintas condiciones de salud, como enfermedades dermatológicas, infecciones y otras afecciones visibles. [cite: 38, 39]
    * [cite_start]**Fuente:** Kaggle [cite: 41] (Ref: Pet Disease Images Dataset) [cite_start][cite: 548]

## 4. Metodología y Enfoque Técnico

El proyecto sigue un flujo de trabajo que incluye preprocesamiento, análisis exploratorio, ingeniería de características y modelado.

### 4.1. Adquisición y Preprocesamiento

* **Datos Tabulares (Clínicos):**
    * [cite_start]Limpieza y estandarización de unidades (ej. duración a días [cite: 117][cite_start], temperatura a °C [cite: 116]).
    * [cite_start]Filtrado de datos para conservar únicamente registros de perros y gatos. [cite: 120]
    * [cite_start]Implementación de **aumento de datos tabulares** (`generate_variation`) para expandir el dataset introduciendo variaciones controladas en variables numéricas (edad, peso, etc.). [cite: 103, 104, 111]
* **Datos de Imágenes:**
    * [cite_start]Adquisición de imágenes mediante la API de Kaggle. [cite: 53]
    * [cite_start]Preprocesamiento estándar: redimensionamiento a 224x224, normalización RGB y aumento de datos (rotaciones, flips, etc.). [cite: 556, 603, 604]

### 4.2. Análisis Exploratorio de Datos (EDA)

[cite_start]Se realizaron múltiples visualizaciones para comprender la estructura y patrones en los datos. [cite: 132] Algunos hallazgos clave incluyen:

* [cite_start]El **Parvovirus** es la enfermedad más frecuente en el conjunto de datos clínicos. [cite: 150]
* [cite_start]Se identificó una fuerte correlación positiva entre la frecuencia cardíaca y la temperatura corporal. [cite: 235]
* [cite_start]La mayoría de las enfermedades se concentran en los primeros 8 días de duración. [cite: 370]

### 4.3. Modelado y Fusión Multimodal

[cite_start]La técnica central es un enfoque de Deep Learning multimodal que combina ambas fuentes de datos: [cite: 514]

1.  **Modelos Clínicos (Tabulares):**
    * [cite_start]*Baselines:* Regresión Logística y Naive Bayes. [cite: 517, 627]
    * [cite_start]*Avanzados:* Random Forest, XGBoost [cite: 519] [cite_start]y una red neuronal (MLP) para generar *embeddings* clínicos. [cite: 520]

2.  **Modelos Visuales (Imágenes):**
    * [cite_start]Se emplean Redes Neuronales Convolucionales (CNN). [cite: 522]
    * [cite_start]Se utiliza *Transfer Learning* con arquitecturas pre-entrenadas como ResNet50 y EfficientNet. [cite: 524]

3.  **Fusión Multimodal:**
    * [cite_start]Los *embeddings* generados por el MLP (clínico) y la CNN (visual) se concatenan. [cite: 531, 611]
    * [cite_start]Estos vectores combinados se pasan a un meta-modelo neuronal final que integra ambas modalidades para la predicción. [cite: 532, 612]

### 4.4. Interpretabilidad

Para garantizar la transparencia del modelo, se planea utilizar:
* [cite_start]**SHAP:** Para explicar el impacto de las variables clínicas. [cite: 538, 623]
* [cite_start]**Grad-CAM:** Para visualizar qué regiones de una imagen son más relevantes para el diagnóstico de la CNN. [cite: 538, 624]

## 5. Resultados Preliminares (Baseline)

[cite_start]En los experimentos iniciales, utilizando **únicamente los datos clínicos**, se establecieron los siguientes modelos de referencia: [cite: 627]

| Modelo | Accuracy (Validación) | Accuracy (Test) | F1-score (Macro) |
| :--- | :---: | :---: | :---: |
| **Regresión Logística** | [cite_start]74% [cite: 649] | [cite_start]74% [cite: 649] | [cite_start]0.87 [cite: 649] |
| **Naive Bayes Gaussiano** | [cite_start]39% [cite: 650] | [cite_start]39% [cite: 650] | - |

[cite_start]Estos resultados confirman que la Regresión Logística sirve como una línea base sólida [cite: 666][cite_start], mientras que Naive Bayes mostró limitaciones. [cite: 650] [cite_start]El objetivo de los modelos avanzados (XGBoost, Random Forest) y el enfoque multimodal es superar este rendimiento. [cite: 667]

## 6. Gestión del Proyecto

La trazabilidad del proyecto se gestiona mediante:
* [cite_start]**Control de Versiones:** Repositorio centralizado en GitHub. [cite: 469]
* [cite_start]**Metodología SCRUM:** Gestión de tareas, backlog y sprints mediante un tablero Kanban en Trello. [cite: 477, 511]

## 7. Estructura del Repositorio (Sugerida)
