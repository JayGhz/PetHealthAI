# 🐶🐱 Proyecto: Detección Multimodal de Enfermedades en Mascotas (Perros y Gatos)

**Informe del Trabajo Parcial**
Curso: *Machine Learning (CC57)*
**Universidad Peruana de Ciencias Aplicadas (UPC)**

---

## 🎯 1. Objetivo del Proyecto

El sector veterinario enfrenta grandes desafíos en la detección temprana y precisa de enfermedades en animales de compañía.
Este proyecto tiene como finalidad **desarrollar un sistema multimodal de predicción temprana de enfermedades en perros y gatos**, combinando información **clínica estructurada (datos tabulares)** con **evidencia visual (imágenes de síntomas o lesiones)**.

El objetivo es **asistir a los profesionales veterinarios** en el diagnóstico, optimizar recursos y mejorar la calidad del cuidado animal.

---

## 📊 2. Datasets

Se empleó un enfoque **multimodal** basado en dos fuentes complementarias de datos provenientes de **Kaggle**:

1. **Dataset Clínico**
   Contiene registros de mascotas con variables como `species`, `breed`, `age`, `weight`, `symptoms` y `disease`.

   * **Fuente:** [Animal Disease Prediction Dataset – Kaggle]
   * Incluye principalmente perros con información clínica estructurada.

2. **Dataset de Imágenes**
   Contiene imágenes de perros y gatos etiquetadas con distintas condiciones de salud (p. ej. enfermedades dermatológicas o infecciones visibles).

   * **Fuente:** [Pet Disease Images Dataset – Kaggle]

---

## ⚙️ 3. Metodología y Enfoque Técnico

El desarrollo sigue un flujo de trabajo basado en las etapas de **preprocesamiento**, **análisis exploratorio**, **modelado** y **fusión multimodal**.

### 3.1. Adquisición y Preprocesamiento

**Datos clínicos (tabulares):**

* Limpieza y estandarización de unidades (ej. duración → días, temperatura → °C).
* Filtrado de registros para conservar únicamente perros y gatos.
* Aumento de datos mediante una función `generate_variation` que introduce variaciones controladas en variables numéricas (edad, peso, etc.).

**Datos visuales (imágenes):**

* Adquisición de imágenes mediante la API de Kaggle.
* Preprocesamiento: redimensionamiento a 224×224, normalización RGB y aumento de datos (rotaciones, flips, etc.).

---

### 3.2. Análisis Exploratorio (EDA)

Se realizaron visualizaciones y análisis estadísticos para comprender la estructura del conjunto de datos.
Principales hallazgos:

* El **Parvovirus** es la enfermedad más frecuente.
* Existe una correlación positiva entre **frecuencia cardíaca** y **temperatura corporal**.
* La mayoría de las enfermedades presentan una **duración promedio menor a 8 días**.

---

### 3.3. Modelado y Fusión Multimodal

El enfoque central utiliza **Deep Learning multimodal**, combinando información clínica y visual.

#### 🔹 Modelos Clínicos

* **Baselines:** Regresión Logística, Naive Bayes.
* **Modelos Avanzados:** Random Forest, XGBoost y una red neuronal (MLP) para generar *embeddings clínicos*.

#### 🔹 Modelos Visuales

* **Arquitecturas CNN:** uso de *Transfer Learning* con modelos preentrenados como **ResNet50** y **EfficientNet**.

#### 🔹 Fusión Multimodal

* Se concatenan los *embeddings* clínicos (MLP) y visuales (CNN).
* El vector resultante se ingresa a un **meta-modelo neuronal final** que combina ambas modalidades para realizar la predicción.

---

### 3.4. Interpretabilidad del Modelo

Para garantizar transparencia y confiabilidad se utilizan técnicas explicativas:

* **SHAP:** Análisis de impacto de las variables clínicas.
* **Grad-CAM:** Visualización de las regiones más relevantes en las imágenes procesadas por la CNN.

---

## 📈 4. Resultados Preliminares (Baseline)

Resultados obtenidos usando únicamente los datos clínicos:

| Modelo                    | Accuracy (Validación) | Accuracy (Test) | F1-score (Macro) |
| :------------------------ | :-------------------: | :-------------: | :--------------: |
| **Regresión Logística**   |          74%          |       74%       |       0.87       |
| **Naive Bayes Gaussiano** |          39%          |       39%       |         —        |

Estos resultados confirman que la **Regresión Logística** representa una línea base sólida, mientras que **Naive Bayes** presenta limitaciones.
Los modelos avanzados (XGBoost, Random Forest y redes neuronales multimodales) buscan **superar estos valores de referencia**.

---

## 🧩 5. Gestión del Proyecto

La gestión del proyecto se organiza mediante:

* **Control de Versiones:** Repositorio centralizado en *GitHub*.
* **Metodología SCRUM:** Gestión de backlog, tareas y sprints a través de un tablero *Kanban* en *Trello*.

---

