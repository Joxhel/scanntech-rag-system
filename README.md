# Scanntech RAG System - ISLR Technical Assistant

![Banner](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Stack](https://img.shields.io/badge/Stack-Llama3%20%7C%20LangChain%20%7C%20Streamlit-orange) ![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-red)

> **Repositorio oficial desarrollado para el Scanntech AI Engineer Challenge.**

## 🚀 Resumen del Proyecto

Este sistema transforma el libro *"An Introduction to Statistical Learning" (ISL)* en un asistente técnico inteligente. El objetivo principal fue construir un ecosistema que no solo responda preguntas técnicas con precisión, sino que ofrezca **transparencia total** sobre las fuentes consultadas y mantenga un **control de calidad riguroso** mediante métricas automatizadas.

A lo largo del desarrollo, implementé una arquitectura basada en tres pilares:
* **Recuperación Semántica Avanzada:** Uso de un flujo de dos pasos (Retrieval + Re-ranking) para asegurar que el modelo trabaje solo con la información más pertinente.
* **Observabilidad en Tiempo Real:** Integración de un panel de métricas RAGAS directamente en la interfaz para monitorear la salud del sistema.
* **Ingesta con Preservación Técnica:** Enfoque en la extracción limpia de fórmulas matemáticas y estructuras jerárquicas del PDF original.

---

## 📸 Evidencias del Sistema (Visual Showcase)

### 1. El Orquestador Central (CLI)
Desarrollé un punto de entrada único (`main.py`) que permite gestionar todo el ciclo de vida del dato: desde la ingesta inicial hasta la ejecución de benchmarks de evaluación.
![Menú Principal CLI](assets/cli_menu.png)

### 2. Dashboard de Usuario y Observabilidad
La interfaz en Streamlit prioriza la experiencia del usuario. El panel lateral muestra los promedios históricos de las métricas de calidad, permitiendo validar la confianza del sistema antes de iniciar el chat.
![Streamlit Dashboard](assets/sidebar_observability.gif)

### 3. Rendimiento y Métricas RAGAS
El sistema genera reportes visuales y descriptivos tras cada evaluación. Además de los benchmarks, el sistema es capaz de **analizar las interacciones reales de los usuarios almacenadas en logs**, calculando métricas de fidelidad mediante una comparativa directa entre el contexto recuperado y la respuesta generada, asegurando una mejora continua basada en datos de uso real.
![Métricas de Rendimiento](assets/ragas_metrics.png)

---

## 🏗️ Arquitectura y Preprocesamiento

El núcleo del proyecto reside en cómo se preparan y recuperan los datos técnicos. Aquí detallo las decisiones clave en `src/ingestion.py`, `src/query_rag.py` y `src/evaluator.py`:

### 1. Ingesta Inteligente con PyMuPDF4LLM
Para manejar la complejidad del libro ISLR, utilicé `PyMuPDF4LLM` combinado con lógica personalizada para:
* **Extracción de Fórmulas y LaTeX:** A diferencia de otros extractores, este flujo preserva la sintaxis matemática, permitiendo que el modelo comprenda las ecuaciones sin errores de caracteres extraños.
* **Análisis de Estructura Jerárquica:** El sistema mapea la Tabla de Contenidos (TOC) para inyectar metadatos de `Capítulo`, `Subcapítulo` y `Sección` en cada fragmento.
* **Filtrado de Ruido Semántico:** Se implementó una lógica para **eliminar el Índice Alfabético y la Tabla de Contenidos** del cuerpo del texto indexado. Esto evita que el motor de búsqueda recupere listas de temas sin contenido explicativo, mejorando drásticamente la precisión del contexto.
* **Limpieza Especializada:** Se eliminan ruidos de edición (DOIs, copyright) que suelen ensuciar los embeddings.

### 2. Recuperación de Dos Pasos (Two-Pass Retrieval)
La recuperación se diseñó en dos fases para garantizar la relevancia máxima del contexto:
* **Búsqueda Vectorial (Broad Search):** Recuperación inicial de 15 fragmentos usando `nomic-ai/nomic-embed-text-v1.5`.
* **Re-ranking Semántico (Deep Search):** Aplicación de un **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) para re-evaluar la relevancia de esos 15 fragmentos, filtrando cualquier contexto que no aporte valor real antes de enviarlo al LLM.
* **Umbral de Calidad:** Se aplica un filtro estricto de score. Si ningún fragmento supera este umbral, el sistema declara que no tiene información suficiente antes de arriesgarse a alucinar.

### 4. Prompt Engineering y Generación
El motor (`src/query_rag.py`) utiliza un protocolo de verificación robusto:
* **Estructura XML:** Los fragmentos se inyectan en etiquetas `<DOCUMENT>` con metadatos de página y capítulo para evitar confusiones de contexto.
* **Thought Process (CoV):** El prompt obliga al modelo a razonar antes de responder (identificar conceptos, verificar presencia en fragmentos y emitir citaciones obligatorias `[Page X]`).
* **Temperatura 0:** Configurada para máxima consistencia y fidelidad técnica (OllamaLLM).


### 3. Prompt Engineering y Generación
El motor (`src/query_rag.py`) utiliza un protocolo de verificación robusto:
* **Estructura XML:** Los fragmentos se inyectan en etiquetas `<DOCUMENT>` con metadatos de página y capítulo para evitar confusiones de contexto.
* **Thought Process (CoV):** El prompt obliga al modelo a razonar antes de responder (identificar conceptos, verificar presencia en fragmentos y emitir citaciones obligatorias `[Page X]`).
* **Temperatura 0:** Configurada para máxima consistencia y fidelidad técnica (OllamaLLM).

### 4. Evaluación (RAGAS Framework)
Para garantizar la fiabilidad de las métricas de **RAGAS**, implementé un flujo de evaluación:
* **Juez Especializado:** Se utiliza `llama3.1:8b` como juez evaluador por su capacidad superior para seguir instrucciones complejas en comparación con modelos más pequeños.
* **Sanitización de Datos:** Desarrollé una lógica que convierte bloques de código y fórmulas complejas en tokens simplificados (`[MATH_BLOCK]`) antes de la evaluación. Esto evita que el juez se distraiga con la sintaxis de LaTeX y se enfoque puramente en la fidelidad semántica de la respuesta.
* **Métricas Core:** El sistema mide continuamente *Faithfulness*, *Answer Relevancy*, *Context Precision* y *Context Recall*.

Se ejecutó un benchmark maestro sobre preguntas técnicas donde:

| Métrica | Resultado | Justificación Técnica |
| :--- | :--- | :--- |
| **Faithfulness** | **0.76** | Representa la capacidad del modelo para mantenerse fiel al contexto recuperado sin alucinar. |
| **Answer Relevancy** | **0.86** | Indica qué tan directa y útil es la respuesta para la consulta del usuario. |
| **Context Precision** | **0.87** | El Re-ranker posiciona los mejores datos en los primeros lugares de forma efectiva. |
| **Context Recall** | **0.81** | El sistema localiza la información necesaria en el 81% de los casos complejos del benchmark. |

---

## 📺 Demos de Interacción

### Interacción Técnica
El asistente explica conceptos estadísticos complejos citando la ubicación exacta en el libro para su verificación.
![Demo Pregunta Válida](assets/valid_query_demo.gif)

### Manejo de Preguntas Fuera de Dominio
El sistema identifica consultas que no pertenecen al dominio del libro (como cultura general), evitando alucinaciones y manteniendo el enfoque técnico.
![Demo Pregunta Inválida](assets/invalid_query_demo.gif)

---

## 🛠️ Guía de Instalación y Setup

### 1. Modelos de IA (Ollama)
Este sistema utiliza **Ollama** para la inferencia local. Asegúrate de tener instalados los siguientes modelos:
```bash
ollama pull llama3.2:3b  # Para el Chat (Velocidad)
ollama pull llama3.1:8b  # Para el Juez Evaluador (Razonamiento)
```

### 2. Entorno y Dependencias
El proyecto utiliza un archivo `requirements.txt` con versiones fijas para garantizar la reproducibilidad del entorno.
```bash
# Crear y activar ambiente (Recomendado Python 3.10 o 3.11)
python -m venv .venv
source .venv/bin/activate  # O .\\.venv\\Scripts\\activate en Windows

# Actualizar pip e instalar dependencias fijas
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
scanntech-rag-system/
├── assets/             # Imágenes y evidencias para el README
├── data/               # Documentos fuente (PDF del libro ISLR)
├── db/                 # Persistencia de ChromaDB (Pre-cargada)
├── eval/               
│   ├── benchmark/      # Ground Truth (QA pairs) para evaluación
│   ├── logs/           # Trazabilidad de interacciones (JSONL)
│   └── reports/        # Gráficos y CSV generados por RAGAS
├── src/                
│   ├── __init__.py
│   ├── config.py       # Single Source of Truth (Rutas, Modelos, Configuración)
│   ├── ingestion.py    # Pipeline ETL (Limpieza, TOC Hierarchy, Indexación)
│   ├── query_rag.py    # Motor RAG (Retrieval + Re-ranker + Chain of Verification)
│   └── evaluator.py    # Lógica de métricas RAGAS con sanitización de texto
├── app.py              # Interfaz de Usuario (Streamlit Dashboard)
├── main.py             # CLI Entrypoint (Orquestador)
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación oficial
```

---

## ⚡ Execution Flow

El proyecto cuenta con un `main.py` que centraliza todas las operaciones.

### Paso 1: Iniciar el Orquestador
Ejecuta el siguiente comando en tu terminal:

```bash
python main.py
```

### Paso 2: Seleccionar Operación
Verás un menú interactivo con las siguientes opciones:

1. **🛠️ INGESTA:** Procesa el PDF `GenAI Challenge.pdf` y crea/actualiza la base de datos vectorial en `db/chroma_db_storage`.
    * *Nota: Se incluye una versión pre-cargada de la DB en el repo para pruebas rápidas.*
2. **📊 EVALUACIÓN:** Ejecuta el benchmark de RAGAS. Compara las respuestas del sistema contra el `ground_truth.json` y genera un reporte en CSV.
3. **💬 CHAT:** Lanza automáticamente la interfaz web de Streamlit.


### Alternativa: Lanzamiento Directo
Si ya tienes la base de datos y quieres ir directo al chat:

```bash
streamlit run app.py
```

---
## 🚀 Trabajo Futuro y Escalabilidad (Roadmap)

Para evolucionar este sistema hacia un entorno de producción de alta disponibilidad, se proponen las siguientes mejoras estratégicas:

1. **Optimización de Recuperación (Query Transformation):**
   * **Query Rewriting:** Implementar un paso previo donde un LLM interprete y limpie la intención del usuario antes de la consulta vectorial.
   * **HyDE (Hypothetical Document Embeddings):** Generar respuestas hipotéticas para capturar fragmentos semánticamente similares con mayor precisión.

2. **Arquitectura GraphRAG:**
   * Migrar a una base de datos de grafos (**Neo4j**) para conectar conceptos transversales del libro que no están cerca físicamente (ej. relacionar técnicas de *Regularización* en el Cap. 6 con su aplicación en *SVM* en el Cap. 9).

3. **Transición a Modelos Cloud:**
   * Integración opcional mediante API con **Gemini 1.5 Pro** o **GPT-4o** para consultas de "borde" que requieran un razonamiento matemático extremo o síntesis de múltiples capítulos.

4. **Refinamiento de Inferencia Local (Prompt Tuning):**
   * Ajustar iterativamente el `Prompt Template` utilizado en `OllamaLLM` para optimizar el uso de la ventana de contexto y mejorar la asimilación de instrucciones de formato específicas del dominio estadístico.
---
## 🛡️ License & Contact

Desarrollado por [**Jose Luis Cabrera Vega**](https://www.linkedin.com/in/josecabrerav) para el proceso de selección de **Scanntech**.

