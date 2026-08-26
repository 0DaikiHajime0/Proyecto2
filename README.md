# MLE-Project2: Customer Churn Analysis

**Especialización Machine Learning Engineering - Curso II**

Predicción de abandono de clientes (Customer Churn) utilizando técnicas de Machine Learning y componentes agenticos de GenAI.

---

## 1. Problema de Machine Learning

### Tipo de problema
- **Categoría**: Aprendizaje Supervisado
- **Subcategoría**: Clasificación Binaria
- **Objetivo**: Predecir si un cliente abandonará (churn) o permanecerá en el servicio.

### Hipótesis
Es posible predecir el abandono de clientes utilizando variables demográficas, de comportamiento y de uso del servicio, lo que permitirá a la empresa tomar acciones preventivas para retener a los clientes en riesgo.

### Componente Agentico (GenAI)
Se implementará un agente que utilize **RAG (Retrieval-Augmented Generation)** para:
- Analizar patrones de churn históricos
- Generar recomendaciones personalizadas de retención
- Explicar las predicciones del modelo en lenguaje natural

---

## 2. Dataset

### Fuente
- **Nombre**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Tamaño**: ~7,000 registros, 21 variables
- **Tipo**: Tabular, < 100MB

### Descripción
Dataset que contiene información de clientes de una empresa de telecomunicaciones, incluyendo datos demográficos, servicios contratados, información de cuenta y si el cliente abandonó o no el servicio.

### Diccionario de Datos

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `customerID` | String | ID único del cliente |
| `gender` | Categorical | Género del cliente (Male/Female) |
| `SeniorCitizen` | Binary | Si el cliente es adulto mayor (1) o no (0) |
| `Partner` | Binary | Si el cliente tiene pareja (Yes/No) |
| `Dependents` | Binary | Si el cliente tiene dependientes (Yes/No) |
| `tenure` | Numeric | Meses que el cliente ha estado con la empresa |
| `PhoneService` | Binary | Si tiene servicio telefónico (Yes/No) |
| `MultipleLines` | Categorical | Si tiene múltiples líneas (Yes/No/No phone service) |
| `InternetService` | Categorical | Tipo de servicio de internet (DSL/Fiber optic/No) |
| `OnlineSecurity` | Categorical | Si tiene seguridad online (Yes/No/No internet service) |
| `OnlineBackup` | Categorical | Si tiene respaldo online (Yes/No/No internet service) |
| `DeviceProtection` | Categorical | Si tiene protección de dispositivo (Yes/No/No internet service) |
| `TechSupport` | Categorical | Si tiene soporte técnico (Yes/No/No internet service) |
| `StreamingTV` | Categorical | Si tiene streaming de TV (Yes/No/No internet service) |
| `StreamingMovies` | Categorical | Si tiene streaming de películas (Yes/No/No internet service) |
| `Contract` | Categorical | Tipo de contrato (Month-to-month/One year/Two year) |
| `PaperlessBilling` | Binary | Si usa facturación sin papel (Yes/No) |
| `PaymentMethod` | Categorical | Método de pago |
| `MonthlyCharges` | Numeric | Cargo mensual |
| `TotalCharges` | Numeric | Cargo total |
| `Churn` | Binary | **Variable objetivo** - Si el cliente abandonó (Yes/No) |

---

## 3. Diagrama de Flujo del Proyecto

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUJO DEL PROYECTO                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   1. DATOS   │───▶│2. PREPROCESO │───▶│ 3. MODELOS   │
│   Raw Data   │    │   Limpieza   │    │  Training    │
│   (CSV)      │    │   Feature    │    │  Evaluation  │
│              │    │   Engineering│    │  Selection   │
└──────────────┘    └──────────────┘    └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  6. RELEASE  │◀───│5. MLflow     │◀───│ 4. MODELO    │
│  v1.0.0      │    │  Tracking    │    │  Final       │
│  GitHub      │    │  Experiments │    │  Optimizado  │
└──────────────┘    └──────────────┘    └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    COMPONENTE AGENTICO                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   RAG        │───▶│  Agente GenAI│───▶│  Recomenda-  │      │
│  │  Vector DB   │    │  Análisis    │    │  ciones      │      │
│  │  Embeddings  │    │  Predictions │    │  Retención   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Model Card

### Model Details
- **Nombre**: Churn Prediction Model v1.0
- **Tipo**: Clasificador Binario
- **Algoritmos a evaluar**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Framework**: scikit-learn / XGBoost
- **Author**: Roman (Estudiante ML Engineering)

### Intended Use
- **Caso de uso principal**: Identificar clientes con alta probabilidad de abandono
- **Usuarios objetivo**: Equipos de retención y marketing
- **Uso no adecuado**: No debe usarse como única herramienta de decisión; requiere intervención humana

### Training Data
- **Fuente**: Kaggle - Telco Customer Churn
- **Período**: Datos históricos de clientes de telecomunicaciones
- **Preprocesamiento**: 
  - Codificación de variables categóricas
  - Normalización de variables numéricas
  - Manejo de valores faltantes

### Evaluation Data
- **Método**: Train/Test Split (80/20) + Cross Validation (5-fold)

### Métricas de Evaluación

#### Métricas Offline
| Métrica | Objetivo |
|---------|----------|
| AUC-ROC | > 0.80 |
| F1-Score | > 0.75 |
| Precision | > 0.70 |
| Recall | > 0.70 |
| Accuracy | > 0.75 |

#### Métricas Online (post-despliegue)
| Métrica | Objetivo |
|---------|----------|
| Latencia de predicción | < 100ms |
| Throughput | > 100 req/s |
| Drift de datos | Monitoreo continuo |

### Limitaciones
- Los datos pueden no representar todos los mercados
- El modelo puede tener sesgo hacia ciertos segmentos demográficos
- Requiere re-entrenamiento periódico

### Ética y Fairness
- Evaluación de fairness por género y edad
- Transparencia en las decisiones del modelo

---

## 5. Estructura del Repositorio

```
mle-project2-churn-analysis/
├── LICENSE
├── Makefile                    <- Comandos: make data, make train, make predict
├── README.md                   <- Este archivo
├── pyproject.toml              <- Configuración del proyecto
├── requirements.txt            <- Dependencias
├── setup.py                    <- Instalación del paquete
├── tox.ini                     <- Configuración de testing
│
├── data/
│   ├── raw/                    <- Datos originales (Telco Customer Churn.csv)
│   ├── interim/                <- Datos intermedios procesados
│   ├── processed/              <- Datos finales para modelado
│   └── external/               <- Datos externos
│
├── notebooks/
│   ├── 01-data-exploration.ipynb       <- Análisis exploratorio (EDA)
│   ├── 02-data-preprocessing.ipynb     <- Preprocesamiento de datos
│   ├── 03-model-training.ipynb         <- Entrenamiento y evaluación
│   └── 04-agentic-rag.ipynb           <- Componente agentic RAG
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py     <- Descarga y carga de datos
│   │   └── preprocess.py       <- Pipeline de preprocesamiento
│   ├── features/
│   │   └── build_features.py   <- Feature engineering
│   ├── models/
│   │   ├── train_model.py      <- Entrenamiento con MLflow
│   │   ├── predict_model.py    <- Predicciones
│   │   └── evaluate.py         <- Métricas de evaluación
│   ├── visualization/
│   │   └── visualize.py        <- Gráficas y reportes
│   └── agentic/
│       ├── rag_engine.py       <- Motor RAG
│       └── agent.py            <- Agente GenAI
│
├── scripts/
│   ├── run_preprocessing.py    <- Script de preprocesamiento
│   ├── run_training.py         <- Script de entrenamiento
│   └── run_prediction.py       <- Script de predicción
│
├── models/                     <- Modelos serializados
│   └── .gitkeep
│
├── reports/
│   └── figures/                <- Gráficas generadas
│
├── references/                 <- Diccionarios de datos, documentación
│
├── mlruns/                     <- Experimentos MLflow (local)
│
└── tests/                      <- Tests unitarios
```

---

## 6. Plan de Trabajo

### Fase 1: Preparación del Entorno (Día 1)
- [ ] Configurar estructura final del repositorio
- [ ] Crear ramas `main` y `development`
- [ ] Configurar MLflow con DagsHub
- [ ] Instalar dependencias actualizadas en `requirements.txt`

### Fase 2: Preprocesamiento de Datos (Día 2)
- [ ] Crear notebook `01-data-exploration.ipynb` (EDA completo)
- [ ] Crear notebook `02-data-preprocessing.ipynb`
- [ ] Implementar `src/data/make_dataset.py` (carga de datos)
- [ ] Implementar `src/data/preprocess.py` (pipeline de preprocesamiento)
- [ ] Implementar `src/features/build_features.py` (feature engineering)
- [ ] Script `scripts/run_preprocessing.py`

### Fase 3: Modelado y Evaluación (Día 3-4)
- [ ] Crear notebook `03-model-training.ipynb`
- [ ] Implementar `src/models/train_model.py` con MLflow tracking
- [ ] Implementar `src/models/evaluate.py` con métricas
- [ ] Experimentar con múltiples algoritmos:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- [ ] Seleccionar mejor modelo basado en métricas
- [ ] Script `scripts/run_training.py`
- [ ] Registrar experimentos en MLflow con:
  - Métricas (AUC, F1, Precision, Recall)
  - Parámetros del modelo
  - Artefactos (modelo serializado, gráficas)

### Fase 4: Componente Agentic / RAG (Día 5)
- [ ] Crear notebook `04-agentic-rag.ipynb`
- [ ] Implementar `src/agentic/rag_engine.py`
  - Crear embeddings de datos históricos de churn
  - Configurar retriever
- [ ] Implementar `src/agentic/agent.py`
  - Agente que analiza predicciones
  - Genera recomendaciones de retención en lenguaje natural
- [ ] Script `scripts/run_prediction.py`

### Fase 5: Documentación y Calidad (Día 6)
- [ ] Actualizar `README.md` con resultados finales
- [ ] Documentar código con docstrings
- [ ] Agregar `.gitignore` completo
- [ ] Crear `requirements.txt` actualizado
- [ ] Escribir instrucciones de ejecución

### Fase 6: CI/CD y Release (Día 7)
- [ ] Crear Pull Request de `development` a `main`
- [ ] Documentar cambios en el PR
- [ ] Merge del PR
- [ ] Crear Release v1.0.0 en GitHub con release notes
- [ ] Verificar que el repositorio esté en tag v1.0.0
- [ ] Documentar estrategia de Git utilizada

---

## 7. Estrategia de Git

### Ramas
- **`main`**: Rama principal con código estable y releases
- **`development`**: Rama de integración para desarrollo activo

### Flujo de Trabajo (GitHub Flow)
1. Crear rama desde `development` para cada feature
2. Hacer commits atómicos con mensajes descriptivos
3. Crear Pull Request hacia `development`
4. Revisar y merge con `--no-ff`
5. Al finalizar, PR de `development` a `main`
6. Crear Release v1.0.0

### Convenciones de Commits
```
feat: agregar pipeline de preprocesamiento
fix: corregir encoding de variables categóricas
docs: actualizar README con resultados
chore: actualizar dependencias
test: agregar tests para train_model
```

---

## 8. Tecnologías

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python 3.14 |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Tracking | MLflow + DagsHub |
| GenAI | LangChain / OpenAI API (para RAG) |
| Datos | pandas, numpy |
| Visualización | matplotlib, seaborn |
| Testing | pytest |
| Git | GitHub Flow |

---

## 9. Métricas de Evaluación

### Offline (en notebook)
- AUC-ROC
- F1-Score
- Precision / Recall
- Accuracy
- Matriz de Confusión
- Curva ROC

### Online (post-despliegue)
- Latencia de predicción
- Throughput
- Monitoreo de data drift

---

## 10. Entregables

| Entregable | Estado |
|------------|--------|
| README.md con estructura completa | Pendiente |
| Notebooks (4 notebooks) | Pendiente |
| Código reusable (src/) | Pendiente |
| Scripts de ejecución | Pendiente |
| Experimentos en MLflow/DagsHub | Pendiente |
| Release v1.0.0 | Pendiente |
| PR mergeado (development → main) | Pendiente |
| Documentación de estrategia Git | Pendiente |
| .gitignore, requirements.txt | Pendiente |

---

**Fecha de entrega**: Domingo 30 de Agosto de 2026
