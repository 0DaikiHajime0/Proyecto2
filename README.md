# MLE-Project2: Customer Churn Analysis

**Especializacion Machine Learning Engineering - Curso II**

Prediccion de abandono de clientes (Customer Churn) utilizando tecnicas de Machine Learning y componentes agenticos de GenAI.

---

## 1. Problema de Machine Learning

### Tipo de problema
- **Categoria**: Aprendizaje Supervisado
- **Subcategoria**: Clasificacion Binaria
- **Objetivo**: Predecir si un cliente abandonara (churn) o permanecera en el servicio.

### Hipotesis
Es posible predecir el abandono de clientes utilizando variables demograficas, de comportamiento y de uso del servicio, lo que permitira a la empresa tomar acciones preventivas para retener a los clientes en riesgo.

### Componente Agentico (GenAI)
Se implementa un agente que utiliza **RAG (Retrieval-Augmented Generation)** para:
- Analizar patrones de churn historicos
- Generar recomendaciones personalizadas de retencion
- Explicar las predicciones del modelo en lenguaje natural

---

## 2. Dataset

### Fuente
- **Nombre**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Tamano**: 7,043 registros, 21 variables
- **Tipo**: Tabular, < 100MB

### Descripcion
Dataset que contiene informacion de clientes de una empresa de telecomunicaciones, incluyendo datos demograficos, servicios contratados, informacion de cuenta y si el cliente abandono o no el servicio.

### Diccionario de Datos

| Variable | Tipo | Descripcion |
|----------|------|-------------|
| `customerID` | String | ID unico del cliente |
| `gender` | Categorical | Genero del cliente (Male/Female) |
| `SeniorCitizen` | Binary | Si el cliente es adulto mayor (1) o no (0) |
| `Partner` | Binary | Si el cliente tiene pareja (Yes/No) |
| `Dependents` | Binary | Si el cliente tiene dependientes (Yes/No) |
| `tenure` | Numeric | Meses que el cliente ha estado con la empresa |
| `PhoneService` | Binary | Si tiene servicio telefonico (Yes/No) |
| `MultipleLines` | Categorical | Si tiene multiples lineas (Yes/No/No phone service) |
| `InternetService` | Categorical | Tipo de servicio de internet (DSL/Fiber optic/No) |
| `OnlineSecurity` | Categorical | Si tiene seguridad online (Yes/No/No internet service) |
| `OnlineBackup` | Categorical | Si tiene respaldo online (Yes/No/No internet service) |
| `DeviceProtection` | Categorical | Si tiene proteccion de dispositivo (Yes/No/No internet service) |
| `TechSupport` | Categorical | Si tiene soporte tecnico (Yes/No/No internet service) |
| `StreamingTV` | Categorical | Si tiene streaming de TV (Yes/No/No internet service) |
| `StreamingMovies` | Categorical | Si tiene streaming de peliculas (Yes/No/No internet service) |
| `Contract` | Categorical | Tipo de contrato (Month-to-month/One year/Two year) |
| `PaperlessBilling` | Binary | Si usa facturacion sin papel (Yes/No) |
| `PaymentMethod` | Categorical | Metodo de pago |
| `MonthlyCharges` | Numeric | Cargo mensual |
| `TotalCharges` | Numeric | Cargo total |
| `Churn` | Binary | **Variable objetivo** - Si el cliente abandono (Yes/No) |

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
│  │  Vector DB   │    │  Analisis    │    │  ciones      │      │
│  │  Embeddings  │    │  Predictions │    │  Retencion   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Model Card

### Model Details
- **Nombre**: Churn Prediction Model v1.0
- **Tipo**: Clasificador Binario
- **Algoritmos evaluados**: Logistic Regression, Random Forest, Gradient Boosting
- **Framework**: scikit-learn
- **Author**: Roman (Estudiante ML Engineering)

### Intended Use
- **Caso de uso principal**: Identificar clientes con alta probabilidad de abandono
- **Usuarios objetivo**: Equipos de retencion y marketing
- **Uso no adecuado**: No debe usarse como unica herramienta de decision; requiere intervencion humana

### Training Data
- **Fuente**: Kaggle - Telco Customer Churn
- **Tamano**: 7,043 registros, 21 variables originales + 5 features derivadas
- **Preprocesamiento**:
  - Manejo de valores faltantes (TotalCharges)
  - One-Hot Encoding para variables categoricas
  - Normalizacion con StandardScaler
  - Feature engineering: avg_monthly_per_tenure, num_services

### Evaluation Data
- **Metodo**: Train/Test Split (80/20) con stratificacion

### Resultados de Evaluacion

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.73 | ~0.50 | ~0.53 | ~0.51 | ~0.73 |
| Random Forest | ~0.72 | ~0.48 | ~0.51 | ~0.49 | ~0.72 |
| Gradient Boosting | ~0.73 | ~0.51 | ~0.55 | ~0.53 | ~0.74 |

*Nota: Metricas con dataset sintetico. Con el dataset real de Kaggle se esperan mejores resultados.*

### Limitaciones
- Los datos sinteticos pueden no capturar todas las correlaciones reales
- Requiere re-entrenamiento con datos reales de Kaggle
- El modelo puede tener sesgo hacia ciertos segmentos

### Etica y Fairness
- Evaluacion de fairness por genero y edad
- Transparencia en las decisiones del modelo

---

## 5. Resultados y Metricas

### Metricas Offline
- **Mejor modelo**: Gradient Boosting
- **ROC-AUC**: ~0.74
- **F1-Score**: ~0.53
- **Features mas importantes**: tenure, MonthlyCharges, TotalCharges, Contract, InternetService

### Graficas generadas
- `reports/figures/churn_distribution.png` - Distribucion de la variable objetivo
- `reports/figures/feature_analysis.png` - Analisis de features vs churn
- `reports/figures/categorical_analysis.png` - Variables categoricas vs churn
- `reports/figures/correlation_matrix.png` - Matriz de correlacion
- `reports/figures/model_comparison.png` - Comparacion de modelos
- `reports/figures/roc_curves.png` - Curvas ROC
- `reports/figures/confusion_matrix.png` - Matriz de confusion del mejor modelo
- `reports/figures/feature_importance.png` - Importancia de features

---

## 6. Componente Agentic / RAG

### Descripcion
Se implementa un agente de IA que combina el modelo de ML con un motor RAG para:
1. **Analizar clientes en riesgo**: Identifica factores de riesgo especificos
2. **Generar recomendaciones**: Propone acciones concretas de retencion
3. **Responder preguntas**: Consultas en lenguaje natural sobre datos de churn

### Uso
```python
from src.agentic.rag_engine import ChurnRAGEngine
from src.agentic.agent import ChurnAgent

# Inicializar motor RAG
rag = ChurnRAGEngine('data/raw/telco_customer_churn.csv')

# Crear agente
agent = ChurnAgent(model=best_model, rag_engine=rag)

# Analizar un cliente
explanation = agent.explain_prediction(customer_data, prediction=1, probability=0.85)

# Generar recomendacion de retencion
recommendation = agent.generate_retention_recommendation(customer_data, prediction=1, probability=0.85)

# Consultar sobre datos
response = agent.query('Cuales son los factores principales de churn?')
```

---

## 7. Estructura del Repositorio

```
mle-project2-churn-analysis/
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── tox.ini
│
├── data/
│   ├── raw/                    <- Datos originales
│   ├── interim/                <- Datos intermedios
│   ├── processed/              <- Datos para modelado (train.csv, test.csv)
│   └── external/
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-data-preprocessing.ipynb
│   ├── 03-model-training.ipynb
│   └── 04-agentic-rag.ipynb
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── evaluate.py
│   ├── visualization/
│   │   └── visualize.py
│   └── agentic/
│       ├── rag_engine.py
│       └── agent.py
│
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_training.py
│   ├── run_prediction.py
│   └── generate_sample_data.py
│
├── models/                     <- best_model.pkl
├── reports/figures/            <- Graficas generadas
├── references/
└── tests/
```

---

## 8. Tecnologias

| Categoria | Tecnologia |
|-----------|------------|
| Lenguaje | Python 3.14 |
| ML Framework | scikit-learn |
| Tracking | MLflow + DagsHub |
| GenAI | LangChain / FAISS (para RAG) |
| Datos | pandas, numpy |
| Visualizacion | matplotlib, seaborn |
| Testing | pytest |
| Git | GitHub Flow |

---

## 9. Instalacion y Uso

### Requisitos
- Python 3.10+
- pip

### Instalacion
```bash
# Clonar repositorio
git clone https://github.com/USER/mle-project2-churn-analysis.git
cd mle-project2-churn-analysis

# Instalar dependencias
pip install -r requirements.txt

# O instalar como paquete
pip install -e .
```

### Ejecucion
```bash
# Preprocesamiento
python scripts/run_preprocessing.py

# Entrenamiento
python scripts/run_training.py

# Prediccion
python scripts/run_prediction.py

# O usar Make
make preprocess
make train
make predict
```

---

## 10. Estrategia de Git

### Ramas
- **`main`**: Rama principal con codigo estable y releases
- **`development`**: Rama de integracion para desarrollo activo

### Flujo de Trabajo (GitHub Flow)
1. Crear rama desde `development` para cada feature
2. Hacer commits atomicos con mensajes descriptivos
3. Crear Pull Request hacia `development`
4. Revisar y merge con `--no-ff`
5. Al finalizar, PR de `development` a `main`
6. Crear Release v1.0.0

### Convenciones de Commits
```
feat: agregar pipeline de preprocesamiento
fix: corregir encoding de variables categoricas
docs: actualizar README con resultados
chore: actualizar dependencias
test: agregar tests para train_model
```

---

## 11. Estado Actual del Proyecto

### Pipeline ML completado
- **Dataset**: 7,043 clientes de telecomunicaciones (sintetico, reemplazar con Kaggle real)
- **Modelos evaluados**: Logistic Regression, Random Forest, Gradient Boosting
- **Mejor modelo**: Gradient Boosting (ROC-AUC ~0.74)
- **Features**: 32 (originales + One-Hot Encoding + derivadas)
- **Notebooks ejecutados**: EDA, Preprocessing, Training, Agentic RAG

### Componente Agentic completado
- **Motor RAG**: Retrieval de documentos de churn con estadisticas historicas
- **Agente**: Explica predicciones, genera recomendaciones de retencion, responde preguntas en lenguaje natural

### Git completado
- Ramas `main` y `development` creadas
- Merge de `development` a `main` realizado
- Tag `v1.0.0` creado localmente

---

## 12. Pendiente para Entrega Final

| Tarea | Descripcion | Estado |
|-------|-------------|--------|
| Dataset real | Descargar Telco Customer Churn de Kaggle y reemplazar el sintetico | Pendiente |
| DagsHub/MLflow | Configurar tracking remoto y subir experimentos con metricas, parametros y artefactos | Pendiente |
| Push a GitHub | `git push origin main --tags` para subir todo al repositorio remoto | Pendiente |
| Release v1.0.0 | Crear Release en GitHub con release notes detalladas | Pendiente |
| Documentacion Git | Documentar la estrategia de ramas y flujo de trabajo utilizada | Pendiente |
| Pull Request | Crear PR formal documentando los cambios (si se usa GitHub Flow) | Pendiente |

---

**Fecha de entrega**: Domingo 30 de Agosto de 2026
