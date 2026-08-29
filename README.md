# MLE-Project2: Spotify Track Popularity Prediction

**Especializacion Machine Learning Engineering - Curso II**

Prediccion de la categoria de popularidad de tracks de Spotify (Low / Medium / High)
utilizando tecnicas de Machine Learning y un componente agentico de GenAI (RAG).

---

## 1. Problema de Machine Learning

### Tipo de problema
- **Categoria**: Aprendizaje Supervisado
- **Subcategoria**: Clasificacion Multiclase (3 clases)
- **Objetivo**: Predecir la categoria de popularidad (`popularity_category`: Low / Medium / High) de un track a partir de sus caracteristicas de audio y metadatos.

### Hipotesis
Las caracteristicas acusticas (danceability, energy, loudness, tempo, etc.), el genero,
el ano de lanzamiento y la exposicion del artista contienen informacion util para
anticipar el nivel de popularidad de un track, permitiendo a sellos y artistas tomar
decisiones de produccion y promocion mas informadas.

### Componente Agentico (GenAI)
Se implementa un agente que utiliza **RAG (Retrieval-Augmented Generation)** para:
- Explicar en lenguaje natural por que un track se clasifica en cierta categoria de popularidad.
- Generar recomendaciones de produccion/promocion personalizadas.
- Responder preguntas sobre patrones historicos del dataset.

---

## 2. Dataset

### Fuente
- **Nombre**: [Spotify Artist Streaming Analytics (2015-2025)](https://www.kaggle.com/datasets) (dataset sintetico derivado de Spotify Music Analytics).
- **Tamano**: 85,000 tracks, 33 columnas originales.
- **Tipo**: Tabular, < 100MB (`data/raw/spotify_data_processed.csv`, ~19 MB).

### Diccionario de Datos (principales columnas)

| Variable | Tipo | Descripcion |
|----------|------|-------------|
| `track_id` | String | ID unico del track |
| `track_name` | String | Nombre del track |
| `artist_name` | String | Artista |
| `album_name` | String | Album |
| `release_date` | Date | Fecha de lanzamiento |
| `genre` | Categorical | Genero musical |
| `duration_ms` | Numeric | Duracion en ms |
| `popularity` | Numeric | Score de popularidad (0-100) |
| `danceability` | Numeric | Bailabilidad (0-1) |
| `energy` | Numeric | Energia (0-1) |
| `key` / `key_name` | Numeric / Categorical | Tonalidad |
| `loudness` | Numeric | Sonoridad (dB) |
| `mode` / `mode_name` | Binary / Categorical | Modalidad (Major/Minor) |
| `instrumentalness` | Numeric | Instrumentalidad (0-1) |
| `tempo` | Numeric | Tempo (BPM) |
| `stream_count` | Numeric | Numero de reproducciones |
| `country` | Categorical | Pais principal |
| `explicit` | Binary | Contenido explicito |
| `label` | String | Sello discografico |
| `release_year` / `release_month` / `release_quarter` | Numeric | Fechas derivadas |
| `release_day_of_week` | Categorical | Dia de lanzamiento |
| `duration_minutes` | Numeric | Duracion en minutos |
| `popularity_category` | **Categorical (objetivo)** | Low / Medium / High |
| `loudness_category` | Categorical | Bins de sonoridad |
| `is_explicit_bool` | Boolean | Explicit (bool) |
| `is_weekend_release` | Boolean | Lanzamiento fin de semana |
| `log_stream_count` | Numeric | log de streams |
| `upbeat_score` | Numeric | (danceability+energy)/2 |
| `artist_track_count` | Numeric | Numero de tracks del artista |

### Ingenieria de features
Se anaden: `dance_energy` (danceability x energy), `is_recent_release` (>=2020) y
`artist_exposure` (1/artist_track_count). Se descartan columnas con **fuga de
informacion** (`popularity`, `stream_count`, `log_stream_count`, `upbeat_score`) y
identificadores/ texto de alta cardinalidad.

---

## 3. Diagrama de Flujo del Proyecto

```
+--------------+    +----------------+    +----------------+
| 1. DATOS     |--->| 2. PREPROCESO  |--->| 3. MODELOS     |
| Raw CSV      |    | Limpieza + FE  |    | Training       |
| (Spotify)    |    | Split 80/20    |    | Evaluacion     |
+--------------+    +----------------+    +----------------+
                                            |
                                            v
+--------------+    +----------------+    +----------------+
| 6. RELEASE   |<---| 5. MLflow      |<---| 4. MODELO      |
| v1.0.0       |    | Tracking       |    | Final (mejor)  |
| GitHub       |    | Experiments    |    | + figuras      |
+--------------+    +----------------+    +----------------+

+----------------------------- COMPONENTE AGENTICO -----------------------------+
|  +-------------+    +------------------+    +-----------------------------+  |
|  | RAG Engine  |--->| Agente GenAI     |--->| Explicaciones /             |  |
|  | TF-IDF KB   |    | (RAG + reglas)   |    | Recomendaciones / Consultas|  |
|  +-------------+    +------------------+    +-----------------------------+  |
+------------------------------------------------------------------------------+
```

---

## 4. Model Card

### Model Details
- **Nombre**: Spotify Popularity Model v1.0
- **Tipo**: Clasificador multiclase (Low / Medium / High)
- **Algoritmos evaluados**: Logistic Regression (multinomial), Random Forest, Gradient Boosting
- **Framework**: scikit-learn
- **Author**: Roman (Estudiante ML Engineering)

### Intended Use
- **Caso de uso principal**: Estimar el nivel de popularidad esperado de un track nuevo.
- **Usuarios objetivo**: Equipos de A&R, sellos discograficos y artistas.
- **Uso no adecuado**: No debe usarse como unica herramienta de decision; requiere intervencion humana.

### Training Data
- **Fuente**: Spotify Artist Streaming Analytics (sintetico, 2015-2025)
- **Tamano**: 85,000 tracks -> 68,000 train / 17,000 test (split 80/20 estratificado)
- **Preprocesamiento**:
  - Sin valores nulos en el dataset.
  - One-Hot Encoding para variables categoricas (`genre`, `loudness_category`, `key_name`, `mode_name`, `release_day_of_week`, `is_weekend_release`).
  - Estandarizacion (`StandardScaler`) de variables numericas.
  - Feature engineering: `dance_energy`, `is_recent_release`, `artist_exposure`.
  - Manejo de fuga: se excluyen `popularity`, `stream_count`, `log_stream_count`, `upbeat_score`.

### Evaluation Data
- **Metodo**: Train/Test Split (80/20) con estratificacion por `popularity_category`.

### Resultados de Evaluacion (offline)

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC (ovr) |
|--------|----------|-------------------|----------------|------------|---------------|---------------|
| Logistic Regression | 0.301 | 0.334 | 0.332 | 0.259 | 0.356 | 0.497 |
| Random Forest | **0.388** | 0.332 | 0.331 | **0.294** | 0.449 | 0.494 |
| Gradient Boosting | 0.754 | 0.585 | 0.333 | 0.287 | 0.649 | 0.498 |
| Baseline (mayoria) | 0.754 | 0.251 | 0.333 | 0.287 | 0.649 | n/a |

**Mejor modelo seleccionado**: Random Forest (mayor F1-macro = 0.294).

> **Nota importante sobre las metricas.** El dataset es **totalmente sintetico** y sus
> variables fueron generadas como independentes: las caracteristicas de audio y metadatos
> tienen correlacion ~0 con la popularidad (la unica correlacion alta es con
> `log_stream_count`, que es una consecuencia y fue descartada por fuga). Por ello todos
> los modelos convergen a la linea base de la clase mayoritaria (Medium, ~75% de los datos)
> y el ROC-AUC es ~0.50 (comportamiento aleatorio). Esto es una propiedad inherente del
> dataset sintetico, documentada en `DataSet Context .md`, y no un defecto del pipeline.
> Con un dataset real de Spotify el mismo pipeline capturaria la senal acustica real.

### Limitations
- El dataset sintetico no reproduce correlaciones del mundo real.
- Desbalance de clases (Medium domina); se compensa con `class_weight="balanced"`.
- El modelo no debe usarse para decisiones de negocio reales sin validacion adicional.

### Etica y Fairness
- Evaluacion por genero y ano para detectar sesgos de prediccion.
- Transparencia mediante el agente RAG que explica cada prediccion.

---

## 5. Resultados y Metricas

### Metricas Offline
- **Mejor modelo**: Random Forest (F1-macro 0.294, ROC-AUC 0.494).
- **Features mas importantes** (ver `reports/figures/feature_importance.png`): combinacion de
  `artist_track_count`, `artist_exposure`, `duration_ms`, `explicit` y `dance_energy`.

### Graficas generadas
- `reports/figures/popularity_distribution.png` - Distribucion de la variable objetivo
- `reports/figures/numeric_feature_analysis.png` - Features numericas vs popularidad
- `reports/figures/categorical_analysis.png` - Variables categoricas vs popularidad
- `reports/figures/correlation_matrix.png` - Matriz de correlacion
- `reports/figures/model_comparison.png` - Comparacion de modelos + baseline
- `reports/figures/roc_curves.png` - Curvas ROC (one-vs-rest)
- `reports/figures/confusion_matrix.png` - Matriz de confusion del mejor modelo
- `reports/figures/feature_importance.png` - Importancia de features

### Metricas Online (simuladas)
El modelo no se sirve en produccion, por lo que la "evaluacion online" se simula
reproduciendo un escenario de streaming: el conjunto de prueba se predice en lotes
(`batch` de 1,000 tracks) y se calcula la macro-F1 acumulada a medida que llegan
los datos. La macro-F1 online se estabiliza en **0.294**, coincidiendo con el valor
offline (0.294), lo que indica estabilidad del modelo bajo una distribucion
estacionaria (propia de datos sinteticos). Los detalles de la simulacion estan en
`scripts/run_prediction.py` y el notebook `03-model-training.ipynb`.

| Tipo | Modelo | Accuracy | F1-macro | ROC-AUC |
|------|--------|----------|----------|---------|
| Offline | Random Forest | 0.388 | 0.294 | 0.494 |
| Online (streaming simulado) | Random Forest | 0.388 | 0.294 | n/a |

---

## 6. Conclusiones

- **Pipeline end-to-end funcional**: se implemento un flujo completo de ML (preproceso,
  ingenieria de features, entrenamiento, evaluacion, prediccion) reproducible via scripts
  y notebooks, con registro de experimentos en MLflow.
- **Componente GenAI (RAG)**: el agente explica predicciones, genera recomendaciones de
  produccion y responde consultas en lenguaje natural usando una base de conocimiento
  construida a partir del dataset (recuperacion TF-IDF, 100% offline).
- **Limitacion central del dataset**: al ser totalmente sintetico y con variables
  independientes, la popularidad no es predecible a partir de las features acusticas
  (correlacion ~0; la unica correlacion alta es con `log_stream_count`, que es fuga y
  fue descartada). Por ello todos los modelos convergen a la linea base de la clase
  mayoritaria (ROC-AUC ~0.50). Esta es una propiedad del datos, documentada en
  `DataSet Context .md`, y no un defecto del pipeline.
- **Valor del proyecto**: demuestra buenas practicas de MLOps (modularidad, MLflow,
  Git Flow con PR y Release, documentacion) que son el nucleo evaluado del curso. Con
  un dataset real de Spotify el mismo pipeline capturaria la senal acustica real y
  superaria la linea base.
- **Buenas practicas de desarrollo**: ramas `main`/`development`, Pull Request #1
  mergeado, Release `v1.0.0` y commits atomicos con convencion semantica.

---

## 7. Componente Agentic / RAG

### Descripcion
El agente (`src/agentic/agent.py`) combina el modelo de ML con un motor RAG
(`src/agentic/rag_engine.py`) que construye una base de conocimiento a partir de
estadisticas agregadas del dataset (perfiles por genero, tendencias por ano, artistas
destacados, insights de audio vs popularidad) y recupera contexto relevante con TF-IDF
(sin dependencias externas, 100% offline). El agente:

1. **Explica predicciones**: describe la clase predicha, su confianza, las caracteristicas
   del track y el contexto RAG recuperado (p.ej. perfil de su genero).
2. **Genera recomendaciones**: propone acciones de produccion/promocion (subir energia,
   aumentar frecuencia de lanzamiento, etc.).
3. **Responde preguntas**: consultas en lenguaje natural sobre el dataset.

### Uso
```python
from src.agentic.rag_engine import SpotifyRAGEngine
from src.agentic.agent import SpotifyPopularityAgent
from src.models.predict_model import load_model, predict

model, name = load_model()
rag = SpotifyRAGEngine('data/raw/spotify_data_processed.csv')
agent = SpotifyPopularityAgent(model=model, rag_engine=rag)

# Explicar y recomendar para un track
explanation = agent.explain_prediction(track_dict, prediction='Medium', probability=0.51)
recommendation = agent.generate_recommendation(track_dict, 'Medium', 0.51)

# Consultar el dataset
response = agent.query('Cuales generos tienen mayor popularidad?')
```

---

## 8. Estructura del Repositorio

```
mle-project2-spotify-popularity/
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── pyproject.toml
├── setup.py
├── DataSet Context .md
├── Proyecto Final - Curso MLE2 (1).pdf
│
├── data/
│   ├── raw/                    <- spotify_data_processed.csv (datos originales)
│   ├── interim/
│   ├── processed/              <- train.csv, test.csv (80/20 estratificado)
│   └── external/
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-data-preprocessing.ipynb
│   ├── 03-model-training.ipynb
│   └── 04-agentic-rag.ipynb
│
├── src/
│   ├── data/        (make_dataset.py, preprocess.py)
│   ├── features/    (build_features.py)
│   ├── models/      (train_model.py, predict_model.py, evaluate.py)
│   ├── visualization/ (visualize.py)
│   └── agentic/     (rag_engine.py, agent.py)
│
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_training.py
│   └── run_prediction.py
│
├── models/                     <- best_model.pkl (modelo productivo)
├── reports/figures/            <- Graficas generadas
├── references/
├── tests/
└── mlflow.db                   <- Experiments de MLflow (tracking local)
```

---

## 9. Tecnologias

| Categoria | Tecnologia |
|-----------|------------|
| Lenguaje | Python 3.14 |
| ML Framework | scikit-learn |
| Tracking | MLflow (sqlite local) + DagsHub (opcional) |
| GenAI | RAG con TF-IDF (sklearn) + agente de generacion de explicaciones |
| Datos | pandas, numpy |
| Visualizacion | matplotlib |
| Testing | pytest |
| Git | GitHub Flow |

---

## 10. Instalacion y Uso

### Requisitos
- Python 3.10+
- pip (o `uv`)

### Instalacion
```bash
git clone <repo-url>
cd mle-project2-spotify-popularity
pip install -r requirements.txt
# O instalar como paquete
pip install -e .
```

### Ejecucion
```bash
# 1. Preprocesamiento (genera data/processed/train.csv y test.csv)
python scripts/run_preprocessing.py

# 2. Entrenamiento + MLflow + figuras
python scripts/run_training.py

# 3. Prediccion + demostracion del agente RAG
python scripts/run_prediction.py

# O usar Make
make preprocess
make train
make predict
```

### MLflow
Los experimentos se registran en **DagsHub** (tracking remoto):

- **Repo**: https://dagshub.com/0DaikiHajime0/Proyecto2
- **Experimentos (runs + artefactos)**: https://dagshub.com/0DaikiHajime0/Proyecto2.mlflow/#/experiments/0
- **Model Registry (modelo productivo)**: https://dagshub.com/0DaikiHajime0/Proyecto2.mlflow/#/models

Cada run contiene **metricas**, **parametros** y el **artefacto del modelo**
(export MLflow: `MLmodel`, `model.skops`, `conda.yaml`, etc.). El mejor modelo
(Random Forest) esta registrado como `spotify_popularity_model` v1.

De forma local (sin credenciales) los experimentos se escriben en `mlflow.db`
(sqlite, `*.db` está ignorado pero `!mlflow.db` lo conserva). Para la UI local:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Para usar DagsHub define las variables en `.env` (ver `.env.example`):
```
DAGSHUB_USER=0DaikiHajime0
DAGSHUB_TOKEN=<tu-token>
DAGSHUB_REPO=Proyecto2
```
y al ejecutar `python scripts/run_training.py` los runs se enviaran
automaticamente a DagsHub.

---

## 11. Estrategia de Git

### Ramas
- **`main`**: Rama principal con codigo estable y releases.
- **`development`**: Rama de integracion para desarrollo activo.

### Flujo de Trabajo (GitHub Flow)
1. Crear rama desde `development` para cada feature.
2. Commits atomicos con mensajes descriptivos (convencion `feat:`, `fix:`, `docs:`, etc.).
3. Pull Request de la feature hacia `development` y merge con `--no-ff`.
4. Al finalizar, PR de `development` a `main`.
5. Crear Release v1.0.0 desde `main`.

---

## 12. Estado Actual del Proyecto

### Pipeline ML completado
- Dataset: 85,000 tracks de Spotify (sintetico).
- Preprocesamiento: limpieza, feature engineering, OHE, scaling, split 80/20 estratificado.
- Modelos evaluados: Logistic Regression, Random Forest, Gradient Boosting + baseline.
- Mejor modelo: Random Forest (F1-macro 0.294); evidencia de convergencia a baseline
  por la naturaleza sintetica del dataset (ROC-AUC ~0.50).
- MLflow: experimentos con metricas, parametros y artefactos (modelo logueado).
- Modelo productivo guardado en `models/best_model.pkl`.

### Componente Agentic completado
- Motor RAG: base de conocimiento de estadisticas agregadas + recuperacion TF-IDF.
- Agente: explica predicciones, genera recomendaciones y responde preguntas en NL.

### Git completado
- Ramas `main` y `development`.
- Pull Request de `development` -> `main` con merge `--no-ff`.
- Tag `v1.0.0`.

---

## 13. Pendiente / Notas para Entrega

| Tarea | Descripcion | Estado |
|-------|-------------|--------|
| Dataset sintetico | El dataset provisto es sintetico; metricas cercanas a baseline son esperadas | Documentado |
| DagsHub/MLflow | Tracking local (sqlite) configurado; soporte remoto via `.env` | Listo |
| Estructura Git | Ramas `main` + `development`, PR #1 mergeado y tag `v1.0.0` | Listo |
| Push a GitHub | `git push origin main development --tags` | **Hecho** |
| Pull Request | PR #1 `development` -> `main` (mergeado y cerrado) | **Hecho** |
| Release v1.0.0 | Release publicado en GitHub con release notes | **Hecho** |
| Documentacion Git | Estrategia documentada en seccion 10 | Listo |

---

**Fecha de entrega**: Domingo 30 de Agosto de 2026
