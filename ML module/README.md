# 🧠 CARIVIX AI - Complete AI Platform

A **production-ready AI platform** encompassing two major pipelines:

1. **🤖 Model Training Pipeline** — End-to-end ML training with **MLflow experiment tracking**, supporting **classification** and **regression** tasks.
2. **📚 RAG Pipeline** — Retrieval-Augmented Generation pipeline for **document Q&A** using FAISS vector search + LLM (Ollama / HuggingFace).

---

## 📋 Project Overview

### 🤖 Model Training Pipeline
- **Data Loading & Validation** — Automatically loads datasets (CSV, Excel, Parquet) and performs quality checks. Auto-generates a sample dataset if none is found.
- **Data Cleaning** — Handles missing values, removes duplicates, detects/removes outliers, encodes categorical variables, scales features, and handles imbalanced data (SMOTE, ADASYN).
- **Feature Engineering** — Creates polynomial, interaction, ratio, frequency, binning, clustering, PCA, text-derived, and date-based features with automated selection.
- **Model Training** — Supports **8 classification** and **9 regression** algorithms with optional hyperparameter tuning (Grid/RandomizedSearchCV) and cross-validation.
- **Model Evaluation** — Comprehensive metrics & plots: Accuracy, Precision, Recall, F1, ROC AUC (classification) / MAE, RMSE, R² (regression), plus confusion matrix, ROC curve, precision-recall, calibration, learning curves, and SHAP plots.
- **Experiment Tracking** — All runs logged to **MLflow** with full reproducibility and artifact storage.
- **Model Versioning** — Every model is saved with a timestamp and logged as an MLflow artifact. Supports model export for downstream deployment.
- **Prediction Pipeline** — Load trained models and make predictions (with optional probability outputs) on new data.
- **Model Comparison** — Train and compare multiple algorithms in a single run.
- **Baseline Pipeline** — Run all baseline models, compare results, and identify the best-performing model automatically.

### 📚 RAG (Retrieval-Augmented Generation) Pipeline
- **Document Loading** — Load documents from PDF, DOCX, TXT, and CSV formats.
- **Text Preprocessing** — Clean and normalize text (whitespace, special characters, control characters).
- **Semantic Chunking** — Split documents into semantic chunks using RecursiveCharacterTextSplitter.
- **Embedding Generation** — Generate dense vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- **FAISS Vector Index** — Store and search embeddings using FAISS (Facebook AI Similarity Search).
- **Similarity Retrieval** — Retrieve top-K most relevant chunks for a user query.
- **Prompt Construction** — Build context-aware prompts from retrieved chunks.
- **LLM Response Generation** — Generate answers using Ollama (Llama 3.1, Mistral) or HuggingFace models.

---

## 📁 Complete Folder Structure

> This folder-tree section is kept up-to-date automatically. To refresh the tree in this README, run the update script included in the project root:

PowerShell (Windows):

```powershell
.\update_readme_tree.ps1
```

The tree output is injected between the markers below. Do not edit the generated block manually; run the script instead.

<!-- DIR_TREE_START -->
```text
CARIVIX_AI_Model_Training
├── .gitignore
├── api.py
├── evaluate_rag.py
├── main.py
├── main_rag.py
├── mlflow.db
├── query_result.json
├── README.md
├── requirements.txt
├── requirements_rag.txt
├── run_baseline_pipeline.py
├── update_readme_tree.ps1
├── config/
│   └── config.yaml
├── data/
│   ├── arxiv_data.csv
│   ├── CARIVIX_Economic_Indicators_By_Dataset.xlsx
│   ├── Economic_Trend_Analysis_Cleaned (1).csv
│   ├── Market_Analysis_1981_2025_Final_Cleaned.csv
│   ├── Public Program Evaluation_final.csv
│   ├── Traffic Analysis_Final.csv
│   ├── documents/
│   │   ├── sample_about_carivix.txt
│   │   ├── sample_economics.txt
│   │   └── sample_report.docx
│   ├── processed/
│   │   └── processed_20260731_165209.csv
│   └── raw/
│       └── dataset.csv
├── experiments/
│   ├── baseline_comparison_20260731_170507.csv
│   ├── baseline_comparison_20260803_172526.csv
│   ├── baseline_summary_20260731_170507.json
│   ├── baseline_summary_20260803_172526.json
│   ├── model_path_20260731_165209.txt
│   ├── model_path_20260804_120027.txt
│   └── models/
│       ├── m-02886553657a428485ea1a40d83ceb2f/
│       │   └── artifacts/
│       │       └── requirements.txt
│       └── m-5c7b1296d8504c85b2b8ae2cea0a410b/
│           └── artifacts/
│               └── requirements.txt
├── exports/
├── logs/
├── models/
├── rag/
├── src/
├── tests/
```
<!-- DIR_TREE_END -->

---

## 🚀 Installation

### 1. Clone the project

```bash
cd CARIVIX_AI_Model_Training
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

All dependencies (including XGBoost, SHAP, imbalanced-learn) are already listed in `requirements.txt` — no separate install steps needed.

## ⚡ Quick Start: Run Each Pipeline

Use these commands from the project root to run the main workflows directly.

```powershell
cd "D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training"

# 1) Baseline model comparison
python .\run_baseline_pipeline.py

# 2) Full ML training pipeline
python .\main.py
python .\main.py --generate-sample-data
python .\main.py --compare
python .\main.py --profile
python .\main.py --predict .\data\raw\dataset.csv

# 3) RAG pipeline
python main_rag.py --index
python main_rag.py --query "What is CARIVIX AI?"
python main_rag.py --interactive

# 4) NLP / full AI routing flow
python nlp_module.py
python api.py

# Call the combined AI endpoint after the API is running
curl.exe -X POST "http://127.0.0.1:8000/api/v1/ai/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Predict next quarter GDP for the project dataset","k":5,"max_tokens":512,"temperature":0.0}'
```

### Pipeline summary
- Baseline: `run_baseline_pipeline.py` compares multiple baseline models and prints the best result.
- ML pipeline: `main.py` runs the end-to-end training, validation, feature engineering, evaluation, and model export workflow.
- RAG pipeline: `main_rag.py` indexes document chunks and answers queries using retrieval + generation.
- NLP / integrated AI: `nlp_module.py` analyzes the query intent, and `api.py` exposes the full NLP + RAG + ML orchestration endpoint.

## 🎯 CLI Commands & Usage

### Basic Training

```bash
python main.py
```

This will:
1. Load the dataset from the configured path (auto-generates sample data if none exists)
2. Validate and clean the data
3. Engineer features
4. Train the selected algorithm (with optional tuning / CV / SHAP)
5. Evaluate performance (metrics + plots)
6. Log everything to MLflow
7. Save the trained model

```bash
# Run the full pipeline with default config
python main.py

# Generate sample data, then train
python main.py --generate-sample-data
python main.py

# Use a custom dataset and enable hyperparameter tuning
python main.py --config my_config.yaml --tune

# Compare RandomForest vs GradientBoosting
python main.py --compare

# Profile the dataset before training
python main.py --profile

# Train, tune, profile, and export in one run
python main.py --tune --profile --export

# Make predictions on new data
python main.py --predict data/raw/new_data.csv
```

### Model Service API

The project includes a production-ready inference API powered by FastAPI. Use the project virtual environment and run the API from the project root so Python can resolve the `src` package and the `models/` directory correctly.

#### 1) Create and activate the environment

```powershell
cd "D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 2) Start the API server

```powershell
cd "D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training"
.\venv\Scripts\Activate.ps1
python api.py
```

Or run it with Uvicorn explicitly:

```powershell
cd "D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training"
.\venv\Scripts\Activate.ps1
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

The app exposes Swagger at:

```text
http://127.0.0.1:8000/docs
```

### AI / RAG + ML integrated query endpoint

This repository exposes an integrated AI query endpoint that routes natural-language queries through the project NLP component, the RAG pipeline, and — when appropriate — the ML pipeline.

- Endpoint: POST /api/v1/ai/query
- Request JSON:

```json
{
  "query": "Your natural language question or instruction",
  "k": 5,                  // optional - number of RAG chunks to retrieve
  "max_tokens": 512,       // optional - LLM max tokens
  "temperature": 0.0,      // optional - LLM temperature
  "auto_fill_missing": false // optional - if true, allow conservative auto-fill of missing ML base fields
}
```

- Behavior:
  - The endpoint runs the project's NLP entrypoint (nlp_module.py) to detect intent and a routing decision ("rag", "ml", or "combined").
  - Routing decisions:
    - rag: retrieve relevant document chunks and generate an LLM answer (RAG). Requires a built FAISS index (see below).
    - ml: attempts a conservative NL→feature extraction for the model's required base fields. If extraction is incomplete it returns the extracted fields and a list of missing fields. If `auto_fill_missing` is set to true, the endpoint will fill missing base fields with conservative defaults (categorical defaults or safe numeric defaults) and attempt ModelService.predict. This auto-fill behavior is explicitly opt-in because ML preprocessing expects specific base fields and types.
    - combined: retrieves context from RAG and attempts generation; ML inclusion is guarded and requires explicit structured input or auto-fill approval.

- Response (RAG example):

```json
{
  "success": true,
  "nlp": { "processed_query": "what is carivix", "intent": "rag_query", "confidence": 0.6, "entities": {}, "route": "rag" },
  "selected_route": "rag",
  "rag_context": { /* top-k chunks with score and source, retrieval/generation times */ },
  "ml_prediction": null,
  "final_response": "Generated answer from the LLM"
}
```

Notes and operational tips:
- RAG index: create the FAISS index before using the RAG route. Use the included CLI: `python main_rag.py --index` (or call `RAGPipeline.index_documents()` programmatically). The index files are saved under `data/vector_store/` (index.faiss and index.pkl).
- LLM backend: Ollama is the default. The repository default Ollama request timeout has been increased to 300s to reduce generation timeouts during long prompts; this can be tuned by passing `llm_kwargs` when constructing the RAGPipeline or by editing `rag/generator.py`.
- ML route: ML inference expects structured tabular base fields (age, income, credit_score, loan_amount, years_employed, education, employment_status, marital_status, housing_type, application_date). If you want reliable automatic NL→feature mapping, provide example NL→feature pairs and the mapping will be refined.
- Tests & examples:
  - `run_api_test.py` exercises the `/api/v1/ai/query` flows (RAG + ML with/without `auto_fill_missing`).
  - Use `main_rag.py --query "your question"` to test the RAG pipeline locally.

- NLP entrypoint: `nlp_module.py` in the repository root. If you have a trained intent classifier, place it at `models/intent_classifier.joblib` or update the module's loading paths.

Use the endpoint responsibly: automatic ML filling is conservative and opt-in (`auto_fill_missing: true`) because wrong defaults can produce misleading predictions.rypoint (nlp_module.analyze).
  - Based on the NLP output, the request is routed to:
    - RAG: retrieve relevant document chunks and generate an LLM answer (default), or
    - ML: (route selected) instructs to call /api/v1/predict with structured features (ModelService expects tabular features), or
    - Combined: returns retrieved context and attempts LLM generation; ML integration requires mapping and is left as a placeholder.

- Response example (RAG):

```json
{
  "success": true,
  "nlp": { "processed_query": "what is carivix", "intent": "rag_query", "confidence": 0.6, "entities": {}, "route": "rag" },
  "selected_route": "rag",
  "rag_context": {
    "retrieved_chunks": [ /* top-k chunks with score and source */ ],
    "retrieval_time": 0.123,
    "generation_time": 0.456,
    "total_time": 0.579
  },
  "ml_prediction": null,
  "final_response": "Generated answer from the LLM"
}
```

Notes:
- ML inference requires structured model inputs; the integrated endpoint does not attempt to guess the tabular feature mapping from free text. Use /api/v1/predict for ML model inference with explicit feature JSON.
- The NLP module used is `nlp_module.py` in the project root. If you have a trained intent classifier, add its path to the module's _MODEL_PATHS or update the module to load it.


#### 3) Health check

```powershell
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "service": "CARIVIX AI Model Service",
  "models_loaded": 8,
  "models_failed": 0,
  "failed_models": null
}
```

#### 4) List available models

```powershell
curl http://127.0.0.1:8000/api/v1/models
```

#### 5) Get default model info

```powershell
curl http://127.0.0.1:8000/api/v1/model/info
```

#### 6) Single prediction request

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
   "age": 43,
   "income": 67976.67,
   "credit_score": 694,
   "loan_amount": 14857.28,
   "years_employed": 19,
   "education": "Master",
   "employment_status": "Employed",
   "marital_status": "Married",
   "housing_type": "Rent",
   "application_date": "2024-09-21"
  }'
```

#### 7) Batch prediction request

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
   "records": [
     {
       "age": 43,
       "income": 67976.67,
       "credit_score": 694,
       "loan_amount": 14857.28,
       "years_employed": 19,
       "education": "Master",
       "employment_status": "Employed",
       "marital_status": "Married",
       "housing_type": "Rent",
       "application_date": "2024-09-21"
     },
     {
       "age": 35,
       "income": 52000,
       "credit_score": 710,
       "loan_amount": 12000,
       "years_employed": 8,
       "education": "Bachelor",
       "employment_status": "Self-Employed",
       "marital_status": "Single",
       "housing_type": "Own",
       "application_date": "2024-10-05"
     }
   ]
  }'
```

The API loads the trained models from `models/`, applies the same preprocessing and feature-engineering logic as the training pipeline, and returns prediction metadata and confidence.

If a model file is incompatible or broken, the service will continue loading the working models and report the failed file names in `/health` under `failed_models`.

### Data Cleanup & Canonical Files

The project uses a single canonical set of data files under `data/` to avoid stale, duplicated, or copy-pasted CSV/Excel sources. Legacy duplicate files under the older Python dataset folders were removed and downstream scripts were updated to point to the canonical dataset locations instead of local copies.

---

## 📊 MLflow Experiment Tracking

### Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open your browser to: **http://localhost:5000**

### What Gets Logged

For every training run, MLflow automatically captures:

| Category | Details |
|----------|---------|
| **Dataset Info** | Dataset name, version, training timestamp |
| **Model Info** | Model name, algorithm, hyperparameters (including tuned `best_*` params) |
| **Metrics** | Classification: Accuracy, Precision, Recall, F1 Score, ROC AUC. Regression: MAE, MSE, RMSE, R². CV mean/std if enabled. |
| **Artifacts** | Confusion Matrix (PNG), Classification Report (CSV), Feature Importance Plot (PNG), ROC Curve (PNG), Precision-Recall Curve (PNG), Calibration Curve (PNG), Learning Curve (PNG), SHAP Summary (PNG), Model File (.pkl) |

### Viewing Experiments

1. Run the MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
2. Navigate to the **CARIVIX_AI** experiment
3. Browse runs, compare metrics across runs, view and download artifacts

---

## 🧪 Sample Dataset

When no dataset is found at the configured path, the pipeline **automatically generates** a **synthetic loan approval dataset** with:

- **1000 samples** with 9 features (5 numerical, 4 categorical, 1 datetime)
- Mixed data types (numerical, categorical, datetime)
- Intentionally introduced missing values (5%) and duplicate rows (10)
- Binary classification target (loan approval: 0/1)
- Realistic feature distributions (age, income, credit_score, education, employment_status, etc.)

This allows you to test the full pipeline **immediately after installation** without any external data.

---

## 📈 Supported Algorithms

### Classification

| Algorithm | Status | Notes |
|-----------|--------|-------|
| **RandomForest** | ✅ Default | Robust ensemble, handles non-linearity well |
| **GradientBoosting** | ✅ Supported | Often higher accuracy, more tuning needed |
| **XGBoost** | ✅ Supported | High performance; installed via `requirements.txt` |

### Regression

| Algorithm | Status | Notes |
|-----------|--------|-------|
| **RandomForestRegressor** | ✅ Supported | Ensemble for non-linear regression |
| **GradientBoostingRegressor** | ✅ Supported | Boosted ensemble for regression |
| **XGBoostRegressor** | ✅ Supported | XGBoost for regression tasks |
| **LinearRegression** | ✅ Supported | Simple linear model |
| **Ridge** | ✅ Supported | L2-regularized linear regression |
| **Lasso** | ✅ Supported | L1-regularized linear regression |
| **ElasticNet** | ✅ Supported | Combined L1 + L2 regularization |

Configure the algorithm in `config/config.yaml` under the `algorithm` field, and set `task_type` to `"classification"` or `"regression"`.

---

## 📐 Evaluation Metrics

### Classification

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Positive predictive value (weighted average for multiclass) |
| **Recall** | Sensitivity / True Positive Rate (weighted average for multiclass) |
| **F1 Score** | Harmonic mean of Precision & Recall (weighted average for multiclass) |
| **ROC AUC** | Area Under the ROC Curve (binary classification only) |

### Regression

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **R² Score** | Coefficient of Determination |

---

## 📊 Generated Plots

The pipeline automatically generates the following visualizations based on the task type:

| Plot | Classification | Regression | Description |
|------|:-------------:|:----------:|-------------|
| **Confusion Matrix** | ✅ | ❌ | True vs predicted labels heatmap |
| **ROC Curve** | ✅ (binary) | ❌ | Trade-off between TPR and FPR |
| **Precision-Recall Curve** | ✅ (binary) | ❌ | Trade-off between precision and recall |
| **Calibration Curve** | ✅ (binary) | ❌ | Reliability of predicted probabilities |
| **Feature Importance** | ✅ | ✅ | Top N feature importance bar chart |
| **Learning Curve** | ✅ | ✅ | Training & CV score vs training examples |
| **SHAP Summary** | ✅ | ✅ | SHAP values for model interpretability |
| **Prediction Scatter** | ❌ | ✅ | Predicted vs actual values |
| **Residuals** | ❌ | ✅ | Residual distribution & residuals vs predicted |

All plots are saved to `experiments/` and logged as MLflow artifacts.

---

## 🏗️ Architecture & Design

### Modular Architecture

```
main.py (CLI orchestrator)
  │
  ├── src/utils.py              → Logging, config loading, data profiling, retry decorator, timer
  ├── src/preprocess.py         → Missing values, duplicates, outliers, encoding, scaling, SMOTE
  ├── src/feature_engineering.py  → Polynomial features, PCA, binning, date features, clustering
  ├── src/train.py              → Model initialization, CV, hyperparameter tuning, MLflow tracking, SHAP
  ├── src/evaluate.py           → Metrics calculation, plot generation (classification + regression)
  └── src/predict.py            → Model loading, single/batch prediction, probability, intervals
```

### Pipeline Flow

```
1. Dataset Loading
    ↓
2. Data Validation (empty check, min rows, data types)
    ↓
3. Data Cleaning (missing values, duplicates)
    ↓
4. Preprocessing (encoding, scaling, outliers, SMOTE, log/Box-Cox)
    ↓
5. Feature Engineering (polynomial, date, frequency, PCA, binning, clustering)
    ↓
6. Feature Selection (variance threshold | correlation removal)
    ↓
7. Train-Test Split
    ↓
8. Cross-Validation (optional)
    ↓
9. Hyperparameter Tuning (optional: GridSearchCV / RandomizedSearchCV)
    ↓
10. Model Training
    ↓
11. Model Evaluation (metrics + plots + SHAP)
    ↓
12. MLflow Experiment Tracking
    ↓
13. Model Versioning & Save (.pkl + MLflow registry)
    ↓
14. (Optional) Export artifacts for deployment
```

---

## 🧪 Advanced Capabilities

| Feature | Config Section | Description |
|---------|---------------|-------------|
| **Outlier Handling** | `preprocessing.handle_outliers` | IQR / Z-score / Isolation Forest detection with clip/remove/winsorize strategies |
| **Imbalanced Data** | `preprocessing.handle_imbalanced` | SMOTE, ADASYN, random oversampling, random undersampling |
| **Log/Box-Cox Transform** | `preprocessing.apply_log_transform` | Transform skewed features to normal distribution |
| **PCA** | `feature_engineering.apply_pca` | Dimensionality reduction with auto component selection |
| **Binning** | `feature_engineering.create_binning_features` | Equal-width or quantile-based binning |
| **Clustering Features** | `feature_engineering.create_clustering_features` | KMeans cluster labels as features |
| **Text Features** | `feature_engineering.create_text_features` | String length, word count, unique word count |
| **Hyperparameter Tuning** | `hyperparameter_tuning.enabled` | GridSearchCV or RandomizedSearchCV |
| **Cross-Validation** | `cross_validation.enabled` | K-Fold cross-validation with scoring |
| **SHAP Explainability** | `explainability.enabled` | SHAP summary plots for model interpretability |
| **Model Comparison** | `--compare` CLI flag | Train & compare multiple algorithms in one run |
| **Artifact Export** | `--export` CLI flag | Export model + metadata for deployment |

---

## 📚 RAG Pipeline - Installation & Setup

### 1. Create a separate virtual environment (recommended)

```bash
# Windows
cd CARIVIX_AI_Model_Training
python -m venv venv_rag
venv_rag\Scripts\activate

# Linux/Mac
cd CARIVIX_AI_Model_Training
python3 -m venv venv_rag
source venv_rag/bin/activate
```

### 2. Install RAG dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements_rag.txt
```

### 3. (Optional) Install & start Ollama for local LLM

```bash
# Download and install Ollama from: https://ollama.com
# Then pull a model:
ollama pull llama3.1

# Start the Ollama server:
ollama server
```

---

## 🎯 RAG Pipeline - CLI Commands & Usage

### Basic Commands

```bash
# Index all documents in data/documents/ into FAISS vector store
python main_rag.py --index

# Query the indexed documents
python main_rag.py --query "What is CARIVIX AI?"

# Run comprehensive indexing & retrieval test
python main_rag.py --test

# Start an interactive Q&A session
python main_rag.py --interactive

# Show pipeline configuration and statistics
python main_rag.py --info
```

### All CLI Flags

| Flag | Description |
|------|-------------|
| `--index` | Index all documents into the FAISS vector database |
| `--query QUESTION` | Query indexed documents with a natural language question |
| `--test` | Run comprehensive indexing and retrieval test |
| `--interactive` | Start an interactive Q&A session |
| `--info` | Show pipeline configuration and statistics |
| `--documents PATH` | Documents directory (default: `data/documents/`) |
| `--vector-store PATH` | Vector store directory (default: `data/vector_store/`) |
| `--chunk-size INT` | Chunk size in characters (default: `500`) |
| `--chunk-overlap INT` | Chunk overlap in characters (default: `50`) |
| `--embedding-model NAME` | Embedding model name (default: `all-MiniLM-L6-v2`) |
| `--retrieval-k INT` | Number of chunks to retrieve (default: `5`) |
| `--llm-backend NAME` | LLM backend: `ollama` or `huggingface` (default: `ollama`) |
| `--llm-model NAME` | LLM model name (default depends on backend) |
| `--llm-url URL` | Ollama server URL (default: `http://localhost:11434`) |
| `--force` | Force re-indexing even if already indexed |
| `--verbose, -v` | Enable verbose (DEBUG) logging |
| `--output PATH` | Save query result to JSON file |

### Examples

```bash
# Index with custom chunk settings
python main_rag.py --index --chunk-size 1000 --chunk-overlap 100

# Query with more retrieved chunks
python main_rag.py --query "How does the platform handle data?" --retrieval-k 10

# Use a different LLM backend
python main_rag.py --query "What is economic analysis?" --llm-backend huggingface --llm-model microsoft/phi-2

# Run interactive session with custom documents
python main_rag.py --interactive --documents data/my_docs/

# Save query result to file
python main_rag.py --query "Tell me about market prediction" --output query_result.json

# Full test with verbose logging
python main_rag.py --test --verbose
```

---

## 🏗️ RAG Pipeline Architecture

### Modular Architecture

```
main_rag.py (CLI orchestrator)
   │
   ├── rag/loader.py           → Document loading (PDF, DOCX, TXT, CSV)
   ├── rag/splitter.py          → Text cleaning & semantic chunk splitting
   ├── rag/embeddings.py        → Dense vector embedding generation
   ├── rag/vector_store.py      → FAISS index creation, save, load, search
   ├── rag/retriever.py         → Query embedding + similarity search
   ├── rag/prompt_builder.py    → Context-aware prompt construction
   ├── rag/generator.py         → LLM response generation (Ollama / HuggingFace)
   └── rag/pipeline.py          → End-to-end pipeline orchestrator
```

### Pipeline Flow

```
Load Documents (PDF, DOCX, TXT, CSV)
    ↓
Clean & Normalize Text (whitespace, control chars)
    ↓
Split into Semantic Chunks (RecursiveCharacterTextSplitter)
    ↓
Generate Embeddings (Sentence Transformers - 384-dim)
    ↓
Create FAISS Vector Index (L2 distance)
    ↓
Save Index to Disk (index.faiss + index.pkl)
    ↓
User Query → Generate Query Embedding
    ↓
Similarity Search → Retrieve Top-K Chunks
    ↓
Build Context-Aware Prompt
    ↓
Generate Response via LLM (Ollama / HuggingFace)
    ↓
Return Answer + Retrieved Context
```

### RAG Module Details

| Module | File | Description |
|--------|------|-------------|
| **DocumentLoader** | `rag/loader.py` | Loads documents from PDF (pypdf), DOCX (python-docx), TXT (utf-8/latin-1), and CSV (pandas) into LangChain Document objects with metadata (source, page, file type). |
| **TextPreprocessor** | `rag/splitter.py` | Cleans and normalizes text: removes extra whitespace, normalizes line breaks, strips control characters, collapses blank lines. Optionally removes special characters. |
| **DocumentSplitter** | `rag/splitter.py` | Splits documents into semantic chunks using `RecursiveCharacterTextSplitter` with configurable chunk size (default: 500) and overlap (default: 50). Adds chunk tracking metadata (chunk_id, chunk_total). |
| **EmbeddingGenerator** | `rag/embeddings.py` | Generates 384-dimensional embeddings using Sentence Transformers (`all-MiniLM-L6-v2`). Supports batched processing, CPU/CUDA auto-detection, and model caching. |
| **VectorStore** | `rag/vector_store.py` | Manages FAISS vector index: create from documents+embeddings, save to disk (`index.faiss` + `index.pkl`), load from disk, and similarity search with configurable index type (L2/IP). |
| **Retriever** | `rag/retriever.py` | Embeds user query and searches FAISS index for top-K most similar chunks. Supports score thresholds, batch queries, and results formatting. |
| **PromptBuilder** | `rag/prompt_builder.py` | Builds structured prompts combining retrieved chunks as context with user query. Supports customizable system prompts, templates, and chat-style messages. |
| **ResponseGenerator** | `rag/generator.py` | Generates LLM responses via Ollama (Llama 3.1, Mistral) REST API or HuggingFace transformers pipeline. Abstract base class for adding new backends. |
| **RAGPipeline** | `rag/pipeline.py` | Orchestrates the complete RAG workflow: index_documents() for ingestion and query() for retrieval+generation. Provides pipeline info and result formatting. |

### Supported LLM Backends

| Backend | Models | Requirements |
|---------|--------|-------------|
| **Ollama** | llama3.1, mistral, phi, gemma, etc. | Ollama installed locally (`ollama serve`) + `requests` |
| **HuggingFace** | microsoft/phi-2, mistralai/Mistral-7B, etc. | `transformers` + `torch` |

---

## 🧪 Running Tests

### RAG Pipeline Tests

```bash
# Run all RAG pipeline tests
python -m pytest tests/test_rag_pipeline.py -v

# Run specific test classes
python -m pytest tests/test_rag_pipeline.py::TestDocumentLoader -v
python -m pytest tests/test_rag_pipeline.py::TestTextPreprocessor -v
python -m pytest tests/test_rag_pipeline.py::TestDocumentSplitter -v
python -m pytest tests/test_rag_pipeline.py::TestEmbeddingGenerator -v
python -m pytest tests/test_rag_pipeline.py::TestVectorStore -v
python -m pytest tests/test_rag_pipeline.py::TestRetriever -v
python -m pytest tests/test_rag_pipeline.py::TestPromptBuilder -v
python -m pytest tests/test_rag_pipeline.py::TestRAGPipeline -v

# Run specific test functions
python -m pytest tests/test_rag_pipeline.py::test_document_loading -v
python -m pytest tests/test_rag_pipeline.py::test_text_preprocessing -v

# Run with coverage report
python -m pytest tests/test_rag_pipeline.py -v --cov=rag --cov-report=term-missing
```

### Test Coverage

The test suite covers:
- **Document Loading** — TXT/CSV loading, nonexistent directories, file counting, supported extensions
- **Text Preprocessing** — Whitespace removal, newline normalization, control character stripping, word counting
- **Document Splitting** — Chunk creation, metadata propagation, empty list handling, config verification
- **Embedding Generation** — Output type/shape, batch processing, error handling, dimension info
- **Vector Store (FAISS)** — Index creation, save/load persistence, similarity search, error handling
- **Retriever** — Top-K retrieval, score thresholds, empty queries, result formatting
- **Prompt Builder** — Context/query injection, no-context handling, statistics
- **End-to-End Pipeline** — Full indexing flow, index+query cycle, save+load+query cycle

---

## 📈 Supported Algorithms (Full List)

### Classification (8 algorithms)

| Algorithm | Config Name | Default | Notes |
|-----------|------------|---------|-------|
| **LogisticRegression** | `LogisticRegression` | — | Linear model for binary/multiclass classification |
| **DecisionTree** | `DecisionTree` | — | Non-linear, interpretable tree-based model |
| **RandomForest** | `RandomForest` | ✅ Default | Robust ensemble of decision trees |
| **GradientBoosting** | `GradientBoosting` | — | Boosted ensemble, often higher accuracy |
| **XGBoost** | `XGBoost` | — | High-performance gradient boosting |
| **SVM** | `SVM` | — | Support Vector Machine with RBF kernel |
| **KNN** | `KNearestNeighbors` | — | K-Nearest Neighbors classifier |
| **NaiveBayes** | `NaiveBayes` | — | Gaussian Naive Bayes for probabilistic classification |

### Regression (9 algorithms)

| Algorithm | Config Name | Notes |
|-----------|------------|-------|
| **LinearRegression** | `LinearRegression` | Simple linear model |
| **DecisionTreeRegressor** | `DecisionTreeRegressor` | Non-linear regression tree |
| **RandomForestRegressor** | `RandomForestRegressor` | Ensemble regression forest |
| **GradientBoostingRegressor** | `GradientBoostingRegressor` | Boosted ensemble for regression |
| **XGBoostRegressor** | `XGBoostRegressor` | XGBoost for regression tasks |
| **SVR** | `SVR` | Support Vector Regression |
| **KNNRegressor** | `KNNRegressor` | K-Nearest Neighbors regression |
| **Ridge** | `Ridge` | L2-regularized linear regression |
| **Lasso** | `Lasso` | L1-regularized linear regression |
| **ElasticNet** | `ElasticNet` | Combined L1 + L2 regularization |

---

## 🏃 Baseline Pipeline

The baseline pipeline (`run_baseline_pipeline.py`) trains all applicable models on a dataset and compares results:

```bash
# Run baseline pipeline with default config
python run_baseline_pipeline.py

# Use custom config
python run_baseline_pipeline.py --config config/config.yaml

# Use a specific processed dataset
python run_baseline_pipeline.py --data data/processed/processed_20260730_125704.csv

# Override target column
python run_baseline_pipeline.py --target target

# Enable verbose logging
python run_baseline_pipeline.py --verbose
```

The baseline pipeline:
1. Loads a prepared dataset (or finds the latest processed dataset)
2. Validates data structure
3. Preprocesses features (missing values, encoding, scaling)
4. Applies feature engineering
5. Splits into train/test sets
6. Trains ALL baseline models for the detected task type
7. Evaluates each model with comprehensive metrics
8. Saves each model via the model registry
9. Tracks all experiments to `experiments/experiments.csv` and `experiments/model_metrics.json`
10. Identifies the best-performing model
11. Generates a comparison table
12. Verifies end-to-end data flow

---

## 📊 Experiment Tracking (Local, No MLflow)

In addition to MLflow tracking, the project provides lightweight local experiment tracking:

```bash
# Experiment records are saved to:
experiments/experiments.csv       # Tabular format (CSV)
experiments/model_metrics.json    # Structured format (JSON)

# Baseline pipeline summary:
experiments/baseline_summary_*.json

# Model comparison tables:
experiments/baseline_comparison_*.csv
```

Data tracked per run:
- Timestamp, model name, dataset name
- Hyperparameters used
- All evaluation metrics
- Training time
- Model file path

---

## 📁 Data Directory Details

### Input Datasets
The `data/` directory contains multiple datasets for model training:

| File | Format | Description |
|------|--------|-------------|
| `arxiv_data.csv` | CSV | ArXiv research paper metadata |
| `CARIVIX_Economic_Indicators_By_Dataset.xlsx` | Excel | Economic indicators dataset |
| `Economic_Trend_Analysis_Cleaned (1).csv` | CSV | Economic trend analysis |
| `Market_Analysis_1981_2025_Final_Cleaned.csv` | CSV | Market analysis data (1981-2025) |
| `Public Program Evaluation_final.csv` | CSV | Public program evaluation data |
| `Traffic Analysis_Final.csv` | CSV | Traffic analysis dataset |

### RAG Documents
The `data/documents/` directory contains sample documents for the RAG pipeline:

| File | Format | Description |
|------|--------|-------------|
| `sample_about_carivix.txt` | TXT | Overview of CARIVIX AI platform features |
| `sample_economics.txt` | TXT | Sample economics and market analysis content |
| `sample_report.docx` | DOCX | Sample report document for Q&A testing |

### Vector Store
The `data/vector_store/` directory stores the FAISS index after indexing:

| File | Description |
|------|-------------|
| `index.faiss` | FAISS binary index file with embedded vectors |
| `index.pkl` | Pickled document metadata aligned with index vectors |

---

## 🛠️ Development Guidelines

The codebase follows:

- **PEP 8** — Python style guide
- **Modular Design** — Single responsibility per module
- **Type Hints** — Full type annotations for all functions
- **Logging** — Comprehensive logging with file + console handlers (no `print` statements)
- **Exception Handling** — Proper error handling with retry decorator for fault tolerance
- **Docstrings** — Google-style docstrings for all public functions
- **Context Managers** — `Timer` context manager for performance profiling

---

## 📄 License

This project is part of the CARIVIX AI platform.

---

## 🤝 Support

For issues, questions, or contributions, please contact the CARIVIX AI team.


