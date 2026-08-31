# Phase 0: Project Setup

## Overview
Phase 0 establishes a clean, modular repository structure for **GlassBox**, an explainable fraud detection platform built for the Kaggle Credit Card Fraud Detection dataset.

## What Was Set Up
- **`/data`**: Dedicated directory for dataset files. The local 150MB `creditcard.csv` dataset is placed here for local development and processing.
- **`/src/data`**: Python package for data ingestion, cleaning, feature scaling, and train/test splitting routines.
- **`/src/models`**: Python package for training classification models (such as XGBoost, Random Forest, Logistic Regression) and computing SHAP explanation values.
- **`/src/api`**: Python package for the FastAPI backend service that will serve real-time predictions and explanation payloads.
- **`/web`**: Directory for the frontend web application where users can upload transaction files or test single transactions interactively.
- **`/docs` & `/docs/phase-notes`**: Documentation workspace, including iterative phase summaries to document progress across each development milestone.
- **`requirements.txt`**: Standardized list of Python dependencies:
  - `pandas`: Tabular data manipulation and analysis
  - `scikit-learn`: Preprocessing, metrics, and baseline modeling
  - `xgboost`: High-performance gradient boosted decision trees
  - `shap`: Explainable AI (XAI) feature attribution values
  - `imbalanced-learn`: Techniques (e.g., SMOTE / undersampling) for severely skewed fraud labels
  - `fastapi` & `uvicorn`: Asynchronous REST API framework and ASGI server
  - `python-multipart`: Handling multipart form and file uploads in FastAPI
- **`.gitignore`**: Configured to exclude raw dataset CSVs (avoiding bloat in git history), Python bytecode (`__pycache__`), virtual environment folders (`.venv`), and OS temporary files, while preserving model artifacts and code.
- **`README.md`**: High-level overview of the GlassBox project purpose and architecture.

## Why It Was Set Up This Way
- **Separation of Concerns**: Keeping data preparation, model training, API serving, and frontend UI in distinct modules prevents tight coupling and makes development faster and easier to test.
- **Git Hygiene**: Datasets of ~150MB do not belong in Git repositories because they slow down cloning and push operations. The `.gitignore` ensures only clean source code, configuration, and documentation are version-controlled.
- **Reproducibility**: Clear dependencies in `requirements.txt` make it simple for anyone on the team or judges/reviewers to install and run the project in any fresh environment.
