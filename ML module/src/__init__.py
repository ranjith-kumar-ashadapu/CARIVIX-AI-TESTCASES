"""
CARIVIX AI - Model Training Pipeline
=====================================

Source package for the machine learning model training pipeline.

Modules:
    utils                - Logging, configuration loading, helper utilities
    preprocess           - Data preprocessing (missing values, encoding, scaling)
    feature_engineering  - Feature creation and selection
    train                - Model training with MLflow experiment tracking
    evaluate             - Model evaluation metrics and visualizations
    predict              - Prediction on new data using trained models
    experiment_tracking  - Experiment tracking (experiments.csv, model_metrics.json)
    model_registry       - Model save/load/list registry
    model_dispatcher     - Auto task type detection & baseline model initialization
    baseline_pipeline    - Complete baseline model training orchestration
"""

__version__ = "1.2.0"
__author__ = "CARIVIX AI"

