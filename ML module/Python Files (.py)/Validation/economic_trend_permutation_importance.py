# ==========================================
# ECONOMIC TREND PERMUTATION IMPORTANCE
# FINAL LEAKAGE-SAFE SIGNED LOG-TARGET VERSION
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# FILE PATHS
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Train.csv"
)

validation_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Validation.csv"
)

output_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Permutation_Importance_Final.csv"
)

target_column = "2025"

# ==========================================
# SIGNED LOG TRANSFORMATION
# ==========================================

def signed_log_transform(values):
    return np.sign(values) * np.log1p(np.abs(values))

def signed_log_inverse(values):
    return np.sign(values) * np.expm1(np.abs(values))

# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

if not os.path.exists(validation_file):
    raise FileNotFoundError(
        f"Validation file was not found:\n{validation_file}"
    )

# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(train_file)
validation_df = pd.read_csv(validation_file)

train_df = train_df.dropna(
    subset=[target_column]
).copy()

validation_df = validation_df.dropna(
    subset=[target_column]
).copy()

print("=" * 70)
print("ECONOMIC TREND PERMUTATION IMPORTANCE")
print("FINAL LEAKAGE-SAFE SIGNED LOG-TARGET VERSION")
print("=" * 70)

# ==========================================
# COMPLETE LEAKAGE FEATURE LIST
# ==========================================

leakage_features = [
    # Target and latest-value features
    "2025",
    "Latest_Year_Value",
    "Latest_Year_Change",
    "Latest_Year_Change_Percentage",
    "Latest_Value_Log",
    "Latest_Value_Outlier_Flag",

# Historical summary features
    "Historical_Average",
    "Historical_Average_Log",
    "Historical_Median",
    "Historical_Minimum",
    "Historical_Maximum",
    "Historical_Standard_Deviation",
    "Historical_Range",
    "Historical_Average_Outlier_Flag",

# Trend and change features
    "Percentage_Change",
    "Absolute_Change",
    "CAGR_Percentage",
    "Trend_Slope",
    "Latest_Value_Status",
    "Trend_Direction",

# Row-level availability features
    "Total_Missing_Values_Per_Row",
    "Total_Available_Values_Per_Row",
    "Missing_Value_Percentage_Per_Row",
]

existing_leakage_features = [
    column
    for column in leakage_features
    if column in train_df.columns
]

print("\nFeatures Excluded to Prevent Leakage:")

for column in existing_leakage_features:
    print("-", column)

# ==========================================
# CREATE FEATURES AND TARGET
# ==========================================

X_train = train_df.drop(
    columns=existing_leakage_features
)

X_validation = validation_df.drop(
    columns=existing_leakage_features,
    errors="ignore"
)

# Ensure the validation data uses the same columns
# and order as the training data
X_validation = X_validation[
    X_train.columns
]

# Original target values
y_train_original = train_df[
    target_column
].to_numpy()

y_validation_original = validation_df[
    target_column
].to_numpy()

# Signed log target values
y_train = signed_log_transform(
    y_train_original
)

y_validation = signed_log_transform(
    y_validation_original
)

# ==========================================
# REMOVE CONSTANT COLUMNS
# ==========================================

constant_columns = [
    column
    for column in X_train.columns
    if X_train[column].nunique(
        dropna=False
    ) <= 1
]

if constant_columns:
    print("\nConstant Columns Removed:")

for column in constant_columns:
        print("-", column)

X_train = X_train.drop(
        columns=constant_columns
    )

X_validation = X_validation.drop(
        columns=constant_columns,
        errors="ignore"
    )

# ==========================================
# IDENTIFY FEATURE TYPES
# ==========================================

categorical_columns = (
    X_train
    .select_dtypes(
        include=["object", "category"]
    )
    .columns
    .tolist()
)

numerical_columns = (
    X_train
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

print(
    "\nNumber of Input Features:",
    X_train.shape[1]
)

print(
    "Numerical Features:",
    len(numerical_columns)
)

print(
    "Categorical Features:",
    len(categorical_columns)
)

# ==========================================
# NUMERICAL PIPELINE
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

# ==========================================
# CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

# ==========================================
# PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

# ==========================================
# COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# ==========================================
# TRAIN MODEL
# ==========================================

print(
    "\nTraining model using leakage-safe "
    "signed log target..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Model training completed."
)

# ==========================================
# VALIDATION PREDICTIONS
# ==========================================

predictions_log = pipeline.predict(
    X_validation
)

predictions_original = signed_log_inverse(
    predictions_log
)

# ==========================================
# SIGNED LOG-SCALE METRICS
# ==========================================

r2_log = r2_score(
    y_validation,
    predictions_log
)

mae_log = mean_absolute_error(
    y_validation,
    predictions_log
)

rmse_log = np.sqrt(
    mean_squared_error(
        y_validation,
        predictions_log
    )
)

print(
    "\nValidation Metrics on Signed Log Scale:"
)

print(
    "R-squared:",
    round(r2_log, 6)
)

print(
    "MAE:",
    round(mae_log, 6)
)

print(
    "RMSE:",
    round(rmse_log, 6)
)

# ==========================================
# ORIGINAL-SCALE METRICS
# ==========================================

mae_original = mean_absolute_error(
    y_validation_original,
    predictions_original
)

rmse_original = np.sqrt(
    mean_squared_error(
        y_validation_original,
        predictions_original
    )
)

print(
    "\nValidation Metrics on Original Scale:"
)

print(
    "MAE:",
    round(mae_original, 6)
)

print(
    "RMSE:",
    round(rmse_original, 6)
)

# ==========================================
# PERMUTATION IMPORTANCE
# ==========================================

print(
    "\nCalculating permutation importance..."
)

permutation_result = permutation_importance(
    pipeline,
    X_validation,
    y_validation,
    scoring="r2",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_table = pd.DataFrame(
    {
        "Feature": X_validation.columns,
        "Mean_Importance": (
            permutation_result
            .importances_mean
        ),
        "Standard_Deviation": (
            permutation_result
            .importances_std
        )
    }
)

importance_table = (
    importance_table
    .sort_values(
        by="Mean_Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_table.insert(
    0,
    "Rank",
    importance_table.index + 1
)

importance_table[
    "R2_Decrease_Percentage"
] = (
    importance_table["Mean_Importance"] * 100
)

# ==========================================
# DISPLAY TOP 20 FEATURES
# ==========================================

print(
    "\nTop 20 Validation-Based "
    "Important Features:"
)

print(
    importance_table
    .head(20)
    .to_string(index=False)
)

# ==========================================
# SAVE RESULTS
# ==========================================

importance_table.to_csv(
    output_file,
    index=False
)

print(
    "\nResults saved to:"
)

print(output_file)

# ==========================================
# CREATE TOP-15 PLOT
# ==========================================

top_features = (
    importance_table
    .head(15)
    .sort_values(
        by="Mean_Importance",
        ascending=True
    )
)

plt.figure(
    figsize=(11, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Mean_Importance"]
)

plt.xlabel(
    "Mean Permutation Importance"
)

plt.ylabel("Feature")

plt.title(
    "Top 15 Validation-Based Feature Importances"
    " - Economic Trend"
)

plt.tight_layout()

plt.show()

print(
    "\n" + "=" * 70
)

print(
    "FINAL LEAKAGE-SAFE PERMUTATION "
    "IMPORTANCE ANALYSIS COMPLETED"
)

print(
    "=" * 70
)
