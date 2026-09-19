# ==========================================
# ECONOMIC TREND FINAL TEST EVALUATION
# LEAKAGE-SAFE SIGNED LOG-TARGET MODEL
# ==========================================

import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
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

test_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Test.csv"
)

output_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Data_Splits"
    r"\Economic_Trend_Test_Predictions.csv"
)

target_column = "2025"

# ==========================================
# SIGNED LOG FUNCTIONS
# ==========================================

def signed_log_transform(values):
    return np.sign(values) * np.log1p(np.abs(values))

def signed_log_inverse(values):
    return np.sign(values) * np.expm1(np.abs(values))

# ==========================================
# CHECK FILES
# ==========================================

for file_path in [
    train_file,
    validation_file,
    test_file
]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File was not found:\n{file_path}"
        )

# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(train_file)
validation_df = pd.read_csv(validation_file)
test_df = pd.read_csv(test_file)

train_validation_df = pd.concat(
    [
        train_df,
        validation_df
    ],
    ignore_index=True
)

train_validation_df = (
    train_validation_df
    .dropna(subset=[target_column])
    .copy()
)

test_df = (
    test_df
    .dropna(subset=[target_column])
    .copy()
)

print("=" * 70)
print("ECONOMIC TREND FINAL TEST EVALUATION")
print("LEAKAGE-SAFE SIGNED LOG-TARGET MODEL")
print("=" * 70)

print("\nTraining Records:", len(train_validation_df))
print("Test Records:", len(test_df))

# ==========================================
# LEAKAGE FEATURES
# ==========================================

leakage_features = [
    "2025",
    "Latest_Year_Value",
    "Latest_Year_Change",
    "Latest_Year_Change_Percentage",
    "Latest_Value_Log",
    "Latest_Value_Outlier_Flag",

"Historical_Average",
    "Historical_Average_Log",
    "Historical_Median",
    "Historical_Minimum",
    "Historical_Maximum",
    "Historical_Standard_Deviation",
    "Historical_Range",
    "Historical_Average_Outlier_Flag",

"Percentage_Change",
    "Absolute_Change",
    "CAGR_Percentage",
    "Trend_Slope",
    "Latest_Value_Status",
    "Trend_Direction",

"Total_Missing_Values_Per_Row",
    "Total_Available_Values_Per_Row",
    "Missing_Value_Percentage_Per_Row",
]

existing_leakage_features = [
    column
    for column in leakage_features
    if column in train_validation_df.columns
]

print("\nFeatures Excluded to Prevent Leakage:")

for column in existing_leakage_features:
    print("-", column)

# ==========================================
# CREATE INPUT DATA
# ==========================================

X_train = train_validation_df.drop(
    columns=existing_leakage_features
)

X_test = test_df.drop(
    columns=existing_leakage_features,
    errors="ignore"
)

# Match test columns with training columns
X_test = X_test[X_train.columns]

# ==========================================
# TARGET DATA
# ==========================================

y_train_original = (
    train_validation_df[target_column]
    .to_numpy()
)

y_test_original = (
    test_df[target_column]
    .to_numpy()
)

y_train = signed_log_transform(
    y_train_original
)

y_test = signed_log_transform(
    y_test_original
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

X_test = X_test.drop(
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
# PREPROCESSING
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
# MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

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
# TRAIN FINAL MODEL
# ==========================================

print(
    "\nTraining final model on "
    "training + validation data..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Final model training completed."
)

# ==========================================
# TEST PREDICTIONS
# ==========================================

predictions_log = pipeline.predict(
    X_test
)

predictions_original = signed_log_inverse(
    predictions_log
)

# ==========================================
# TEST METRICS: SIGNED LOG SCALE
# ==========================================

r2_log = r2_score(
    y_test,
    predictions_log
)

mae_log = mean_absolute_error(
    y_test,
    predictions_log
)

rmse_log = np.sqrt(
    mean_squared_error(
        y_test,
        predictions_log
    )
)

print(
    "\nTest Metrics on Signed Log Scale:"
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
# TEST METRICS: ORIGINAL SCALE
# ==========================================

mae_original = mean_absolute_error(
    y_test_original,
    predictions_original
)

rmse_original = np.sqrt(
    mean_squared_error(
        y_test_original,
        predictions_original
    )
)

print(
    "\nTest Metrics on Original Scale:"
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
# SAVE TEST PREDICTIONS
# ==========================================

test_predictions = test_df.copy()

test_predictions[
    "Predicted_2025"
] = predictions_original

test_predictions[
    "Predicted_2025_Signed_Log"
] = predictions_log

test_predictions[
    "Actual_2025_Signed_Log"
] = y_test

test_predictions.to_csv(
    output_file,
    index=False
)

print(
    "\nTest predictions saved to:"
)

print(output_file)

print(
    "\n" + "=" * 70
)

print(
    "FINAL TEST EVALUATION COMPLETED"
)

print(
    "=" * 70
)
