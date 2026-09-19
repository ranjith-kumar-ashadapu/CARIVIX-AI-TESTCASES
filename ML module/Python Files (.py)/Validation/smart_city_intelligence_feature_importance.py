# ==========================================
# SMART CITY INTELLIGENCE FEATURE IMPORTANCE
# RANDOM FOREST + PERMUTATION IMPORTANCE
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# FILE PATHS
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Data_Splits"
    r"\Smart_City_Intelligence_Train.csv"
)

validation_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Data_Splits"
    r"\Smart_City_Intelligence_Validation.csv"
)

importance_output_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Data_Splits"
    r"\Smart_City_Intelligence_Feature_Importance.csv"
)

permutation_output_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Data_Splits"
    r"\Smart_City_Intelligence_Permutation_Importance.csv"
)

target_column = "Traffic_Condition"

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

# Remove rows with missing target values
train_df = train_df.dropna(
    subset=[target_column]
).copy()

validation_df = validation_df.dropna(
    subset=[target_column]
).copy()

print("=" * 80)
print("SMART CITY INTELLIGENCE FEATURE IMPORTANCE")
print("=" * 80)

print("\nTraining Records:", len(train_df))
print("Validation Records:", len(validation_df))
print("Target Column:", target_column)

# ==========================================
# TARGET CLASS DISTRIBUTION
# ==========================================

print("\nTraining Target Distribution:")
print(
    train_df[target_column]
    .value_counts()
    .to_string()
)

print("\nValidation Target Distribution:")
print(
    validation_df[target_column]
    .value_counts()
    .to_string()
)

# ==========================================
# REMOVE TARGET-DERIVED FEATURES
# ==========================================

target_derived_features = [
    # Target and encoded target
    "Traffic_Condition",
    "Traffic_Condition_ID",

# Features calculated by traffic-condition group
    "Average_Vehicles_by_Traffic_Condition",
    "Vehicles_vs_Traffic_Condition_Average",

# Direct traffic/congestion flags
    "Congestion_Risk_Flag",
    "High_Traffic_Flag",
    "Low_Speed_Flag",

# Categorised versions of traffic measurements
    "Speed_Category",
    "Speed_Category_ID",
    "Occupancy_Category",
    "Occupancy_Category_ID",
    "Vehicle_Count_Category",
    "Vehicle_Count_Category_ID",
]

existing_target_derived_features = [
    column
    for column in target_derived_features
    if column in train_df.columns
]

print(
    "\nFeatures Excluded Because They May Reveal "
    "or Derive the Target:"
)

for column in existing_target_derived_features:
    print("-", column)

# ==========================================
# CREATE FEATURES AND TARGET
# ==========================================

X_train = train_df.drop(
    columns=existing_target_derived_features
)

y_train = train_df[target_column]

X_validation = validation_df.drop(
    columns=existing_target_derived_features,
    errors="ignore"
)

y_validation = validation_df[target_column]

# ==========================================
# REMOVE RAW DATE/TIMESTAMP COLUMNS
# ==========================================

# Date components such as hour, day, and weekday
# are already present as separate columns.
# Raw timestamp/date strings are therefore removed.
raw_datetime_columns = [
    "Timestamp",
    "Date"
]

existing_datetime_columns = [
    column
    for column in raw_datetime_columns
    if column in X_train.columns
]

print("\nRaw Date/Time Columns Removed:")
for column in existing_datetime_columns:
    print("-", column)

X_train = X_train.drop(
    columns=existing_datetime_columns
)

X_validation = X_validation.drop(
    columns=existing_datetime_columns,
    errors="ignore"
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

print("\nConstant Columns Removed:")

if len(constant_columns) == 0:
    print("No constant columns found.")
else:
    for column in constant_columns:
        print("-", column)

X_train = X_train.drop(
    columns=constant_columns
)

X_validation = X_validation.drop(
    columns=constant_columns,
    errors="ignore"
)

# Keep identical feature order
X_validation = X_validation[
    X_train.columns
]

# ==========================================
# IDENTIFY COLUMN TYPES
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
    ],
    remainder="drop"
)

# ==========================================
# RANDOM FOREST CLASSIFIER
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    class_weight="balanced"
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
# TRAIN MODEL
# ==========================================

print("\nTraining Random Forest classifier...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")

# ==========================================
# VALIDATION PREDICTIONS
# ==========================================

validation_predictions = pipeline.predict(
    X_validation
)

# ==========================================
# VALIDATION METRICS
# ==========================================

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

precision_macro = precision_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)

print("\nValidation Metrics:")
print("Accuracy:", round(accuracy, 6))
print("Macro Precision:", round(precision_macro, 6))
print("Macro Recall:", round(recall_macro, 6))
print("Macro F1-score:", round(f1_macro, 6))
print("Weighted F1-score:", round(f1_weighted, 6))

print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        zero_division=0
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

class_labels = sorted(
    y_validation.unique()
)

confusion = confusion_matrix(
    y_validation,
    validation_predictions,
    labels=class_labels
)

confusion_table = pd.DataFrame(
    confusion,
    index=[
        f"Actual_{label}"
        for label in class_labels
    ],
    columns=[
        f"Predicted_{label}"
        for label in class_labels
    ]
)

print("Confusion Matrix:")
print(confusion_table.to_string())

# ==========================================
# TREE-BASED FEATURE IMPORTANCE
# ==========================================

trained_preprocessor = (
    pipeline.named_steps["preprocessor"]
)

trained_model = (
    pipeline.named_steps["model"]
)

transformed_feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

importance_values = (
    trained_model.feature_importances_
)

importance_table = pd.DataFrame(
    {
        "Transformed_Feature":
            transformed_feature_names,
        "Importance":
            importance_values
    }
)

# Remove preprocessing prefixes
importance_table[
    "Transformed_Feature"
] = (
    importance_table[
        "Transformed_Feature"
    ]
    .str.replace(
        "numeric__",
        "",
        regex=False
    )
    .str.replace(
        "categorical__",
        "",
        regex=False
    )
)

# Group one-hot encoded features by
# their original feature names
def get_original_feature_name(
    feature_name
):
    for column in categorical_columns:
        if feature_name.startswith(
            column + "_"
        ):
            return column

    return feature_name

importance_table[
    "Original_Feature"
] = (
    importance_table[
        "Transformed_Feature"
    ]
    .apply(get_original_feature_name)
)

feature_importance = (
    importance_table
    .groupby(
        "Original_Feature",
        as_index=False
    )["Importance"]
    .sum()
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_importance.insert(
    0,
    "Rank",
    feature_importance.index + 1
)

feature_importance[
    "Importance_Percentage"
] = (
    feature_importance["Importance"] * 100
)

print("\nTop 20 Random Forest Important Features:")
print(
    feature_importance
    .head(20)
    .to_string(index=False)
)

feature_importance.to_csv(
    importance_output_file,
    index=False
)

print(
    "\nRandom Forest feature-importance results saved to:"
)

print(importance_output_file)

# ==========================================
# PERMUTATION IMPORTANCE
# ==========================================

print(
    "\nCalculating validation permutation importance..."
)

permutation_result = permutation_importance(
    pipeline,
    X_validation,
    y_validation,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_table = pd.DataFrame(
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

permutation_table = (
    permutation_table
    .sort_values(
        by="Mean_Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

permutation_table.insert(
    0,
    "Rank",
    permutation_table.index + 1
)

permutation_table[
    "F1_Macro_Decrease"
] = (
    permutation_table["Mean_Importance"]
)

print(
    "\nTop 20 Validation-Based "
    "Permutation Features:"
)

print(
    permutation_table
    .head(20)
    .to_string(index=False)
)

permutation_table.to_csv(
    permutation_output_file,
    index=False
)

print(
    "\nPermutation-importance results saved to:"
)

print(permutation_output_file)

# ==========================================
# VISUALIZATION
# ==========================================

top_features = (
    permutation_table
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
    "Mean Decrease in Validation Macro F1"
)

plt.ylabel("Feature")

plt.title(
    "Top 15 Smart City Feature Importances"
)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print(
    "SMART CITY FEATURE-IMPORTANCE ANALYSIS COMPLETED"
)
print("=" * 80)
