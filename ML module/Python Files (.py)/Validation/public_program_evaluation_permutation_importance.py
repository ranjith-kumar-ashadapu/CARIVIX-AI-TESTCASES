# ==========================================
# PUBLIC PROGRAM EVALUATION
# PERMUTATION IMPORTANCE
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ==========================================
# FILE PATHS
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Government_Intelligence_Data_Splits"
    r"\Public_Program_Evaluation_Train.csv"
)

validation_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Government_Intelligence_Data_Splits"
    r"\Public_Program_Evaluation_Validation.csv"
)

target_column = "Total No. of Workers"

# ==========================================
# LOAD DATA
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

if not os.path.exists(validation_file):
    raise FileNotFoundError(
        f"Validation file was not found:\n{validation_file}"
    )

train_df = pd.read_csv(train_file)
validation_df = pd.read_csv(validation_file)

train_df = train_df.dropna(
    subset=[target_column]
).copy()

validation_df = validation_df.dropna(
    subset=[target_column]
).copy()

print("=" * 70)
print("PUBLIC PROGRAM EVALUATION PERMUTATION IMPORTANCE")
print("=" * 70)

# ==========================================
# REMOVE LEAKAGE FEATURES
# ==========================================

leakage_features = [
    "Total No. of Workers",
    "Workers_Log",
    "Workers_Missing_Flag",
    "State_Total_Workers",
    "State_Average_Workers",
    "Worker_Share_of_State",
    "Workers_Per_Expenditure_Lakh",
    "Worker_Wage_Ratio",
    "Worker_Category",
    "Workers_Outlier_Flag",
    "Workers_vs_State_Median",
    "Expenditure_Per_Worker_Lakhs",
    "Expenditure_Per_Worker_Rs",
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
# CREATE TRAINING AND VALIDATION DATA
# ==========================================

X_train = train_df.drop(
    columns=existing_leakage_features
)

y_train = train_df[target_column]

X_validation = validation_df.drop(
    columns=existing_leakage_features,
    errors="ignore"
)

y_validation = validation_df[target_column]

# Keep identical column order
X_validation = X_validation[X_train.columns]

# ==========================================
# REMOVE CONSTANT TRAINING COLUMNS
# ==========================================

constant_columns = [
    column
    for column in X_train.columns
    if X_train[column].nunique(dropna=False) <= 1
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
# IDENTIFY COLUMN TYPES
# ==========================================

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumber of Input Features:", X_train.shape[1])
print("Numerical Features:", len(numerical_columns))
print("Categorical Features:", len(categorical_columns))

# ==========================================
# PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
# MODEL PIPELINE
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model on training data...")
pipeline.fit(X_train, y_train)

print("Model training completed.")

# ==========================================
# BASELINE VALIDATION SCORE
# ==========================================

baseline_r2 = pipeline.score(
    X_validation,
    y_validation
)

print("\nBaseline Validation R-squared:")
print(round(baseline_r2, 6))

# ==========================================
# PERMUTATION IMPORTANCE
# ==========================================

print("\nCalculating permutation importance...")

permutation_result = permutation_importance(
    pipeline,
    X_validation,
    y_validation,
    scoring="r2",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_table = pd.DataFrame(
    {
        "Feature": X_validation.columns,
        "Mean_Importance": (
            permutation_result.importances_mean
        ),
        "Standard_Deviation": (
            permutation_result.importances_std
        )
    }
)

permutation_table = permutation_table.sort_values(
    by="Mean_Importance",
    ascending=False
).reset_index(drop=True)

permutation_table.insert(
    0,
    "Rank",
    permutation_table.index + 1
)

# This is a performance decrease, not a percentage share
permutation_table["R2_Decrease_Percentage"] = (
    permutation_table["Mean_Importance"] * 100
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nTop 20 Validation-Based Important Features:")
print(
    permutation_table.head(20).to_string(
        index=False
    )
)

# ==========================================
# SAVE RESULTS
# ==========================================

output_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Government_Intelligence_Data_Splits"
    r"\Public_Program_Evaluation_Permutation_Importance.csv"
)

permutation_table.to_csv(
    output_file,
    index=False
)

print("\nPermutation-importance results saved to:")
print(output_file)

# ==========================================
# VISUALIZATION
# ==========================================

top_features = permutation_table.head(15).copy()

top_features = top_features.sort_values(
    by="Mean_Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["Feature"],
    top_features["Mean_Importance"]
)

plt.xlabel("Mean Permutation Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Validation-Based Feature Importances - "
    "Public Program Evaluation"
)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)
