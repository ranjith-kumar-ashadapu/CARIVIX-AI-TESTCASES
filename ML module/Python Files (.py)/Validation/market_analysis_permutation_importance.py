# ==========================================
# MARKET ANALYSIS PERMUTATION IMPORTANCE
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
    r"\Market_Analysis_Data_Splits"
    r"\Market_Analysis_Train.csv"
)

validation_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Market_Analysis_Data_Splits"
    r"\Market_Analysis_Validation.csv"
)

target_column = "2025 [YR2025]"

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

# Remove rows with missing target values
train_df = train_df.dropna(
    subset=[target_column]
).copy()

validation_df = validation_df.dropna(
    subset=[target_column]
).copy()

print("=" * 70)
print("MARKET ANALYSIS PERMUTATION IMPORTANCE")
print("=" * 70)

# ==========================================
# REMOVE LEAKAGE FEATURES
# ==========================================

leakage_features = [
    "2025 [YR2025]",
    "Latest_Year_Value",
    "Latest_Value_Log",
    "Latest_Year_Change",
    "Latest_Year_Change_Percentage",
    "Absolute_Change",
    "Percentage_Change",
    "CAGR_Percentage",
    "Trend_Slope",
    "Trend_Direction",
    "Latest_Value_Status",
    "Historical_Average",
    "Historical_Median",
    "Historical_Minimum",
    "Historical_Maximum",
    "Historical_Standard_Deviation",
    "Historical_Range",
    "Average_Value_Log",
]

leakage_features = [
    column for column in leakage_features
    if column in train_df.columns
]

# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

X_train = train_df.drop(
    columns=leakage_features
)

y_train = train_df[target_column]

X_validation = validation_df.drop(
    columns=leakage_features,
    errors="ignore"
)

y_validation = validation_df[target_column]

# Keep validation columns in the same order as training
X_validation = X_validation[X_train.columns]

# ==========================================
# REMOVE CONSTANT COLUMNS
# ==========================================

constant_columns = [
    column for column in X_train.columns
    if X_train[column].nunique(dropna=False) <= 1
]

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
    ],
    remainder="drop"
)

# ==========================================
# MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
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

baseline_score = pipeline.score(
    X_validation,
    y_validation
)

print("\nBaseline Validation R-squared:")
print(round(baseline_score, 6))

# ==========================================
# PERMUTATION IMPORTANCE
# ==========================================

print("\nCalculating permutation importance...")

permutation_result = permutation_importance(
    pipeline,
    X_validation,
    y_validation,
    n_repeats=5,
    random_state=42,
    scoring="r2",
    n_jobs=-1
)

permutation_table = pd.DataFrame(
    {
        "Feature": X_validation.columns,
        "Mean_Importance": permutation_result.importances_mean,
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

# Convert importance to percentage
permutation_table["Importance_Percentage"] = (
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
    r"\Market_Analysis_Data_Splits"
    r"\Market_Analysis_Permutation_Importance.csv"
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

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Mean_Importance"]
)

plt.xlabel("Mean Permutation Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Validation-Based Feature Importances"
)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)
