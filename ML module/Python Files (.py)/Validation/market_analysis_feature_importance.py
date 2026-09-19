# ==========================================
# MARKET ANALYSIS FEATURE IMPORTANCE
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# FILE PATH
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Market_Analysis_Data_Splits"
    r"\Market_Analysis_Train.csv"
)

# ==========================================
# TARGET COLUMN
# ==========================================

target_column = "2025 [YR2025]"

# ==========================================
# LOAD DATA
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

df = pd.read_csv(train_file)

print("=" * 70)
print("MARKET ANALYSIS FEATURE IMPORTANCE")
print("=" * 70)

# ==========================================
# REMOVE MISSING TARGET VALUES
# ==========================================

df = df.dropna(subset=[target_column]).copy()

# ==========================================
# REMOVE TARGET-LEAKAGE FEATURES
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
    if column in df.columns
]

# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=leakage_features)
y = df[target_column]

# ==========================================
# REMOVE CONSTANT COLUMNS
# ==========================================

constant_columns = [
    column for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

if constant_columns:
    print("\nConstant Columns Removed:")
    for column in constant_columns:
        print("-", column)

X = X.drop(columns=constant_columns)

# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumber of Input Features:", X.shape[1])
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
    ],
    remainder="drop"
)

# ==========================================
# RANDOM FOREST MODEL
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

print("\nTraining Random Forest model...")
pipeline.fit(X, y)

print("Model training completed.")

# ==========================================
# GET TRANSFORMED FEATURE NAMES
# ==========================================

trained_preprocessor = pipeline.named_steps["preprocessor"]
trained_model = pipeline.named_steps["model"]

transformed_feature_names = (
    trained_preprocessor.get_feature_names_out()
)

transformed_importances = trained_model.feature_importances_

importance_table = pd.DataFrame(
    {
        "Transformed_Feature": transformed_feature_names,
        "Importance": transformed_importances
    }
)

# ==========================================
# CONVERT TRANSFORMED NAMES TO ORIGINAL NAMES
# ==========================================

def get_original_feature_name(feature_name):
    """
    Converts encoded feature names back to their
    original input feature names.
    """

    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", "")

    if feature_name.startswith("categorical__"):
        remaining_name = feature_name.replace(
            "categorical__",
            "",
            1
        )

    for column in categorical_columns:
            if remaining_name.startswith(column + "_"):
                return column

    return remaining_name

    return feature_name

importance_table["Original_Feature"] = (
    importance_table["Transformed_Feature"]
    .apply(get_original_feature_name)
)

# ==========================================
# AGGREGATE IMPORTANCE BY ORIGINAL FEATURE
# ==========================================

feature_importance = (
    importance_table
    .groupby("Original_Feature", as_index=False)["Importance"]
    .sum()
    .sort_values(
        by="Importance",
        ascending=False
    )
)

feature_importance["Importance_Percentage"] = (
    feature_importance["Importance"] * 100
)

feature_importance = feature_importance.reset_index(
    drop=True
)

feature_importance.index = feature_importance.index + 1
feature_importance.insert(
    0,
    "Rank",
    feature_importance.index
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nTop 20 Important Features:")
print(
    feature_importance.head(20).to_string(
        index=False
    )
)

print("\nComplete Feature-Importance Ranking:")
print(
    feature_importance.to_string(
        index=False
    )
)

# ==========================================
# DISPLAY TOP 15 FEATURES AS A CHART
# ==========================================

top_features = feature_importance.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Original_Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Feature Importances - Market Analysis"
)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("FEATURE-IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)
