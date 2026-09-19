# ==========================================
# PUBLIC PROGRAM EVALUATION FEATURE IMPORTANCE
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
    r"\Government_Intelligence_Data_Splits"
    r"\Public_Program_Evaluation_Train.csv"
)

target_column = "Total No. of Workers"

# ==========================================
# LOAD DATA
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file was not found:\n{train_file}"
    )

df = pd.read_csv(train_file)
df = df.dropna(subset=[target_column]).copy()

print("=" * 70)
print("PUBLIC PROGRAM EVALUATION FEATURE IMPORTANCE")
print("=" * 70)

# ==========================================
# REMOVE TARGET AND LEAKAGE FEATURES
# ==========================================

leakage_features = [
    # Target
    "Total No. of Workers",

# Direct worker-derived features
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

# Expenditure-per-worker features use the target
    "Expenditure_Per_Worker_Lakhs",
    "Expenditure_Per_Worker_Rs",
]

existing_leakage_features = [
    column
    for column in leakage_features
    if column in df.columns
]

print("\nFeatures Excluded to Prevent Leakage:")
for column in existing_leakage_features:
    print("-", column)

X = df.drop(columns=existing_leakage_features)
y = df[target_column]

# ==========================================
# REMOVE CONSTANT COLUMNS
# ==========================================

constant_columns = [
    column
    for column in X.columns
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
# EXTRACT FEATURE IMPORTANCE
# ==========================================

trained_preprocessor = pipeline.named_steps["preprocessor"]
trained_model = pipeline.named_steps["model"]

feature_names = (
    trained_preprocessor.get_feature_names_out()
)

importance_values = trained_model.feature_importances_

importance_table = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_values
    }
)

# Remove transformer prefixes
importance_table["Feature"] = (
    importance_table["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

# Group one-hot encoded features by original feature
def get_original_feature(feature_name):
    for column in categorical_columns:
        if feature_name.startswith(column + "_"):
            return column
    return feature_name

importance_table["Original_Feature"] = (
    importance_table["Feature"]
    .apply(get_original_feature)
)

feature_importance = (
    importance_table
    .groupby("Original_Feature", as_index=False)
    ["Importance"]
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

feature_importance["Importance_Percentage"] = (
    feature_importance["Importance"] * 100
)

# ==========================================
# DISPLAY TOP FEATURES
# ==========================================

print("\nTop 20 Important Features:")
print(
    feature_importance.head(20).to_string(
        index=False
    )
)

# ==========================================
# SAVE RESULTS
# ==========================================

output_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Government_Intelligence_Data_Splits"
    r"\Public_Program_Evaluation_Feature_Importance.csv"
)

feature_importance.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)

# ==========================================
# PLOT TOP 15 FEATURES
# ==========================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        by="Importance",
        ascending=True
    )
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["Original_Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    "Top 15 Feature Importances - "
    "Public Program Evaluation"
)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("FEATURE-IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)
