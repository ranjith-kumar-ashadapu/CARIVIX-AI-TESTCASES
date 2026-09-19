import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Economic_Trend_Analysis_Cleaned.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Baseline_Models\Economic_Trend"
)

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 3. Identify the 2025 target column
# --------------------------------------------------

target_candidates = [
    "2025",
    "2025 [YR2025]",
    "2025_Value",
    "Latest_Year_Value"
]

target_column = None

for candidate in target_candidates:
    if candidate in df.columns:
        target_column = candidate
        break

if target_column is None:
    raise ValueError(
        "Could not find a 2025 target column. "
        "Please check the dataset column names."
    )

print("Target column:", target_column)

# --------------------------------------------------
# 4. Remove rows with missing target values
# --------------------------------------------------

df = df.dropna(subset=[target_column]).copy()

y = df[target_column]
X = df.drop(columns=[target_column])

# --------------------------------------------------
# 5. Remove columns that would cause target leakage
# --------------------------------------------------

leakage_columns = [
    "Latest_Value",
    "Latest_Year_Value",
    "Latest_Value_Log",
    "Latest_Year",
    "Latest_Year_Change",
    "Latest_Year_Change_Percentage",
    "Latest_Value_Status",
    "Latest_Value_Outlier_Flag",
    "Percentage_Change",
    "CAGR_Percentage"
]

leakage_columns = [
    column for column in leakage_columns
    if column in X.columns
]

X = X.drop(columns=leakage_columns)

# Remove any columns containing the target year
year_columns = [
    column for column in X.columns
    if str(column).startswith("2025")
]

X = X.drop(columns=year_columns)

# --------------------------------------------------
# 6. Identify feature types
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# --------------------------------------------------
# 7. Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# --------------------------------------------------
# 8. Create Random Forest baseline model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# --------------------------------------------------
# 9. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 11. Generate predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

# --------------------------------------------------
# 12. Calculate evaluation metrics
# --------------------------------------------------

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

results = pd.DataFrame({
    "Metric": ["R2 Score", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

print("\nEconomic Trend Baseline Results")
print(results)

# --------------------------------------------------
# 13. Save predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "Actual_2025_Value": y_test.values,
    "Predicted_2025_Value": y_pred
})

predictions.to_csv(
    os.path.join(
        output_folder,
        "Economic_Trend_Baseline_Predictions.csv"
    ),
    index=False
)

# --------------------------------------------------
# 14. Save results
# --------------------------------------------------

results.to_csv(
    os.path.join(
        output_folder,
        "Economic_Trend_Baseline_Results.csv"
    ),
    index=False
)

# --------------------------------------------------
# 15. Save trained model
# --------------------------------------------------

joblib.dump(
    pipeline,
    os.path.join(
        output_folder,
        "Economic_Trend_Baseline_Model.pkl"
    )
)

print("\nFiles saved successfully in:")
print(output_folder)
