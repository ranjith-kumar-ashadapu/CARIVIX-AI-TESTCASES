import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

project_root = Path(__file__).resolve().parents[2]
input_file = project_root / "data" / "Public Program Evaluation_final.csv"
output_folder = project_root / "experiments" / "Public_Program_Evaluation_Baseline"

os.makedirs(output_folder, exist_ok=True)

# 2. Load data
df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# 3. Define target
target_column = "Total Exp(Rs. in Lakhs.)"

y = df[target_column]
X = df.drop(columns=[target_column])

# 4. Remove direct expenditure components
# These values are directly related to Total Exp
leakage_columns = [
    "Wages(Rs. In Lakhs)",
    "Material and skilled Wages(Rs. In Lakhs)",
    "Total Adm Expenditure (Rs. in Lakhs.)"
]

X = X.drop(
    columns=[column for column in leakage_columns if column in X.columns]
)

# 5. Identify feature types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 6. Preprocessing
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

# 7. Create baseline model
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

# 8. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 9. Train the model
pipeline.fit(X_train, y_train)

# 10. Generate predictions
y_pred = pipeline.predict(X_test)

# 11. Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

results = pd.DataFrame({
    "Metric": ["R2 Score", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

print("\nPublic Program Evaluation Baseline Results")
print(results)

# 12. Save predictions
predictions = pd.DataFrame({
    "Actual_Total_Expenditure": y_test.values,
    "Predicted_Total_Expenditure": y_pred
})

predictions.to_csv(
    os.path.join(
        output_folder,
        "Public_Program_Evaluation_Baseline_Predictions.csv"
    ),
    index=False
)

# 13. Save results
results.to_csv(
    os.path.join(
        output_folder,
        "Public_Program_Evaluation_Baseline_Results.csv"
    ),
    index=False
)

# 14. Save model
joblib.dump(
    pipeline,
    os.path.join(
        output_folder,
        "Public_Program_Evaluation_Baseline_Model.pkl"
    )
)

print("\nFiles saved successfully in:")
print(output_folder)
