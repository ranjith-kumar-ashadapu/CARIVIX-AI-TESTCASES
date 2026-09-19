import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. File paths
input_file = r"D:\OneDrive\Desktop\Day 3 Task\Market_Analysis_Feature_Engineering\Market_Analysis_Feature_Engineered.csv"

output_folder = r"D:\OneDrive\Desktop\Day 3 Task\Baseline_Models\Market_Analysis"
os.makedirs(output_folder, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# 3. Define target column
target_column = "2025 [YR2025]"

# Remove rows where the target is missing
df = df.dropna(subset=[target_column])

# 4. Select safe input features
# Use historical values only: 1981 to 2024
historical_columns = [
    column for column in df.columns
    if column[:4].isdigit()
    and int(column[:4]) <= 2024
]

categorical_columns = [
    "Country Name",
    "Country Code",
    "Series Name",
    "Series Code"
]

feature_columns = historical_columns + categorical_columns

X = df[feature_columns]
y = df[target_column]

# 5. Identify numerical and categorical columns
numeric_features = [
    column for column in historical_columns
    if column in X.columns
]

categorical_features = [
    column for column in categorical_columns
    if column in X.columns
]

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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
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

# 10. Make predictions
y_pred = pipeline.predict(X_test)

# 11. Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

results = pd.DataFrame({
    "Metric": ["R2 Score", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

print("\nMarket Analysis Baseline Results")
print(results)

# 12. Save predictions
predictions = pd.DataFrame({
    "Actual_2025_Value": y_test.values,
    "Predicted_2025_Value": y_pred
})

predictions.to_csv(
    os.path.join(output_folder, "Market_Analysis_Baseline_Predictions.csv"),
    index=False
)

# 13. Save evaluation metrics
results.to_csv(
    os.path.join(output_folder, "Market_Analysis_Baseline_Results.csv"),
    index=False
)

# 14. Save trained model
joblib.dump(
    pipeline,
    os.path.join(output_folder, "Market_Analysis_Baseline_Model.pkl")
)

print("\nFiles saved successfully in:")
print(output_folder)
