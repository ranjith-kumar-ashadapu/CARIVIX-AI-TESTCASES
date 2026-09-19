import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Smart_City_Intelligence_Feature_Engineering"
    r"\Smart_City_Intelligence_Feature_Engineered.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Baseline_Models\Smart_City_Intelligence"
)

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 3. Define target
# --------------------------------------------------

target_column = "Traffic_Condition"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' was not found."
    )

df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
y = df[target_column]

# --------------------------------------------------
# 4. Remove leakage and unsuitable columns
# --------------------------------------------------

columns_to_drop = [
    "Timestamp",
    "Date",

# Direct encoded version of the target
    "Traffic_Condition_ID",

# Features calculated directly by traffic condition
    "Average_Vehicles_by_Traffic_Condition",
    "Vehicles_vs_Traffic_Condition_Average"
]

columns_to_drop = [
    column for column in columns_to_drop
    if column in X.columns
]

X = X.drop(columns=columns_to_drop)

# --------------------------------------------------
# 5. Identify feature types
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# --------------------------------------------------
# 6. Build preprocessing pipeline
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
# 7. Create Random Forest classifier
# --------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", classifier)
    ]
)

# --------------------------------------------------
# 8. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 10. Generate predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

# --------------------------------------------------
# 11. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

print("\nSmart City Intelligence Baseline Results")
print(results)

print("\nClassification Report")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 12. Save predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "Actual_Traffic_Condition": y_test.values,
    "Predicted_Traffic_Condition": y_pred
})

predictions.to_csv(
    os.path.join(
        output_folder,
        "Smart_City_Intelligence_Baseline_Predictions.csv"
    ),
    index=False
)

# --------------------------------------------------
# 13. Save evaluation results
# --------------------------------------------------

results.to_csv(
    os.path.join(
        output_folder,
        "Smart_City_Intelligence_Baseline_Results.csv"
    ),
    index=False
)

# --------------------------------------------------
# 14. Save trained model
# --------------------------------------------------

joblib.dump(
    pipeline,
    os.path.join(
        output_folder,
        "Smart_City_Intelligence_Baseline_Model.pkl"
    )
)

print("\nFiles saved successfully in:")
print(output_folder)
