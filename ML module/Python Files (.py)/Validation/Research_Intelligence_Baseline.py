import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Feature_Engineering"
    r"\Research_Intelligence_Feature_Engineered.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Baseline_Models\Research_Intelligence"
)

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 3. Define text and target columns
# --------------------------------------------------

text_columns = [
    "titles",
    "summaries",
    "terms"
]

target_column = "Primary_Category"

required_columns = text_columns + [target_column]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# --------------------------------------------------
# 4. Remove missing values
# --------------------------------------------------

df = df.dropna(subset=required_columns).copy()

# Combine text columns
df["combined_text"] = (
    df["titles"].astype(str)
    + " "
    + df["summaries"].astype(str)
    + " "
    + df["terms"].astype(str)
)

X = df["combined_text"]
y = df[target_column]

# --------------------------------------------------
# 5. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 6. Create Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=100000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

# --------------------------------------------------
# 9. Evaluation Metrics
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

print("\nResearch Intelligence Baseline Results")
print(results)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 10. Save Predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "Actual_Primary_Category": y_test.values,
    "Predicted_Primary_Category": y_pred
})

predictions.to_csv(
    os.path.join(
        output_folder,
        "Research_Intelligence_Baseline_Predictions.csv"
    ),
    index=False
)

# --------------------------------------------------
# 11. Save Evaluation Results
# --------------------------------------------------

results.to_csv(
    os.path.join(
        output_folder,
        "Research_Intelligence_Baseline_Results.csv"
    ),
    index=False
)

# --------------------------------------------------
# 12. Save Model
# --------------------------------------------------

joblib.dump(
    pipeline,
    os.path.join(
        output_folder,
        "Research_Intelligence_Baseline_Model.pkl"
    )
)

# --------------------------------------------------
# 13. Save TF-IDF Feature Importance
# --------------------------------------------------

tfidf = pipeline.named_steps["tfidf"]
classifier = pipeline.named_steps["classifier"]

feature_names = tfidf.get_feature_names_out()

# Binary Classification
if len(classifier.classes_) == 2:

    importance = classifier.coef_[0]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": importance,
        "Absolute_Importance": abs(importance)
    })

# Multi-class Classification
else:

    rows = []

    for class_index, class_name in enumerate(classifier.classes_):

        class_coefficients = classifier.coef_[class_index]

        class_features = pd.DataFrame({
            "Class": class_name,
            "Feature": feature_names,
            "Coefficient": class_coefficients,
            "Absolute_Importance": abs(class_coefficients)
        })

        rows.append(class_features)

    feature_importance = pd.concat(
        rows,
        ignore_index=True
    )

feature_importance = feature_importance.sort_values(
    by="Absolute_Importance",
    ascending=False
)

feature_importance.to_csv(
    os.path.join(
        output_folder,
        "Research_Intelligence_TFIDF_Feature_Importance.csv"
    ),
    index=False
)

print("\nFiles saved successfully in:")
print(output_folder)