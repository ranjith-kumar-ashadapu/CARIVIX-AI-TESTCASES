# ==========================================
# RESEARCH INTELLIGENCE TF-IDF CLASSIFICATION
# ==========================================

import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# FILE PATHS
# ==========================================

train_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Data_Splits"
    r"\Research_Intelligence_Train.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Data_Splits"
)

feature_importance_file = os.path.join(
    output_folder,
    "Research_Intelligence_TFIDF_Feature_Importance.csv"
)

prediction_file = os.path.join(
    output_folder,
    "Research_Intelligence_TFIDF_Predictions.csv"
)

# ==========================================
# CHECK FILE
# ==========================================

if not os.path.exists(train_file):
    raise FileNotFoundError(
        "Training file was not found:\n"
        + train_file
    )

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(train_file)

print("=" * 80)
print("RESEARCH INTELLIGENCE TF-IDF CLASSIFICATION")
print("=" * 80)

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# REQUIRED COLUMNS
# ==========================================

text_columns = [
    "titles",
    "summaries",
    "terms"
]

target_column = "Primary_Category"

required_columns = [
    "titles",
    "summaries",
    "terms",
    "Primary_Category"
]

missing_columns = []

for column in required_columns:
    if column not in df.columns:
        missing_columns.append(column)

if len(missing_columns) > 0:
    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )

# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================

def clean_text(value):
    """
    Convert a value into clean lowercase text.
    """

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()

# ==========================================
# COMBINE TEXT COLUMNS
# ==========================================

df["Combined_Text"] = (
    df["titles"].apply(clean_text)
    + " "
    + df["summaries"].apply(clean_text)
    + " "
    + df["terms"].apply(clean_text)
)

df["Combined_Text"] = (
    df["Combined_Text"]
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)

# Remove records without text
df = df[
    df["Combined_Text"].str.len() > 0
].copy()

print("\nUsable Records:")
print(len(df))

# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print("\nTarget Column:")
print(target_column)

print("\nTarget Distribution:")
print(
    df[target_column]
    .value_counts()
    .to_string()
)

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X = df["Combined_Text"]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:")
print(len(X_train))

print("\nTesting Records:")
print(len(X_test))

# ==========================================
# TF-IDF VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

feature_names = vectorizer.get_feature_names_out()

print("\nTF-IDF Training Matrix Shape:")
print(X_train_tfidf.shape)

print("\nNumber of TF-IDF Features:")
print(len(feature_names))

# ==========================================
# LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)

# ==========================================
# MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(round(accuracy, 4))

print("\nModel Accuracy Percentage:")
print(round(accuracy * 100, 2), "%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

# ==========================================
# CONFUSION MATRIX
# ==========================================

class_labels = model.classes_

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

confusion_df = pd.DataFrame(
    matrix,
    index=[
        "Actual_" + str(label)
        for label in class_labels
    ],
    columns=[
        "Predicted_" + str(label)
        for label in class_labels
    ]
)

print("\nConfusion Matrix:")
print(confusion_df.to_string())

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\nTop TF-IDF Features by Category:")

importance_records = []

for class_index in range(
    len(model.classes_)
):
    class_name = model.classes_[class_index]

coefficients = model.coef_[class_index]

sorted_indices = np.argsort(
        coefficients
    )[::-1]

print("\nCategory:", class_name)

for rank in range(30):
        feature_index = sorted_indices[rank]

feature_name = feature_names[
            feature_index
        ]

importance_value = coefficients[
            feature_index
        ]

print(
            rank + 1,
            "| Feature:",
            feature_name,
            "| Importance:",
            round(importance_value, 6)
        )

importance_records.append(
            {
                "Category": class_name,
                "Rank": rank + 1,
                "Feature": feature_name,
                "Importance": importance_value
            }
        )

# ==========================================
# SAVE FEATURE IMPORTANCE FILE
# ==========================================

importance_df = pd.DataFrame(
    importance_records
)

importance_df.to_csv(
    feature_importance_file,
    index=False
)

print("\nFeature-Importance File Saved:")
print(feature_importance_file)

# ==========================================
# SAVE PREDICTIONS FILE
# ==========================================

prediction_df = pd.DataFrame(
    {
        "Actual_Category": y_test.values,
        "Predicted_Category": y_pred
    }
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nPrediction File Saved:")
print(prediction_file)

# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n" + "=" * 80)
print("TF-IDF ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
