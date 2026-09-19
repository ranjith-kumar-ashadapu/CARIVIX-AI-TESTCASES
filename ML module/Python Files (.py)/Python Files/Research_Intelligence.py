# ============================================================
# RESEARCH INTELLIGENCE
# ARXIV RESEARCH PAPER ANALYSIS AND CLASSIFICATION
# Dataset: arxiv_data.csv
# Target: Research Categories (terms)
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ast
from collections import Counter

# ============================================================
# 2. LOAD DATASET
# ============================================================

project_root = Path(__file__).resolve().parents[2]
dataset_path = project_root / "data" / "arxiv_data.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print(
    "\nShape after removing duplicates:",
    df.shape
)

# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "titles",
    "summaries",
    "terms"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
        +
        "\nAvailable columns: "
        + str(df.columns.tolist())
    )

# ============================================================
# 6. CLEAN TEXT DATA
# ============================================================

df = df[required_columns].copy()

df["titles"] = (
    df["titles"]
    .astype(str)
    .str.strip()
)

df["summaries"] = (
    df["summaries"]
    .astype(str)
    .str.strip()
)

df["terms"] = (
    df["terms"]
    .astype(str)
    .str.strip()
)

# Remove empty records

df = df[
    (df["titles"] != "")
    &
    (df["summaries"] != "")
    &
    (df["terms"] != "")
]

print(
    "\nShape after cleaning:",
    df.shape
)

# ============================================================
# 7. CREATE TEXT FEATURE
# ============================================================

df["text"] = (
    df["titles"]
    +
    " "
    +
    df["summaries"]
)

# ============================================================
# 8. SUMMARY LENGTH ANALYSIS
# ============================================================

df["Summary_Length"] = (
    df["summaries"]
    .str.split()
    .str.len()
)

print("\nSummary Length Statistics:")
print(
    df["Summary_Length"]
    .describe()
)

plt.figure(figsize=(8,5))

plt.hist(
    df["Summary_Length"],
    bins=30
)

plt.title(
    "Distribution of Research Summary Length"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()

# ============================================================
# 9. FINAL CLEAN DATASET PREVIEW
# ============================================================

print("\nFinal Dataset:")
print(df.head())

print("\nFinal Shape:")
print(df.shape)

# ============================================================
# 10. EXTRACT RESEARCH CATEGORIES
# ============================================================

all_terms = []

for item in df["terms"]:

    try:

        categories = ast.literal_eval(item)

        if isinstance(categories, list):

            all_terms.extend(categories)

    except:

        continue

term_counts = Counter(all_terms)

category_df = pd.DataFrame(
    term_counts.items(),
    columns=[
        "Research_Category",
        "Paper_Count"
    ]
)

category_df = category_df.sort_values(
    by="Paper_Count",
    ascending=False
)

print("\nTop Research Categories:")
print(category_df.head(20))

# ============================================================
# 11. TOP 10 RESEARCH CATEGORIES
# ============================================================

top_categories = category_df.head(10)

plt.figure(figsize=(10,5))

plt.bar(
    top_categories["Research_Category"],
    top_categories["Paper_Count"]
)

plt.title(
    "Top 10 Research Categories"
)

plt.xlabel(
    "Research Category"
)

plt.ylabel(
    "Number of Papers"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()

# ============================================================
# 12. CATEGORY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=top_categories,
    x="Paper_Count",
    y="Research_Category"
)

plt.title(
    "Research Category Distribution"
)

plt.xlabel(
    "Number of Papers"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.show()

# ============================================================
# 13. TITLE LENGTH ANALYSIS
# ============================================================

df["Title_Length"] = (
    df["titles"]
    .str.split()
    .str.len()
)

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Title_Length",
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Research Paper Title Length"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()

# ============================================================
# 14. TEXT LENGTH COMPARISON
# ============================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Title_Length",
    y="Summary_Length"
)

plt.title(
    "Title Length vs Summary Length"
)

plt.xlabel(
    "Title Length"
)

plt.ylabel(
    "Summary Length"
)

plt.tight_layout()

plt.show()

# ============================================================
# 15. CATEGORY COUNT INFORMATION
# ============================================================

df["Category_Count"] = df["terms"].apply(
    lambda x:
        len(ast.literal_eval(x))
        if isinstance(x, str)
        else 0
)

print("\nCategory Count Statistics:")
print(
    df["Category_Count"]
    .describe()
)

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Category_Count",
    bins=20
)

plt.title(
    "Number of Categories per Research Paper"
)

plt.xlabel(
    "Number of Categories"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.show()

# ============================================================
# 16. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

from sklearn.model_selection import train_test_split

# Features
X = df["text"].copy()

# Target
y = df["terms"].copy()

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# ============================================================
# 17. CLEAN TARGET LABELS
# ============================================================

def clean_labels(value):

    try:

        labels = ast.literal_eval(value)

        if isinstance(labels, list):

            return labels[0]

        else:

            return str(value)

    except:

        return str(value)

y = y.apply(clean_labels)

print("\nSample Target Labels:")
print(y.head())

print("\nNumber of Unique Categories:")
print(y.nunique())

# ============================================================
# 18. TRAIN VALIDATION TEST SPLIT
# ============================================================

class_counts = y.value_counts()

can_stratify = (
    len(class_counts) > 1
    and class_counts.min() >= 2
)

if can_stratify:

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    temp_counts = y_temp.value_counts()

    can_stratify_temp = (
        len(temp_counts) > 1
        and temp_counts.min() >= 2
    )

    if can_stratify_temp:

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )

    else:

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42
        )

else:

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42
    )

print("\nDataset Split")
print("---------------------------")

print(
    "Training Set:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation Set:",
    X_val.shape,
    y_val.shape
)

print(
    "Testing Set:",
    X_test.shape,
    y_test.shape
)

# ============================================================
# 19. TF-IDF FEATURE EXTRACTION
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_val_tfidf = vectorizer.transform(
    X_val
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("\nTF-IDF Feature Shape")

print(
    "Training:",
    X_train_tfidf.shape
)

print(
    "Validation:",
    X_val_tfidf.shape
)

print(
    "Testing:",
    X_test_tfidf.shape
)

# ============================================================
# 20. FEATURE IMPORTANCE USING CHI-SQUARE TEST
# ============================================================

from sklearn.feature_selection import chi2

chi_scores, p_values = chi2(
    X_train_tfidf,
    y_train
)

feature_names = (
    vectorizer
    .get_feature_names_out()
)

chi_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Chi2_Score": chi_scores
    }
)

chi_importance = chi_importance.sort_values(
    by="Chi2_Score",
    ascending=False
)

print("\nTop 20 Important Research Terms")
print("--------------------------------")

print(
    chi_importance.head(20)
)

# ============================================================
# 21. VISUALIZE TOP IMPORTANT FEATURES
# ============================================================

top_features = (
    chi_importance
    .head(20)
    .sort_values(
        by="Chi2_Score",
        ascending=True
    )
)

plt.figure(figsize=(10,8))

plt.barh(
    top_features["Feature"],
    top_features["Chi2_Score"]
)

plt.title(
    "Top 20 Important Research Keywords"
)

plt.xlabel(
    "Chi-Square Score"
)

plt.ylabel(
    "Research Terms"
)

plt.tight_layout()

plt.show()

# ============================================================
# 22. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)

print(
    "\nLogistic Regression Model Trained Successfully."
)

# ============================================================
# 23. VALIDATION PREDICTION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

val_predictions = model.predict(
    X_val_tfidf
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

print("\nValidation Performance")
print("----------------------------")

print(
    f"Accuracy : {val_accuracy:.4f}"
)

print(
    f"Precision: {val_precision:.4f}"
)

print(
    f"Recall   : {val_recall:.4f}"
)

print(
    f"F1 Score : {val_f1:.4f}"
)

# ============================================================
# 24. TEST SET EVALUATION
# ============================================================

test_predictions = model.predict(
    X_test_tfidf
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

print("\nResearch Intelligence Test Performance")
print("----------------------------------------")

print(
    f"Accuracy : {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall   : {test_recall:.4f}"
)

print(
    f"F1 Score : {test_f1:.4f}"
)

# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("----------------------------")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

labels = model.classes_

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title(
    "Research Category Classification Confusion Matrix"
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.show()

# ============================================================
# 27. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame(
    {
        "Research_Text": X_test.values,
        "Actual_Category": y_test.values,
        "Predicted_Category": test_predictions
    }
)

prediction_results.to_csv(
    "research_predictions.csv",
    index=False
)

# ============================================================
# 28. SAVE FEATURE IMPORTANCE
# ============================================================

chi_importance.to_csv(
    "research_feature_importance.csv",
    index=False
)

# ============================================================
# 29. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "research_cleaned_dataset.csv",
    index=False
)

# ============================================================
# 30. COMPLETION MESSAGE
# ============================================================

print("\nFiles saved successfully:")
print("1. research_predictions.csv")
print("2. research_feature_importance.csv")
print("3. research_cleaned_dataset.csv")

print("\n==============================================")
print("RESEARCH INTELLIGENCE ANALYSIS COMPLETED")
print("EDA Completed")
print("Text Feature Engineering Completed")
print("TF-IDF Feature Extraction Completed")
print("Model Training Completed")
print("Validation and Testing Completed")
print("Results Exported Successfully")
print("==============================================")
