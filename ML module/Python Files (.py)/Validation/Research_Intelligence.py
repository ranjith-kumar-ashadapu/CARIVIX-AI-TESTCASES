# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("arxiv_data.csv")

print("="*60)
print("ARXIV RESEARCH PAPER DATASET")
print("="*60)

print("Dataset Shape:", df.shape)

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDATASET INFORMATION")
df.info()

print("\nFIRST 5 RECORDS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

# ==========================================
# MISSING VALUES
# ==========================================

print("\nMISSING VALUES")

missing = df.isnull().sum()
print(missing[missing > 0])

print("\nMISSING VALUE PERCENTAGE")

missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# ==========================================
# DUPLICATE RECORDS
# ==========================================

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\nDESCRIPTIVE STATISTICS")

print(df.describe(include="all"))

# ==========================================
# COLUMN NAMES
# ==========================================

print("\nCOLUMN NAMES")

print(df.columns)

# ==========================================
# TOP RESEARCH CATEGORIES
# ==========================================

all_terms = []

for item in df["terms"].dropna():
    try:
        categories = ast.literal_eval(item)
        all_terms.extend(categories)
    except:
        pass

term_counts = Counter(all_terms)

top_terms = pd.DataFrame(
    term_counts.items(),
    columns=["Category", "Count"]
).sort_values(by="Count", ascending=False)

plt.figure(figsize=(12,6))

plt.bar(
    top_terms["Category"][:10],
    top_terms["Count"][:10]
)

plt.xticks(rotation=45)

plt.title("Top 10 Research Categories")
plt.xlabel("Research Category")
plt.ylabel("Number of Papers")

plt.tight_layout()
plt.show()

# ==========================================
# SUMMARY LENGTH
# ==========================================

df["Summary_Length"] = df["summaries"].fillna("").str.split().str.len()

plt.figure(figsize=(8,5))

sns.histplot(
    df["Summary_Length"],
    bins=30,
    kde=True
)

plt.title("Distribution of Summary Length")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==========================================
# TOP AUTHORS
# ==========================================

if "authors" in df.columns:

    top_authors = df["authors"].value_counts().head(10)

    plt.figure(figsize=(12,6))

    top_authors.plot(kind="bar")

    plt.title("Top 10 Authors by Number of Papers")

    plt.xlabel("Author")
    plt.ylabel("Number of Papers")

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

# ==========================================
# SUMMARY LENGTH BOXPLOT
# ==========================================

plt.figure(figsize=(8,4))

sns.boxplot(x=df["Summary_Length"])

plt.title("Summary Length Outliers")

plt.tight_layout()
plt.show()

# ==========================================
# SUMMARY STATISTICS
# ==========================================

print("\nSUMMARY LENGTH STATISTICS")

print(df["Summary_Length"].describe())

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "="*60)
print("EDA COMPLETED SUCCESSFULLY")
print("="*60)

print("Final Dataset Shape:", df.shape)
print("Total Research Papers:", len(df))
print("Unique Categories:", len(top_terms))
print("Average Summary Length:", round(df["Summary_Length"].mean(), 2), "words")