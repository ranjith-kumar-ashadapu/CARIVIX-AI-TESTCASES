# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

project_root = Path(__file__).resolve().parents[2]
dataset_path = project_root / "data" / "Public Program Evaluation_final.csv"

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(dataset_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print("Shape of Dataset:", df.shape)

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

output_folder = project_root / "experiments" / "Government_Intelligence_EDA"
os.makedirs(output_folder, exist_ok=True)

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()

print("\nData Types:")
print(df.dtypes)

# ==========================================
# MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
print(missing[missing > 0])

print("\nMissing Value Percentage:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Dataset Shape After Removing Duplicates:", df.shape)

# ==========================================
# CONVERT NUMERIC COLUMNS
# ==========================================

numeric_columns_to_convert = [
    "Total No. of Workers",
    "Total Exp(Rs. in Lakhs.)",
    "Average Wage rate per day per person(Rs.)"
]

for column in numeric_columns_to_convert:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[column] = pd.to_numeric(df[column], errors="coerce")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())

# ==========================================
# UNIQUE VALUES
# ==========================================

print("\nNumber of States:", df["state_name"].nunique())
print("Number of Districts:", df["district_name"].nunique())

print("\nColumn Names:")
print(df.columns.tolist())

# ==========================================
# 1. STATE-WISE RECORD COUNT
# ==========================================

plt.figure(figsize=(14, 6))

df["state_name"].value_counts().plot(
    kind="bar",
    color="steelblue"
)

plt.title("Number of District Records by State")
plt.xlabel("State")
plt.ylabel("Record Count")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "01_State_Wise_Record_Count.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 2. TOP 10 DISTRICTS BY TOTAL WORKERS
# ==========================================

top_workers = (
    df.dropna(subset=["Total No. of Workers"])
    .sort_values(
        by="Total No. of Workers",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.bar(
    top_workers["district_name"],
    top_workers["Total No. of Workers"],
    color="seagreen"
)

plt.title("Top 10 Districts by Total Workers")
plt.xlabel("District")
plt.ylabel("Total Workers")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "02_Top10_Districts_Workers.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 3. HISTOGRAMS OF NUMERIC COLUMNS
# ==========================================

df[numeric_cols].hist(
    figsize=(20, 18),
    bins=30,
    edgecolor="black"
)

plt.suptitle(
    "Distribution of Numeric Features",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "03_Numeric_Features_Distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 4. WAGE RATE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Average Wage rate per day per person(Rs.)",
    bins=30,
    kde=True,
    color="orange"
)

plt.title("Distribution of Average Wage Rate")
plt.xlabel("Average Wage Rate per Day (Rs.)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "04_Wage_Rate_Distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 5. BOXPLOT - TOTAL EXPENDITURE
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["Total Exp(Rs. in Lakhs.)"],
    color="tomato"
)

plt.title("Outlier Detection - Total Expenditure")
plt.xlabel("Total Expenditure (Rs. in Lakhs.)")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "05_Total_Expenditure_Boxplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 6. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(18, 14))

corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "06_Correlation_Heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 7. TOP 10 DISTRICTS BY EXPENDITURE
# ==========================================

top_exp = (
    df.dropna(subset=["Total Exp(Rs. in Lakhs.)"])
    .sort_values(
        by="Total Exp(Rs. in Lakhs.)",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.bar(
    top_exp["district_name"],
    top_exp["Total Exp(Rs. in Lakhs.)"],
    color="purple"
)

plt.title("Top 10 Districts by Total Expenditure")
plt.xlabel("District")
plt.ylabel("Expenditure (Rs. in Lakhs.)")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "07_Top10_Districts_Expenditure.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 8. WORKERS VS TOTAL EXPENDITURE
# ==========================================

scatter_data = df[
    [
        "Total No. of Workers",
        "Total Exp(Rs. in Lakhs.)"
    ]
].dropna()

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=scatter_data,
    x="Total No. of Workers",
    y="Total Exp(Rs. in Lakhs.)",
    color="darkblue"
)

plt.title("Workers vs Total Expenditure")
plt.xlabel("Total Workers")
plt.ylabel("Total Expenditure (Rs. in Lakhs.)")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "08_Workers_vs_Expenditure.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 9. TOP 10 STATES BY TOTAL EXPENDITURE
# ==========================================

state_exp = (
    df.groupby("state_name")["Total Exp(Rs. in Lakhs.)"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

state_exp.plot(
    kind="bar",
    color="darkorange"
)

plt.title("Top 10 States by Total Expenditure")
plt.xlabel("State")
plt.ylabel("Total Expenditure (Rs. in Lakhs.)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "09_Top10_States_Expenditure.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 10. TOP 10 STATES BY TOTAL WORKERS
# ==========================================

state_workers = (
    df.groupby("state_name")["Total No. of Workers"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

state_workers.plot(
    kind="bar",
    color="teal"
)

plt.title("Top 10 States by Total Workers")
plt.xlabel("State")
plt.ylabel("Total Workers")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "10_Top10_States_Workers.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 11. PAIRPLOT
# ==========================================

selected_cols = [
    "Total No. of Workers",
    "Total Exp(Rs. in Lakhs.)",
    "Average Wage rate per day per person(Rs.)"
]

pairplot_data = df[selected_cols].dropna()

pair_plot = sns.pairplot(
    pairplot_data,
    diag_kind="hist"
)

pair_plot.fig.suptitle(
    "Relationship Between Workers, Expenditure and Wage Rate",
    y=1.02
)

pair_plot.savefig(
    os.path.join(output_folder, "11_Pairplot_Numeric_Features.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close("all")

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Final Dataset Shape:", df.shape)
print("Total States:", df["state_name"].nunique())
print("Total Districts:", df["district_name"].nunique())
print("Total Numeric Features:", len(numeric_cols))

print("\nAll visualizations have been saved in:")
print(os.path.abspath(output_folder))

print("=" * 60)
