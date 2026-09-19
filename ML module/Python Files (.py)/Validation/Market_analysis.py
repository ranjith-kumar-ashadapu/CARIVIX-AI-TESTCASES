# ================================
# IMPORT LIBRARIES
# ================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# LOAD DATASET
# ================================

df = pd.read_csv("Market_Analysis_1981_2025_Final_Cleaned.csv")

# ================================
# CREATE OUTPUT FOLDER
# ================================

output_folder = r"D:\OneDrive\Desktop\Day 3 Task\Market_Analysis_EDA"
os.makedirs(output_folder, exist_ok=True)

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

print("\nDataset Shape:", df.shape)

# ================================
# DATASET INFORMATION
# ================================

print("\n" + "=" * 50)
print("Dataset Information")
print("=" * 50)
df.info()

print("\nData Types")
print(df.dtypes)

# ================================
# MISSING VALUES
# ================================

print("\nMissing Values")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nMissing Value Percentage")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# ================================
# DUPLICATES
# ================================

print("\nDuplicate Rows:", df.duplicated().sum())

# ================================
# DESCRIPTIVE STATISTICS
# ================================

print("\nDescriptive Statistics")
print(df.describe())

# ================================
# UNIQUE VALUES
# ================================

print("\nNumber of Countries:", df["Country Name"].nunique())

print("\nCountries:")
print(df["Country Name"].unique())

print("\nNumber of Economic Indicators:", df["Series Name"].nunique())

# ================================
# COUNTRY DISTRIBUTION
# ================================

plt.figure(figsize=(10, 6))

df["Country Name"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Countries by Number of Records")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "01_Top10_Countries.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ================================
# ECONOMIC INDICATOR DISTRIBUTION
# ================================

plt.figure(figsize=(15, 6))

df["Series Name"].value_counts().plot(kind="bar")

plt.title("Distribution of Economic Indicators")
plt.xlabel("Economic Indicator")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "02_Economic_Indicators.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ================================
# HISTOGRAM OF 2024 VALUES
# ================================

year_column = "2024 [YR2024]"

if year_column in df.columns:

    df[year_column] = pd.to_numeric(df[year_column], errors="coerce")

    plt.figure(figsize=(8, 5))

    sns.histplot(df[year_column], bins=40, kde=True)

    plt.title("Distribution of 2024 Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "03_2024_Histogram.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

# ================================
# BOXPLOT
# ================================

if year_column in df.columns:

    plt.figure(figsize=(10, 4))

    sns.boxplot(x=df[year_column])

    plt.title("Outlier Detection (2024)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "04_Boxplot_2024.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

# ================================
# CORRELATION HEATMAP
# ================================

year_cols = [col for col in df.columns if "YR" in col]

for col in year_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

plt.figure(figsize=(16, 12))

corr = df[year_cols].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Between Year Columns")

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "05_Correlation_Heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ================================
# INDIA GDP TREND
# ================================

india_gdp = df[
    (df["Country Name"] == "India") &
    (df["Series Name"] == "GDP (current US$)")
]

if india_gdp.empty:

    print("\nIndia GDP data not found.")

else:

    years = list(range(1981, 2026))

    values = india_gdp.iloc[:, 4:].values.flatten()
    values = pd.to_numeric(values, errors="coerce")

    plt.figure(figsize=(14, 5))

    plt.plot(years, values, marker='o', linewidth=2)

    plt.title("India GDP Trend (1981–2025)")
    plt.xlabel("Year")
    plt.ylabel("GDP (Current US$)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "06_India_GDP_Trend.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

# ================================
# END OF EDA
# ================================

print("\n" + "=" * 50)
print("EDA Completed Successfully.")
print("All visualizations have been saved in:")
print(os.path.abspath(output_folder))
print("=" * 50)