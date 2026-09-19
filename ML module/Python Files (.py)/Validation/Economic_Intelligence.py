# ==========================================
# ECONOMIC TREND ANALYSIS - COMPLETE EDA
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ==========================================
# FILE PATHS
# ==========================================

input_file = r"D:\OneDrive\Desktop\Day 3 Task\Economic_Trend_Analysis_Cleaned.csv"

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task\Economic_Intelligence_EDA"
)

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

# Convert all column names to strings
df.columns = df.columns.astype(str).str.strip()

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print("Shape:", df.shape)

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()

print("\nDATA TYPES")
print(df.dtypes)

# ==========================================
# MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
print(missing[missing > 0])

print("\nMISSING VALUE PERCENTAGE")

missing_percent = (df.isnull().sum() / len(df)) * 100
print(
    missing_percent[missing_percent > 0]
    .sort_values(ascending=False)
)

# ==========================================
# DUPLICATE VALUES
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Shape After Removing Duplicates:", df.shape)

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all"))

# ==========================================
# UNIQUE VALUES
# ==========================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

if "COUNTRY" in df.columns:
    print("\nNumber of Countries:", df["COUNTRY"].nunique())
    print(df["COUNTRY"].unique())

if "INDICATOR" in df.columns:
    print("\nNumber of Indicators:", df["INDICATOR"].nunique())
    print(df["INDICATOR"].unique())

if "UNIT" in df.columns:
    print("\nNumber of Units:", df["UNIT"].nunique())
    print(df["UNIT"].unique())

print("\nColumns:")
print(df.columns.tolist())

# ==========================================
# IDENTIFY YEAR COLUMNS
# ==========================================

year_cols = [
    column
    for column in df.columns
    if column.isdigit()
]

print("\nYear Columns:")
print(year_cols)

# Convert year columns to numeric
for column in year_cols:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ==========================================
# 1. COUNTRY DISTRIBUTION
# ==========================================

if "COUNTRY" in df.columns:

    plt.figure(figsize=(12, 6))

    df["COUNTRY"].value_counts().head(20).plot(
        kind="bar",
        color="steelblue"
    )

    plt.title("Top 20 Countries by Number of Indicators")
    plt.xlabel("Country")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "01_Top20_Countries.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# 2. INDICATOR DISTRIBUTION
# ==========================================

if "INDICATOR" in df.columns:

    plt.figure(figsize=(12, 6))

    df["INDICATOR"].value_counts().head(20).plot(
        kind="bar",
        color="darkorange"
    )

    plt.title("Top 20 Economic Indicators")
    plt.xlabel("Indicator")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "02_Top20_Economic_Indicators.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# 3. SCALE DISTRIBUTION
# ==========================================

if "SCALE" in df.columns:

    plt.figure(figsize=(8, 5))

    df["SCALE"].value_counts().plot(
        kind="bar",
        color="seagreen"
    )

    plt.title("Distribution of Measurement Scales")
    plt.xlabel("Scale")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "03_Scale_Distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# 4. UNIT DISTRIBUTION
# ==========================================

if "UNIT" in df.columns:

    plt.figure(figsize=(8, 5))

    df["UNIT"].value_counts().plot(
        kind="bar",
        color="purple"
    )

    plt.title("Distribution of Units")
    plt.xlabel("Unit")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "04_Unit_Distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# 5. HISTOGRAMS OF YEAR COLUMNS
# ==========================================

if len(year_cols) > 0:

    df[year_cols].hist(
        figsize=(20, 18),
        bins=30,
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Yearly Economic Indicator Values",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "05_Yearly_Values_Distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

else:

    print("\nNo year columns were found.")

# ==========================================
# 6. 2025 HISTOGRAM
# ==========================================

if "2025" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["2025"].dropna(),
        bins=40,
        kde=True,
        color="orange"
    )

    plt.title("Distribution of Indicator Values in 2025")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "06_2025_Histogram.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

else:

    print("\nThe 2025 column was not found.")

# ==========================================
# 7. 2025 BOXPLOT
# ==========================================

if "2025" in df.columns:

    plt.figure(figsize=(10, 4))

    sns.boxplot(
        x=df["2025"].dropna(),
        color="tomato"
    )

    plt.title("Outlier Detection for 2025")
    plt.xlabel("Indicator Value")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "07_2025_Boxplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# 8. YEARLY CORRELATION HEATMAP
# ==========================================

if len(year_cols) > 1:

    correlation_matrix = df[year_cols].corr()

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Between Yearly Variables")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "08_Yearly_Correlation_Heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

else:

    print("\nNot enough year columns for correlation analysis.")

# ==========================================
# 9. INDIA TREND ANALYSIS
# ==========================================

if "COUNTRY" in df.columns and "INDICATOR" in df.columns:

    country = "India"

    india_data = df[
        df["COUNTRY"].astype(str).str.strip() == country
    ]

if india_data.empty:

    print("\nNo data available for", country)

else:

    indicator = india_data["INDICATOR"].iloc[0]

    india_indicator_data = india_data[
            india_data["INDICATOR"] == indicator
        ]

if (
            not india_indicator_data.empty
            and len(year_cols) > 0
        ):

    values = india_indicator_data[year_cols].iloc[0]

    values = pd.to_numeric(
                values,
                errors="coerce"
            )

    trend_data = pd.DataFrame({
                "Year": [
                    int(year)
                    for year in year_cols
                ],
                "Value": values.values
            })

    trend_data = trend_data.dropna()

    plt.figure(figsize=(14, 5))

    plt.plot(
                trend_data["Year"],
                trend_data["Value"],
                marker="o",
                linewidth=2,
                color="darkblue"
            )

    plt.title(
                f"{country} - {indicator}"
            )
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
                os.path.join(
                    output_folder,
                    "09_India_Economic_Trend.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

    plt.show()
    plt.close()

else:

    print(
                "\nIndia trend data is not available."
            )

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Final Dataset Shape:", df.shape)

if "COUNTRY" in df.columns:
    print("Countries:", df["COUNTRY"].nunique())

if "INDICATOR" in df.columns:
    print("Indicators:", df["INDICATOR"].nunique())

if "UNIT" in df.columns:
    print("Units:", df["UNIT"].nunique())

print("Year Columns:", len(year_cols))

print("\nAll visualizations have been saved in:")
print(os.path.abspath(output_folder))

print("=" * 60)

