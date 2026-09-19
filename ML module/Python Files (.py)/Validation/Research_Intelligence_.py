# ==========================================
# ARXIV RESEARCH INTELLIGENCE
# FEATURE ENGINEERING
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import ast
import re

import numpy as np
import pandas as pd

# ==========================================
# FILE PATHS
# ==========================================

input_file = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\arxiv_data.csv"
)

output_folder = (
    r"D:\OneDrive\Desktop\Day 3 Task"
    r"\Research_Intelligence_Feature_Engineering"
)

os.makedirs(
    output_folder,
    exist_ok=True
)

output_file = os.path.join(
    output_folder,
    "Research_Intelligence_Feature_Engineered.csv"
)

dictionary_file = os.path.join(
    output_folder,
    "Research_Intelligence_Feature_Dictionary.csv"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(input_file)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

original_columns = df.columns.tolist()

print("=" * 60)
print("ARXIV RESEARCH INTELLIGENCE FEATURE ENGINEERING")
print("=" * 60)

print(
    "Original Dataset Shape:",
    df.shape
)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print(
    "\nDuplicate Rows:",
    df.duplicated().sum()
)

df = (
    df
    .drop_duplicates()
    .copy()
)

print(
    "Shape After Removing Duplicates:",
    df.shape
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_list_value(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if value == "":
        return []

    try:

        parsed_value = ast.literal_eval(value)

        if isinstance(parsed_value, list):

            return [

                str(item).strip()

                for item in parsed_value

                if str(item).strip()

            ]

    except:

        pass

    if "," in value:

        return [

            item.strip()

            for item in value.split(",")

            if item.strip()

        ]

    return [value]

def count_words(value):

    if pd.isna(value):

        return 0

    value = str(value).strip()

    if value == "":

        return 0

    return len(
        re.findall(
            r"\b\w+\b",
            value
        )
    )

def count_sentences(value):

    if pd.isna(value):

        return 0

    value = str(value).strip()

    if value == "":

        return 0

    return len(
        re.findall(
            r"[.!?]+",
            value
        )
    )

# ==========================================
# FIND TITLE COLUMN
# ==========================================

title_column = None

for column in df.columns:

    if column.lower() in [

        "title",
        "titles"

    ]:

        title_column = column
        break

if title_column:

    df["Title_Length_Characters"] = (

        df[title_column]
        .fillna("")
        .astype(str)
        .str.len()

    )

    df["Title_Word_Count"] = (

        df[title_column]
        .apply(count_words)

    )

    df["Title_Contains_Question"] = (

        df[title_column]
        .fillna("")
        .astype(str)
        .str.contains(
            r"\?",
            regex=True
        )
        .astype(int)

    )

    df["Title_Contains_Hyphen"] = (

        df[title_column]
        .fillna("")
        .astype(str)
        .str.contains(
            "-",
            regex=False
        )
        .astype(int)

    )

else:

    print(
        "\nTitle column not found"
    )

# ==========================================
# SUMMARY FEATURES
# ==========================================

summary_column = None

for column in df.columns:

    if column.lower() in [

        "summary",
        "summaries",
        "abstract",
        "abstracts"

    ]:

        summary_column = column
        break

if summary_column:

    df["Summary_Length_Characters"] = (

        df[summary_column]
        .fillna("")
        .astype(str)
        .str.len()

    )

    df["Summary_Word_Count"] = (

        df[summary_column]
        .apply(count_words)

    )

    df["Summary_Sentence_Count"] = (

        df[summary_column]
        .apply(count_sentences)

    )

    df["Summary_Average_Word_Length"] = np.where(

        df["Summary_Word_Count"] > 0,

        df["Summary_Length_Characters"]

        /

        df["Summary_Word_Count"],

        0

    )

else:

    print(
        "\nSummary column not found"
    )

# ==========================================
# SUMMARY TEXT INTELLIGENCE FEATURES
# ==========================================

if summary_column:

    summary_text = (

        df[summary_column]
        .fillna("")
        .astype(str)
        .str.lower()

    )

    df["Summary_Contains_Problem"] = (

        summary_text
        .str.contains(
            "problem|challenge|issue|limitation",
            regex=True
        )

        .astype(int)

    )

    df["Summary_Contains_Method"] = (

        summary_text
        .str.contains(
            "method|approach|algorithm|model|technique",
            regex=True
        )

        .astype(int)

    )

    df["Summary_Contains_Result"] = (

        summary_text
        .str.contains(
            "result|performance|accuracy|evaluation|experiment",
            regex=True
        )

        .astype(int)

    )

# ==========================================
# TITLE SUMMARY RATIO
# ==========================================

if (

    "Title_Word_Count" in df.columns

    and

    "Summary_Word_Count" in df.columns

):

    df["Title_to_Summary_Word_Ratio"] = np.where(

        df["Summary_Word_Count"] > 0,

        (

            df["Title_Word_Count"]

            /

            df["Summary_Word_Count"]

        ),

        0

    )

# ==========================================
# RESEARCH CATEGORY FEATURES
# ==========================================

terms_column = None

for column in df.columns:

    if column.lower() in [

        "terms",
        "categories",
        "category",
        "subject"

    ]:

        terms_column = column
        break

if terms_column:

    parsed_terms = (

        df[terms_column]
        .apply(parse_list_value)

    )

    df["Category_Count"] = (

        parsed_terms
        .apply(len)

    )

    df["Primary_Category"] = (

        parsed_terms
        .apply(

            lambda x:

            x[0]

            if len(x) > 0

            else "Unknown"

        )

    )

    df["Category_Text_Length"] = (

        parsed_terms
        .apply(

            lambda x:

            len(" ".join(x))

        )

    )

    df["Has_Multiple_Categories"] = (

        df["Category_Count"] > 1

    ).astype(int)

    df["Primary_Category_ID"] = (

        df["Primary_Category"]
        .astype("category")
        .cat.codes

    )

else:

    print(
        "\nCategory column not found"
    )

# ==========================================
# AUTHOR FEATURES
# ==========================================

authors_column = None

for column in df.columns:

    if column.lower() in [

        "authors",
        "author",
        "author_names"

    ]:

        authors_column = column
        break

if authors_column:

    parsed_authors = (

        df[authors_column]
        .apply(parse_list_value)

    )

    df["Author_Count"] = (

        parsed_authors
        .apply(len)

    )

    df["Has_Multiple_Authors"] = (

        df["Author_Count"] > 1

    ).astype(int)

    df["First_Author"] = (

        parsed_authors
        .apply(

            lambda x:

            x[0]

            if len(x) > 0

            else "Unknown"

        )

    )

    df["First_Author_ID"] = (

        df["First_Author"]
        .astype("category")
        .cat.codes

    )

else:

    print(
        "\nAuthor column not found"
    )

# ==========================================
# PUBLICATION DATE FEATURES
# ==========================================

date_column = None

possible_date_columns = [

    "published",
    "publication_date",
    "published_date",
    "created",
    "updated",
    "date",
    "year"

]

for column in df.columns:

    if column.lower() in possible_date_columns:

        date_column = column
        break

if date_column:

    parsed_dates = pd.to_datetime(

        df[date_column],

        errors="coerce"

    )

    df["Publication_Year"] = (

        parsed_dates
        .dt.year

    )

    df["Publication_Month"] = (

        parsed_dates
        .dt.month

    )

    df["Publication_Quarter"] = (

        parsed_dates
        .dt.quarter

    )

    df["Publication_Day_of_Week"] = (

        parsed_dates
        .dt.dayofweek

    )

    df["Publication_Weekday_Name"] = (

        parsed_dates
        .dt.day_name()

    )

    df["Publication_Date_Missing_Flag"] = (

        parsed_dates
        .isnull()
        .astype(int)

    )

else:

    print(
        "\nPublication date column not found"
    )

# ==========================================
# IDENTIFIER FEATURES
# ==========================================

identifier_column = None

identifier_candidates = [

    "id",
    "paper_id",
    "arxiv_id",
    "entry_id"

]

for column in df.columns:

    if column.lower() in identifier_candidates:

        identifier_column = column
        break

if identifier_column:

    df["Identifier_Length"] = (

        df[identifier_column]
        .fillna("")
        .astype(str)
        .str.len()

    )

    df["Identifier_Missing_Flag"] = (

        df[identifier_column]
        .isnull()
        .astype(int)

    )

# ==========================================
# DATA QUALITY FEATURES
# ==========================================

df["Total_Missing_Values_Per_Row"] = (

    df.isnull()
    .sum(axis=1)

)

df["Total_Available_Values_Per_Row"] = (

    df.notnull()
    .sum(axis=1)

)

df["Missing_Value_Percentage_Per_Row"] = (

    df["Total_Missing_Values_Per_Row"]

    /

    len(df.columns)

    *

    100

)

# ==========================================
# REPLACE INFINITE VALUES
# ==========================================

df = df.replace(

    [np.inf, -np.inf],

    np.nan

)

# ==========================================
# IDENTIFY NEW FEATURES
# ==========================================

new_features = [

    column

    for column in df.columns

    if column not in original_columns

]

print("\n" + "=" * 60)

print("NEW FEATURES CREATED")

print("=" * 60)

for feature in new_features:

    print("-", feature)

# ==========================================
# FEATURE DESCRIPTIONS
# ==========================================

feature_descriptions = {

    "Title_Length_Characters":
        "Number of characters in the research paper title",

    "Title_Word_Count":
        "Number of words in the research paper title",

    "Title_Contains_Question":
        "Indicates whether the title contains a question mark",

    "Title_Contains_Hyphen":
        "Indicates whether the title contains a hyphen",

    "Summary_Length_Characters":
        "Number of characters in the research summary",

    "Summary_Word_Count":
        "Number of words in the research summary",

    "Summary_Sentence_Count":
        "Approximate number of sentences in the summary",

    "Summary_Average_Word_Length":
        "Average characters per word in the summary",

    "Summary_Contains_Problem":
        "Indicates whether summary discusses problems or challenges",

    "Summary_Contains_Method":
        "Indicates whether summary discusses methods or approaches",

    "Summary_Contains_Result":
        "Indicates whether summary discusses results or performance",

    "Title_to_Summary_Word_Ratio":
        "Ratio between title word count and summary word count",

    "Category_Count":
        "Number of research categories assigned to paper",

    "Primary_Category":
        "Primary research category",

    "Category_Text_Length":
        "Character length of category information",

    "Has_Multiple_Categories":
        "Indicates whether paper belongs to multiple categories",

    "Primary_Category_ID":
        "Encoded numerical value of primary category",

    "Author_Count":
        "Number of authors in research paper",

    "Has_Multiple_Authors":
        "Indicates whether paper has multiple authors",

    "First_Author":
        "First listed author name",

    "First_Author_ID":
        "Encoded numerical value of first author",

    "Publication_Year":
        "Year extracted from publication date",

    "Publication_Month":
        "Month extracted from publication date",

    "Publication_Quarter":
        "Quarter extracted from publication date",

    "Publication_Day_of_Week":
        "Numeric weekday value of publication",

    "Publication_Weekday_Name":
        "Publication weekday name",

    "Publication_Date_Missing_Flag":
        "Indicates missing publication date",

    "Identifier_Length":
        "Length of research paper identifier",

    "Identifier_Missing_Flag":
        "Indicates missing identifier",

    "Total_Missing_Values_Per_Row":
        "Total missing values present in each row",

    "Total_Available_Values_Per_Row":
        "Total available values present in each row",

    "Missing_Value_Percentage_Per_Row":
        "Percentage of missing values per row"

}

# ==========================================
# CREATE FEATURE DICTIONARY
# ==========================================

dictionary_rows = []

for column in df.columns:

    if column in feature_descriptions:

        description = feature_descriptions[column]

    else:

        description = (
            "Original dataset column"
        )

    dictionary_rows.append(

        {

            "Feature_Name":
                column,

            "Feature_Type":
                str(df[column].dtype),

            "Description":
                description,

            "Missing_Values":
                int(
                    df[column]
                    .isnull()
                    .sum()
                ),

            "Unique_Values":
                int(
                    df[column]
                    .nunique()
                )

        }

    )

feature_dictionary = pd.DataFrame(
    dictionary_rows
)

# ==========================================
# SAVE OUTPUT FILES
# ==========================================

df.to_csv(

    output_file,

    index=False

)

feature_dictionary.to_csv(

    dictionary_file,

    index=False

)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)

print(
    "FEATURE ENGINEERING COMPLETED SUCCESSFULLY"
)

print("=" * 60)

print(

    "Original Number of Columns:",

    len(original_columns)

)

print(

    "Final Number of Columns:",

    len(df.columns)

)

print(

    "New Features Created:",

    len(new_features)

)

print(

    "Final Dataset Shape:",

    df.shape

)

print("\nFeature Engineered Dataset Saved At:")

print(output_file)

print("\nFeature Dictionary Saved At:")

print(dictionary_file)

print("=" * 60)
