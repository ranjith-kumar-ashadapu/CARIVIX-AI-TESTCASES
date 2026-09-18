import pandas as pd
from data_validation import DataValidator, Schema, ColumnRule


def make_schema():
    return Schema(columns=[
        ColumnRule(name="district", dtype=str, required=True),
        ColumnRule(name="unemployment_rate", dtype=float, required=True, min_value=0, max_value=100),
    ])


def test_valid_rows_pass():
    df = pd.DataFrame({
        "district": ["Hyderabad", "Warangal"],
        "unemployment_rate": [5.2, 7.8],
    })
    report = DataValidator(make_schema()).validate(df)
    assert report.valid_rows == 2


def test_null_required_field_is_flagged():
    df = pd.DataFrame({
        "district": ["Hyderabad", None],
        "unemployment_rate": [5.2, 6.0],
    })
    report = DataValidator(make_schema()).validate(df)
    assert report.missing_value_counts.get("district", 0) == 1


def test_out_of_range_value_is_flagged():
    df = pd.DataFrame({
        "district": ["Hyderabad"],
        "unemployment_rate": [150.0],
    })
    report = DataValidator(make_schema()).validate(df)
    assert len(report.row_errors) >= 1


def test_duplicate_rows_are_counted():
    df = pd.DataFrame({
        "district": ["Hyderabad", "Hyderabad"],
        "unemployment_rate": [5.2, 5.2],
    })
    report = DataValidator(make_schema()).validate(df)
    assert report.duplicate_row_count == 1


def test_clean_removes_duplicates_and_invalid_rows():
    df = pd.DataFrame({
        "district": ["Hyderabad", "Hyderabad", None],
        "unemployment_rate": [5.2, 5.2, 6.0],
    })
    cleaned = DataValidator(make_schema()).clean(df)
    assert len(cleaned) == 1