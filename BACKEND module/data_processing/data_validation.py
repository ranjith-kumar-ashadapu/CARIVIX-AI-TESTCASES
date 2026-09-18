from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from error_handling import DataValidationError, ErrorCollector, SchemaError
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnRule:
    name: str
    dtype: Optional[type] = None
    required: bool = True
    allow_null: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    custom_check: Optional[Callable[[Any], bool]] = None


@dataclass
class Schema:
    columns: List[ColumnRule] = field(default_factory=list)

    def get(self, name: str) -> Optional[ColumnRule]:
        return next((c for c in self.columns if c.name == name), None)


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_rows: int = 0
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_row_count: int = 0
    schema_errors: List[str] = field(default_factory=list)
    row_errors: List[dict] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.schema_errors and not self.row_errors and self.duplicate_row_count == 0

    def summary(self) -> str:
        lines = [
            "=== CARIVIX Data Validation Report ===",
            f"Total rows:           {self.total_rows}",
            f"Valid rows:           {self.valid_rows}",
            f"Duplicate rows:       {self.duplicate_row_count}",
            f"Schema errors:        {len(self.schema_errors)}",
            f"Row-level errors:     {len(self.row_errors)}",
        ]
        if self.missing_value_counts:
            lines.append("Missing values by column:")
            for col, count in self.missing_value_counts.items():
                if count > 0:
                    lines.append(f"  - {col}: {count}")
        if self.schema_errors:
            lines.append("Schema issues:")
            lines.extend(f"  - {e}" for e in self.schema_errors)
        return "\n".join(lines)


class DataValidator:
    def __init__(self, schema: Schema, strict: bool = False):
        self.schema = schema
        self.strict = strict

    def _check_schema_structure(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for rule in self.schema.columns:
            if rule.required and rule.name not in df.columns:
                msg = f"Missing required column: '{rule.name}'"
                report.schema_errors.append(msg)
                if self.strict:
                    raise SchemaError(msg)

    def _check_missing_values(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for rule in self.schema.columns:
            if rule.name not in df.columns:
                continue
            null_count = int(df[rule.name].isna().sum())
            report.missing_value_counts[rule.name] = null_count
            if null_count > 0 and not rule.allow_null:
                logger.warning(
                    f"Column '{rule.name}' has {null_count} missing value(s) "
                    f"but nulls are not allowed.",
                    extra={"context": "validation.missing"},
                )

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        report.duplicate_row_count = int(df.duplicated().sum())
        if report.duplicate_row_count > 0:
            logger.warning(
                f"Found {report.duplicate_row_count} duplicate row(s).",
                extra={"context": "validation.duplicates"},
            )

    def _validate_row(self, row: pd.Series) -> None:
        for rule in self.schema.columns:
            if rule.name not in row.index:
                continue
            value = row[rule.name]

            if pd.isna(value):
                if not rule.allow_null:
                    raise DataValidationError(f"Column '{rule.name}' is null")
                continue

            if rule.dtype is not None and not isinstance(value, rule.dtype):
                try:
                    rule.dtype(value)
                except (TypeError, ValueError):
                    raise DataValidationError(
                        f"Column '{rule.name}' expected type {rule.dtype.__name__}, "
                        f"got {type(value).__name__} ({value!r})"
                    )

            if rule.min_value is not None and value < rule.min_value:
                raise DataValidationError(
                    f"Column '{rule.name}' value {value} below minimum {rule.min_value}"
                )
            if rule.max_value is not None and value > rule.max_value:
                raise DataValidationError(
                    f"Column '{rule.name}' value {value} above maximum {rule.max_value}"
                )
            if rule.allowed_values is not None and value not in rule.allowed_values:
                raise DataValidationError(
                    f"Column '{rule.name}' value {value!r} not in allowed set {rule.allowed_values}"
                )
            if rule.custom_check is not None and not rule.custom_check(value):
                raise DataValidationError(f"Column '{rule.name}' failed custom check for value {value!r}")

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport(total_rows=len(df))
        logger.info(f"Starting validation on {len(df)} row(s)", extra={"context": "validation"})

        self._check_schema_structure(df, report)
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)

        collector = ErrorCollector()
        for idx, row in df.iterrows():
            try:
                self._validate_row(row)
            except DataValidationError as e:
                collector.add(idx, e)

        report.row_errors = collector.as_dicts()
        report.valid_rows = report.total_rows - len(report.row_errors)

        if collector.has_errors():
            collector.log_summary(logger)
        logger.info(
            f"Validation complete: {report.valid_rows}/{report.total_rows} row(s) valid",
            extra={"context": "validation"},
        )
        return report

    def clean(self, df: pd.DataFrame, drop_duplicates: bool = True, drop_invalid_rows: bool = True) -> pd.DataFrame:
        cleaned = df.copy()

        if drop_duplicates:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            logger.info(
                f"Dropped {before - len(cleaned)} duplicate row(s)",
                extra={"context": "validation.clean"},
            )

        if drop_invalid_rows:
            valid_mask = []
            for _, row in cleaned.iterrows():
                try:
                    self._validate_row(row)
                    valid_mask.append(True)
                except DataValidationError:
                    valid_mask.append(False)
            before = len(cleaned)
            cleaned = cleaned[valid_mask]
            logger.info(
                f"Dropped {before - len(cleaned)} invalid row(s)",
                extra={"context": "validation.clean"},
            )

        return cleaned.reset_index(drop=True)


if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "district": ["Hyderabad", "Warangal", None, "Hyderabad", "Nizamabad"],
            "unemployment_rate": [5.2, 7.8, 4.1, 5.2, 150.0],
        }
    )

    schema = Schema(
        columns=[
            ColumnRule(name="district", dtype=str, required=True, allow_null=False),
            ColumnRule(name="unemployment_rate", dtype=float, min_value=0, max_value=100),
        ]
    )

    validator = DataValidator(schema)
    report = validator.validate(sample_data)
    print(report.summary())

    clean_df = validator.clean(sample_data)
    print("\nCleaned DataFrame:")
    print(clean_df)