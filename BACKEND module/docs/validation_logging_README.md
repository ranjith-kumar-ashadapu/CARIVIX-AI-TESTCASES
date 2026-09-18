# CARIVIX AI — Data Validation, Error Handling & Logging

**Author:** Sindhu Bollikonda (Python/R Developer)
**Component:** Data Processing Engine (Module 2), CARIVIX AI Platform

Implements two assigned deliverables:
1. Data validation and error-handling modules
2. Logging improvements

## Files

| File | Purpose |
|---|---|
| `logging_config.py` | Centralized logger factory — console + rotating file handlers, optional JSON output for log aggregation, per-module loggers via `get_logger(__name__)`. |
| `error_handling.py` | Custom exception hierarchy (`DataValidationError`, `DataSourceError`, `SchemaError`, `ProcessingError`, `ConfigurationError`), `@safe_run` retry/backoff decorator, `ErrorCollector` for batch jobs that should keep going past bad rows. |
| `data_validation.py` | Schema-driven validator for pandas DataFrames: missing values, duplicates, type/range/allowed-value checks, `ValidationReport`, and `.clean()` to drop bad rows. |
| `data_validation.R` | R equivalent of the above (logger, custom conditions, `safe_run`, schema validation, `clean_data`) for R-based statistical/forecasting scripts. |
| `example_pipeline.py` | End-to-end demo wiring all three modules together, styled after the CARIVIX Data Acquisition → Data Processing flow. |

## Quick start (Python)

```python
from data_validation import Schema, ColumnRule, DataValidator
from logging_config import get_logger

logger = get_logger(__name__)

schema = Schema(columns=[
    ColumnRule(name="district", dtype=str, required=True, allow_null=False),
    ColumnRule(name="unemployment_rate", dtype=float, min_value=0, max_value=100),
])

validator = DataValidator(schema)
report = validator.validate(df)
print(report.summary())

clean_df = validator.clean(df)
```

## Quick start (R)

```r
source("data_validation.R")

schema <- list(
  column_rule("district", required = TRUE, allow_null = FALSE),
  column_rule("unemployment_rate", min_value = 0, max_value = 100)
)

report <- validate_data(df, schema)
clean_df <- clean_data(df, schema)
```

## Retry-safe operations

```python
from error_handling import safe_run, DataSourceError

@safe_run(retries=3, delay_seconds=2, exceptions=(DataSourceError,))
def fetch_from_api(url):
    ...
```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `CARIVIX_LOG_DIR` | `./logs` | Log file directory |
| `CARIVIX_LOG_LEVEL` | `INFO` | Minimum log level |
| `CARIVIX_LOG_MAX_BYTES` | `5242880` (5MB) | Rotation size threshold |
| `CARIVIX_LOG_BACKUP_COUNT` | `5` | Number of rotated files kept |
| `CARIVIX_LOG_JSON` | `false` | Write file logs as JSON (for log aggregation) |

## Design notes

- Validation is **schema-driven** (`Schema` + `ColumnRule`) so new datasets across Business, Economic, Government, and Smart City Intelligence domains can reuse the same engine without new code.
- `ErrorCollector` lets batch jobs report *all* bad rows at once rather than halting on the first failure — important for large government/census datasets.
- `safe_run` retry/backoff targets the flaky external sources described in the CARIVIX Data Acquisition Engine (APIs, web scraping, IoT sensor feeds).
- Logging is centralized so every module's logs are consistent, timestamped, and rotated — ready to feed into the cloud log pipelines implied by the CARIVIX Technology Stack (AWS/Azure/GCP).

## Tested

All Python modules include a `if __name__ == "__main__":` self-test block and were run successfully in this environment. The R script follows the same logic but could not be executed here (R not installed in this sandbox) — verify with `Rscript data_validation.R` in an R-enabled environment.
