# CARIVIX AI — Data Import Modules

Built for: **Module 1 — Data Acquisition Engine**
Assigned task: *Develop data import modules for CSV, Excel, JSON, and APIs; create reusable configuration files.*

## What's included

```
carivix_data_import/
├── config/
│   └── data_sources.yaml      # ALL data sources defined here — no code changes needed to add one
├── importers/
│   ├── base_importer.py       # Shared interface: load(), validation, error handling
│   ├── csv_importer.py        # CSV files (census data, budget reports, etc.)
│   ├── excel_importer.py      # Excel workbooks, single or all sheets
│   ├── json_importer.py       # JSON, including nested records + flattening
│   └── api_importer.py        # REST APIs: auth, retries/backoff, pagination
├── utils/
│   ├── config_loader.py       # Loads YAML config, merges with defaults
│   └── logger.py              # Consistent logging across all importers
├── main.py                    # CLI: load any source by name
└── requirements.txt
```

## Design principles

- **One config file, no hardcoded paths.** Every data source (its file path, sheet name,
  delimiter, API auth, etc.) lives in `config/data_sources.yaml`. Adding a new source means
  editing YAML, not Python.
- **Consistent interface.** Every importer inherits `BaseImporter` and exposes the same
  `.load()` method returning a pandas DataFrame, so downstream code (Data Processing Engine,
  Module 2) doesn't care which source type it's dealing with.
- **Defaults + overrides.** Global defaults per category (csv/excel/json/api) live under
  `defaults:` in the YAML; each source only needs to specify what's different.
- **API resilience.** The API importer supports API-key / bearer / basic auth (via env vars,
  never hardcoded secrets), automatic retries with exponential backoff, and page-based
  pagination.

## Usage

```bash
pip install -r requirements.txt

# List all configured sources
python main.py --list

# Load a specific source and preview it
python main.py --category csv_sources --name census_data

# Load and save to disk
python main.py --category excel_sources --name company_financials --save output.csv
```

### Programmatic usage

```python
from utils.config_loader import ConfigLoader
from importers import get_importer

loader = ConfigLoader()
cfg = loader.get_source("json_sources", "government_open_data")
importer = get_importer("json_sources", cfg, source_name="government_open_data")

df = importer.load()
```

## Adding a new source

Just add an entry under the relevant category in `config/data_sources.yaml` — no code
changes required. For API sources, set `api_key_env_var` to the name of an environment
variable holding the credential (never commit real keys to the config file).

## Notes / next steps

- `data/raw/` is where source files are expected to live locally (not included — add your
  own CSV/Excel/JSON files matching the paths in the config, or point the config at your
  actual data directory).
- Hooks into **Module 2 (Data Processing Engine)** naturally: each importer returns a plain
  pandas DataFrame, ready for the cleaning/transformation pipeline (Spark/Kafka/Airflow).
