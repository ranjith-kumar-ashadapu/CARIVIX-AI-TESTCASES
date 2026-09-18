"""
main.py
CARIVIX AI - Data Acquisition Engine
Command-line entry point: load any configured data source by name
and preview/save the result.

Usage:
    python main.py --category csv_sources --name census_data
    python main.py --category excel_sources --name company_financials --save data/processed/out.csv
    python main.py --list
"""

from __future__ import annotations

import argparse
import sys

from importers import get_importer
from utils.config_loader import ConfigError, ConfigLoader


def list_all_sources(loader: ConfigLoader) -> None:
    for category in ("csv_sources", "excel_sources", "json_sources", "api_sources"):
        names = loader.list_sources(category)
        print(f"\n{category}:")
        if not names:
            print("  (none configured)")
        for name in names:
            print(f"  - {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CARIVIX AI Data Import CLI")
    parser.add_argument("--config", help="Path to data_sources.yaml (optional)")
    parser.add_argument(
        "--category",
        choices=["csv_sources", "excel_sources", "json_sources", "api_sources"],
        help="Category of the source to load",
    )
    parser.add_argument("--name", help="Name of the source as defined in the config")
    parser.add_argument("--save", help="Optional path to save the result as CSV")
    parser.add_argument("--list", action="store_true", help="List all configured sources")
    args = parser.parse_args()

    try:
        loader = ConfigLoader(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    if args.list:
        list_all_sources(loader)
        return 0

    if not args.category or not args.name:
        parser.print_help()
        return 1

    try:
        source_cfg = loader.get_source(args.category, args.name)
        importer = get_importer(args.category, source_cfg, source_name=args.name)
        result = importer.load()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load '{args.name}': {exc}", file=sys.stderr)
        return 1

    if isinstance(result, dict):  # multi-sheet Excel result
        for sheet, df in result.items():
            print(f"\n--- Sheet: {sheet} ---")
            print(df.head())
    else:
        print(result.head())

        if args.save:
            result.to_csv(args.save, index=False)
            print(f"\nSaved to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
