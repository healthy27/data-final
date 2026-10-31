"""Small CLI with an offline demo and a separate authenticated data path."""

import argparse
import json
from pathlib import Path

import pandas as pd

from data_final.analysis import run_analysis
from data_final.dart import collect
from data_final.data import quality_summary, synthetic_panel, validate_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a clearly labelled synthetic demonstration")
    demo.add_argument("--output", type=Path, default=Path("reports/runs/demo"))
    demo.add_argument("--seed", type=int, default=42)
    fetch = commands.add_parser("collect", help="Collect annual OFS data; needs DART_API_KEY")
    fetch.add_argument("--universe", type=Path, required=True)
    fetch.add_argument("--start-year", type=int, default=2018)
    fetch.add_argument("--end-year", type=int, default=2024)
    fetch.add_argument("--output", type=Path, default=Path("data/processed/panel.csv"))
    fetch.add_argument("--cache", type=Path, default=Path("data/raw/dart"))
    validate = commands.add_parser("validate")
    validate.add_argument("input", type=Path)
    run = commands.add_parser("analyze")
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, default=Path("reports/runs/real"))
    run.add_argument("--test-year", type=int, default=2024)
    args = parser.parse_args()
    if args.command == "demo":
        panel = synthetic_panel(args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.output / "panel.csv", index=False)
        result = run_analysis(panel, args.output, seed=args.seed)
    elif args.command == "collect":
        if not 2015 <= args.start_year <= args.end_year <= 2100:
            parser.error("Expected 2015 <= start-year <= end-year <= 2100")
        result = collect(
            args.universe, list(range(args.start_year, args.end_year + 1)), args.output, args.cache
        )
    else:
        panel = validate_panel(pd.read_csv(args.input, dtype={"corp_code": str, "reprt_code": str}))
        result = (
            quality_summary(panel)
            if args.command == "validate"
            else run_analysis(panel, args.output, test_year=args.test_year)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
