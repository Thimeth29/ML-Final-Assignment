"""data_ingestion.py

Optional stage: if you want to download or validate the raw dataset automatically.
For this assignment, many teams simply place the CSV into data/raw/ and track it with DVC.

TODO (optional):
- validate that file exists
- validate required columns
- write a simple data quality report
"""

import argparse
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    # TODO: fill with actual columns from telco dataset once you inspect it
    # e.g. "customerID", "gender", ...
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--report_out", default="reports/data_quality_report.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Raw data not found: {in_path}")

    df = pd.read_csv(in_path)

    # TODO: check required columns, missing values, basic stats, etc.
    report = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_by_column": df.isna().sum().to_dict(),
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Data ingestion/validation report saved to: {args.report_out}")

if __name__ == "__main__":
    main()
