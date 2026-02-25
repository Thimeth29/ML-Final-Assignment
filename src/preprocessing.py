"""preprocessing.py

PART 1 – Data Engineering:
- Handle missing values
- Convert TotalCharges properly
- Encode categorical variables
- Scale numeric features
- Train-test split
- Save processed train/test CSVs into data/processed/
- Track processed data with DVC

This file is a template. Replace TODO sections with your implementation.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # TODO 1: Convert TotalCharges properly (often has blanks -> NaN)
    # Example idea (adjust to your dataset columns):
    # df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # TODO 2: Handle missing values (drop or impute depending on column)
    # Example:
    # df = df.dropna(subset=["TotalCharges"])

    # TODO 3: Encode categorical variables
    # Option A: one-hot encoding via pandas.get_dummies
    # Option B: sklearn ColumnTransformer + OneHotEncoder (more production-friendly)

    # TODO 4: Scale numeric features (StandardScaler / MinMaxScaler)
    # If you use sklearn pipeline, you may not need to output scaled CSVs, but assignment asks processed data.
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--outdir", required=True, help="Output directory for processed data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # TODO 5: Ensure target column name matches dataset (commonly 'Churn' Yes/No)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found. Update preprocessing accordingly.")

    df_clean = preprocess(df)

    # Split: keep target in the CSVs for simplicity
    train_df, test_df = train_test_split(
        df_clean,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df_clean["Churn"] if "Churn" in df_clean.columns else None,
    )

    train_path = outdir / "train.csv"
    test_path = outdir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Saved: {train_path}")
    print(f"✅ Saved: {test_path}")

if __name__ == "__main__":
    main()
