# src/data_ingestion.py
import argparse
from pathlib import Path
import shutil

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Path to original dataset CSV")
    p.add_argument("--dest", required=True, help="Path to data/raw/*.csv")
    args = p.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    shutil.copy2(src, dst)
    print(f"Copied dataset to {dst}")

if __name__ == "__main__":
    main()