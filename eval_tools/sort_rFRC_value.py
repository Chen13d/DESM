import argparse
from pathlib import Path

import pandas as pd


def sort_rfrc_csv(in_csv, out_csv=None):
    in_csv = Path(in_csv)
    out_csv = Path(out_csv) if out_csv else in_csv.with_name(f"{in_csv.stem}_sorted.csv")
    df = pd.read_csv(in_csv)

    value_cols = [c for c in df.columns if "rFRC_value" in c]
    pairs = []
    for vcol in value_cols:
        mcol = vcol.replace("rFRC_value", "mean_FRC")
        if mcol in df.columns:
            pairs.append((mcol, vcol))
        else:
            print(f"[Skipped] Could not find the mean_FRC column corresponding to {vcol}: {mcol}")

    sorted_parts = []
    for mcol, vcol in pairs:
        part = df[[mcol, vcol]].sort_values(by=vcol, ascending=True, na_position="last").reset_index(drop=True)
        sorted_parts.append(part)

    other_cols = [c for c in df.columns if c not in {c for p in pairs for c in p}]
    if other_cols:
        sorted_parts.append(df[other_cols].reset_index(drop=True))

    out_df = pd.concat(sorted_parts, axis=1)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved to: {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort rFRC CSV columns by rFRC value.")
    parser.add_argument("in_csv", help="Path to the input rFRC CSV file.")
    parser.add_argument("-o", "--out-csv", default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    sort_rfrc_csv(args.in_csv, args.out_csv)
