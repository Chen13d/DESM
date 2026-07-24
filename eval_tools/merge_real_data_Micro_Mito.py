import os
from glob import glob

import pandas as pd


def merge_prefixed_csvs(folder_path, output_path):
    csv_files = glob(os.path.join(folder_path, "*.csv"))
    merged_df = None

    for file in csv_files:
        try:
            filename = os.path.basename(file)
            prefix = os.path.splitext(filename)[0]
            df = pd.read_csv(file).add_prefix(f"{prefix}_")
            merged_df = df if merged_df is None else pd.concat([merged_df, df], axis=1)
        except Exception as e:
            print(f"Failed to read {file}. Error: {e}")

    if merged_df is not None:
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully saved to: {output_path}")
    else:
        print("No files available for merging.")


if __name__ == "__main__":
    print("merge_real_data_Micro_Mito.py is a utility helper. Please edit the input/output folders before running it directly.")
