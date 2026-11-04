"""
EECS 445 - Fall 2026
Project 2 - test_output.py

Use this script to confirm that your challenge predictions (stored in uniqname.csv) are correctly formatted.

Usage: python test_output.py uniqname.csv

Do not edit test_output.py
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", type=str, help="your uniqname.csv file")


def main():
    """Check that the output file is correctly formatted."""
    args = parser.parse_args()

    input_file = args.input_csv

    if input_file[-4:] != ".csv":
        raise RuntimeError("Input file must be a csv file")
    
    df = pd.read_csv(input_file)

    if len(df.columns) != 1 or df.columns[0] not in ["predictions", "predictions_gpu"]:
        raise RuntimeError("Input file must have only one column named 'predictions' or 'predictions_gpu'.")
    
    if len(df) != 200:
        raise RuntimeError(f"There are 200 challenge heldout datapoints. You have {len(df)} predictions.")
    
    # Check that each row is a float between 0 and 1
    for i, row in df.iterrows():
        col = df.columns[0]
        if not (isinstance(row[col], float) and 0 <= row[col] <= 1):
            raise RuntimeError(f"Row {i} is not a float between 0 and 1.")

    print("Output file is correctly formatted.")

if __name__ == "__main__":
    main()
