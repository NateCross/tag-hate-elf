"""
This script is to use for the final preparation before training
Make sure to scrape for reddit with reddit-scrape.py,
then run the filter-data.py to prepare for annotation,
and finally this one
"""
import argparse
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_filename",
        help="The csv to filter",
    ) # Command line argument for the CSV filename
    args = parser.parse_args()

    CSV_FILENAME = args.csv_filename

    # Read the CSV file
    try:
        csv = pd.read_csv(CSV_FILENAME, lineterminator='\n')
    except FileNotFoundError:
        print("ERROR: File not found")
        exit(1)

    # Drop submission-related columns
    csv = csv.drop(columns=[
        'submission_name',
        'submission_text',
    ])

    # Rename body column to be the same as the
    # 2016 and 2022 PH Hate Speech dataset
    csv = csv.rename(columns={
        'body': 'text'
    })

    # Drop all rows with missing or non-number values
    csv = csv[pd.to_numeric(csv['label'], errors='coerce').notnull()]

    # Save the filtered CSV
    split_filename = CSV_FILENAME.split(".")

    csv.to_csv(
        f"{split_filename[0]}-final.csv",
        index=False,    # Prevent index from being saved as column
    )

    print("Finished")
