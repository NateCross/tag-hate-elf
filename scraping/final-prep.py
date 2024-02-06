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

    # Drop all rows with missing or non-number values
    csv = csv[pd.to_numeric(csv['label'], errors='coerce').notnull()]

    # Extract X and y from the csv, this allows the data to be
    # undersampled
    X = csv.iloc[:, 0]
    y = csv.iloc[:, 1]

    # Reshape X into a 2D array to be compatible with the
    # undersampler
    X = X.values.reshape(-1, 1)

    # Initialize the random undersampler
    sampler = RandomUnderSampler(random_state=42)

    # Undersample the data
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    except ValueError:
        print("ERROR: Insufficient data")
        exit()

    # Flatten X again after resampling so it returns to
    # a 1D list
    X_resampled = X_resampled.flatten()

    # Make a new dataframe with the resampled data
    # These columns have the same name as the 
    # 2016 and 2022 PH Hate Speech dataset
    final_csv = pd.DataFrame(
        list(zip(X_resampled, y_resampled)),
        columns=['text', 'label']
    )

    # Save the filtered and resampled CSV
    split_filename = CSV_FILENAME.split(".")

    final_csv.to_csv(
        f"{split_filename[0]}-final.csv",
        index=False,    # Prevent index from being saved as column
    )

    print("Finished")
