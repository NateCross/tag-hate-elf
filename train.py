import argparse
import pandas as pd

def parse_arguments() -> argparse.Namespace:
    """
    Define the command line arguments for this training script

    Returns:
        argparse.Namespace: The arguments of the script.
            Accessible through 'file' and 'ensemble_method'
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'file',
        help='CSV File to use as training data',
        type=str,
    )

    parser.add_argument(
        'ensemble_method',
        help='Ensemble learning method to be used, between hard voting, soft voting, and stacking',
        choices=('hard', 'soft', 'stacking')
    )

    return parser.parse_args()

def read_csv_file(filename: str) -> pd.DataFrame:
    """
    Read the given file. Raises an error and quits script
    if the file does not exist.

    Args:
        filename (str): A string filename. Must be a csv file.

    Returns:
        pd.DataFrame: A pandas data frame whose data can be
            manipulated
    """
    try:
        return pd.read_csv(filename, lineterminator='\n')
    except FileNotFoundError:
        print("ERROR: File not found")
        exit(1)

if __name__ == "__main__":
    args = parse_arguments()

    FILENAME = args.file
    ENSEMBLE_METHOD = args.ensemble_method

    FILE = read_csv_file(FILENAME)
