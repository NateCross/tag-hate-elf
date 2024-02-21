import argparse
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src import Ensemble, LSTM
from imblearn.under_sampling import RandomUnderSampler

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
        return pd.read_csv(
            filename, 
            lineterminator='\n',
            dtype={
                'label': float,
            },
        )
    except FileNotFoundError:
        print("ERROR: File not found")
        exit(1)

def seed_random_number_generators(seed = 0):
    """
    Seeds pytorch and numpy for consistency
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_train_test_split(data_frame: pd.DataFrame):
    """
    Args:
        data_frame (pd.DataFrame): The data frame obtained from
            reading a csv file
    Returns:
        X_train
        X_test
        y_train
        y_test
    """
    text = data_frame['text']
    labels = data_frame['label']

    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        text, 
        labels, 
        test_size=0.2,  # 80/20 train/test split
        random_state=42,
    )

    # y_train = y_train.float()
    # y_test = y_test.float()

    return X_train, X_test, y_train, y_test

def undersample_data(X_train: list, y_train: list):
    """
    Undersample the data in case of imbalances.
    This way, there will be an even distribution
    among the hate and non-hate speech.
    Uses random undersampling because it is simple
    but sufficient for this use case.

    Args:
        X_train: From the csv after train-test split
        y_train: From the csv after train-test split
    """
    sampler = RandomUnderSampler(random_state=42)
    return sampler.fit_resample(X_train, y_train)


def train_ensemble(X_train: list, y_train: list, ensemble):
    """
    Train the ensemble using the csv data.

    Args:
        X_train: From the csv after train-test split
        y_train: From the csv after train-test split
        ensemble (VotingClassifier | StackingClassifier):
            A hard voting, soft voting, or stacking ensemble
            from scikit-learn
    """
    seed_random_number_generators()
    ensemble.fit(X_train, y_train)
    return ensemble

def get_prediction_results(X_test: list, y_test: list, ensemble):
    """
    Predict with the trained ensemble using the csv data.
    Gives quantifiable results.

    Args:
        X_test: From the csv after train-test split
        y_test: From the csv after train-test split
        ensemble (VotingClassifier | StackingClassifier):
            A trained hard voting, soft voting, or stacking 
            ensemble from scikit-learn
    """
    with torch.inference_mode():
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

def save_trained_model(ensemble, filename = "Ensemble.pt"):
    """
    Saves the ensemble to disk.
    We use save_params for portability, but we must
    ensure to initialize the ensemble before loading params
    when we predict.
    The behavior of save_params is further documented here: 
    https://skorch.readthedocs.io/en/stable/user/save_load.html

    Args:
        ensemble (VotingClassifier | StackingClassifier):
            A trained hard voting, soft voting, or stacking 
            ensemble from scikit-learn
    """
    ensemble.save_params(f_params=filename)

if __name__ == "__main__":
    args = parse_arguments()

    FILENAME = args.file
    ENSEMBLE_METHOD = args.ensemble_method

    DATA_FRAME = read_csv_file(FILENAME)

    X_train, X_test, y_train, y_test = get_train_test_split(
        DATA_FRAME
    )

    # Set the dataset to properly get dictionary length
    # for the tokenization and vectorization
    LSTM.dataset.change(X_train)

    # Select the ensemble method from the script's args
    ENSEMBLE = {
        'hard': Ensemble.HardVotingEnsemble(),
        'soft': Ensemble.SoftVotingEnsemble(),
        'stacking': Ensemble.StackingEnsemble(),
    }[ENSEMBLE_METHOD]

    ENSEMBLE = train_ensemble(X_train, y_train, ENSEMBLE)

    accuracy = get_prediction_results(X_test, y_test, ENSEMBLE)

    print(f"Accuracy: {accuracy}")

    save_trained_model(ENSEMBLE)

    print("Saved ensemble to disk")


