from markdown import Markdown
from io import StringIO
import re
import emoji
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import torch
import math
from os.path import dirname

def seed_random_number_generators(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

seed_random_number_generators()

# Function to convert markdown formatted text to plain text
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()

# Patching the Markdown library to add a plain text output format
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False

# Wrapper function to convert markdown text to plain text
def unmark(text):
    return __md.convert(text)

# Function to remove URLs from a text string and replace them with '[LINK]'
def remove_urls(text):
    url_pattern = re.compile(r'http\S+', re.IGNORECASE)
    return url_pattern.sub('[LINK]', text)

# Function to remove Reddit usernames from a text string and replace them with '[USERNAME]'
def remove_usernames(text):
    username_pattern = re.compile(r"/?u/[A-Za-z0-9_-]+", re.IGNORECASE)
    return username_pattern.sub('[USERNAME]', text)

# Function to remove emojis using the emoji python library
def remove_emojis(text):
    return emoji.replace_emoji(text, '')

# Replace escape sequences with a space
def remove_escape_sequences(text):
    escape_pattern = re.compile(r'[\r\n\t]', re.IGNORECASE)
    return escape_pattern.sub(' ', text)

def read_csv_file(filename: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filename, lineterminator='\n', usecols=range(2))
        print("CSV file read successfully!")
        return data
    except FileNotFoundError:
        print("ERROR: File not found")
        exit(1)

def save_trained_model(model, filename="Ensemble"):
    import joblib
    joblib.dump(model, f'{filename}.pkl', compress=True)
    print(f"Ensemble model saved to {model}.pkl")

def get_prediction_results(X_test: list, y_test: list, model):
    with torch.inference_mode():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nF1-score: {f1}")
        return accuracy, recall, precision, f1

def train_ensemble(X_train: list, y_train: list, ensemble):
    seed_random_number_generators()  # Ensure reproducibility
    ensemble.fit(X_train, y_train)
    print("Ensemble model trained.")
    return ensemble

def shuffle_data_frame(data_frame):
    text = list(data_frame['text'])
    label = list(data_frame['label'])

    assert(len(text) == len(label))

    indices = list(range(len(label)))

    # Make a random number generator that will shuffle list of indices
    # It is seeded to be reproducible
    random_number_generator = np.random.default_rng(seed=0)
    random_number_generator.shuffle(indices)

    shuffled_text = []
    shuffled_labels = []

    # Iterate through the list of indices and add the original data
    # from those shuffled indices
    for index in indices:
        shuffled_text.append(text[index])
        shuffled_labels.append(label[index])

    return pd.DataFrame({
        'text': shuffled_text,
        'label': shuffled_labels,
    })


def get_train_test_split(data_frame: pd.DataFrame, test_size: float):
    """
    Makes a stratified train test split.
    This aims to preserve the distribution between classes.
    """
    if not (1 >= test_size >= 0):
        print('ERROR: test_size must be between 0 and 1')
        return

    data_frame = shuffle_data_frame(data_frame)

    data_frame_length = len(data_frame)
    train_size = 1 - test_size

    nonhate_rows = data_frame[data_frame['label'] == 0] 
    nonhate_row_length = len(nonhate_rows)

    nonhate_row_train_size = math.ceil(nonhate_row_length * train_size)

    nonhate_row_train = nonhate_rows[0:nonhate_row_train_size]
    nonhate_row_test = nonhate_rows[nonhate_row_train_size:nonhate_row_length]

    assert(len(nonhate_row_train) + len(nonhate_row_test) == nonhate_row_length)

    hate_rows = data_frame[data_frame['label'] == 1] 
    hate_row_length = len(hate_rows)

    hate_row_train_size = math.ceil(hate_row_length * train_size)

    hate_row_train = hate_rows[0:hate_row_train_size]
    hate_row_test = hate_rows[hate_row_train_size:hate_row_length]

    assert(len(hate_row_train) + len(hate_row_test) == hate_row_length)

    combined_train = pd.concat([nonhate_row_train, hate_row_train])
    combined_test = pd.concat([nonhate_row_test, hate_row_test])

    assert(len(combined_train) + len(combined_test) == data_frame_length)

    shuffled_train = shuffle_data_frame(combined_train)
    shuffled_test = shuffle_data_frame(combined_test)

    assert(len(shuffled_train) + len(shuffled_test) == data_frame_length)

    return (
        shuffled_train['text'],
        shuffled_test['text'],
        shuffled_train['label'],
        shuffled_test['label'],
    )
def save_history_to_csv(model, pipeline_model_name, filename):
    data_frame = pd.DataFrame(
        model[pipeline_model_name].history
    ).set_index('epoch')
    data_frame.to_csv(filename)

def get_stop_words():
    STOP_WORDS = open(
        f"{dirname(__file__)}/stopwords-tl.txt", 
        "r",
    ).read().split('\n')
    return STOP_WORDS
