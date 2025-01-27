import numpy as np
import pandas as pd
import PySimpleGUI as sg
import joblib
from PIL import Image
import io
from src import Utils
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import calamancy
import torch
from torch import nn

from transformers import (
    BertForSequenceClassification, 
    BertTokenizer
)

##### INITIALIZATION #####

DEVICE = 'cpu'

MODELS_FOLDER = 'models'

LEARNERS = None

##### LOADING FUNCTIONS #####

def load_joblib(filename: str):
    """
    Loads a pre-trained model from a specified file.

    Parameters:
    - filename (str): The path to the file containing the saved model.

    Returns:
    - The loaded model if the file is found, otherwise None.
    """
    try:
        model = joblib.load(f'{MODELS_FOLDER}/{filename}')
    except FileNotFoundError:
        sg.PopupError(
            f'ERROR: "{filename}" not found'
        )
        return None
    return model

def load_state_dict(model: nn.Module, filename: str):
    """
    Loads a pre-trained model from a specified state dict file.
    Used for Pytorch neural networks.

    Parameters:
    - filename (str): The path to the file containing the saved state dict.

    Returns:
    - The loaded model if the file is found, otherwise None.
    """
    try:
        state_dict = torch.load(
            f'{MODELS_FOLDER}/{filename}',
            map_location=DEVICE
        )
        model.load_state_dict(state_dict['model'])
    except FileNotFoundError:
        sg.PopupError(
            f'ERROR: "{filename}" not found'
        )
        return None
    return model


###### LEARNER PREPARATION ######

def load_tfidf() -> TfidfVectorizer:
    """
    Load TF-IDF. Since it is sklearn and saved through
    joblib, we can just load the joblib file
    """
    return load_joblib('model_bayes/best/tfidf.pkl')

def load_bayes() -> BernoulliNB:
    """
    Load Bernoulli Naive Bayes. Since it is sklearn and saved 
    through joblib, we can just load the joblib file
    """
    return load_joblib('model_bayes/best/bayes.pkl')

def prepare_calamancy():
    calamancy_model_name = "tl_calamancy_md-0.1.0"
    return calamancy.load(calamancy_model_name)

def prepare_lstm() -> nn.Module:
    INPUT_SIZE = 200  # Size of CalamanCy token vectors
    HIDDEN_SIZE = 50
    LINEAR_OUTPUT_SIZE = 2

    class LstmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                INPUT_SIZE,
                HIDDEN_SIZE,
            )
            self.linear = nn.Linear(
                HIDDEN_SIZE, 
                LINEAR_OUTPUT_SIZE,
            )

        def forward(self, input):
            lstm_output, _ = self.lstm(input)

            linear_output = self.linear(lstm_output[-1])

            return linear_output

    Lstm = LstmModel()

    Lstm.to(DEVICE)

    return Lstm

def load_lstm(model: nn.Module):
    return load_state_dict(
        model, 
        'model_lstm/best/lstm_checkpoint.pth',
    )

def prepare_bert_tokenizer():
    MODEL_NAME = "bert-base-multilingual-uncased"

    return BertTokenizer.from_pretrained(MODEL_NAME)

def prepare_bert() -> BertForSequenceClassification:
    MODEL_NAME = "bert-base-multilingual-uncased"

    return BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(DEVICE)

def load_bert(model: BertForSequenceClassification):
    return load_state_dict(
        model, 
        'model_bert/best/bert_checkpoint.pth',
    )

def load_logistic_regression() -> LogisticRegression:
    return load_joblib('model_lr/best/lr.pkl')


##### LOAD LEARNERS #####

def load_learners():
    """
    Window to show loading for learners
    """
    from time import sleep

    layout = [
        [sg.Text('Loading learners...', key='text')],
        [sg.ProgressBar(
            max_value=7,
            orientation='h', 
            # expand_x=True,
            size=(35, 20),
            key='progress',
        )],
    ]
    window = sg.Window(
        'Loading learners', 
        layout, 
        finalize=True
    )

    text = window['text']
    progress_bar = window['progress']

    # Load learners and update progress

    text.update('Loading TF-IDF')
    tfidf = load_tfidf()
    progress_bar.update_bar(1)
    sleep(1)

    text.update('Loading Bernoulli Naive Bayes')
    bayes = load_bayes()
    progress_bar.update_bar(2)
    sleep(1)

    text.update('Loading CalamanCy')
    calamancy = prepare_calamancy()
    progress_bar.update_bar(3)
    sleep(1)

    text.update('Loading LSTM')
    lstm = prepare_lstm()
    lstm_loaded = load_lstm(lstm)
    progress_bar.update_bar(4)
    sleep(1)

    text.update('Loading BERT Tokenizer')
    bert_tokenizer = prepare_bert_tokenizer()
    progress_bar.update_bar(5)
    sleep(1)

    text.update('Loading BERT')
    bert = prepare_bert()
    bert_loaded = load_bert(bert)
    progress_bar.update_bar(6)
    sleep(1)

    text.update('Loading Logistic Regression')
    lr = load_logistic_regression()
    sleep(1)

    window.close()

    return (
        tfidf,
        bayes,
        calamancy,
        lstm_loaded,
        bert_tokenizer,
        bert_loaded,
        lr
    )

def clean_text(text: str):
    """
    Processes the input text by removing unnecessary characters and sequences.

    Parameters:
    - text (str): The input text to be cleaned.

    Returns:
    - A cleaned version of the input text.
    """
    text = Utils.unmark(text)
    text = Utils.remove_urls(text)
    text = Utils.remove_usernames(text)
    text = Utils.remove_emojis(text)
    text = Utils.remove_escape_sequences(text)
    text = text.rstrip()
    return text

##### LEARNER PREDICTION FUNCTIONS #####

### Bayes ###

def predict_bayes(inputs: list):
    inputs_transformed = LEARNERS['tfidf'].transform(inputs)
    predictions = LEARNERS['bayes'].predict(inputs_transformed)
    return predictions

def predict_proba_bayes(inputs: list):
    inputs_transformed = LEARNERS['tfidf'].transform(inputs)
    predictions = LEARNERS['bayes'].predict_proba(inputs_transformed)
    return predictions

### LSTM ###

# Processing

def get_calamancy_tokens(data):
  # Allows it to work with both dataframes and
  # simple lists of strings
  if isinstance(data, pd.Series):
    data = data.values

  data = [text.lower() for text in data]

  samples = []

  for sample in LEARNERS['calamancy'].pipe(data):
    tokens = []
    for token in sample:
      if token.is_punct: continue
      if token.is_space: continue

      tokens.append(token)

    samples.append(tokens)

  return samples

def get_calamancy_token_vectors(tokens):
  vectors = []

  for sample in tokens:
    # vector = Tensor(np.array([token.vector for token in sample]))
    token_vectors = []
    # Check in case empty due to processing
    if not sample:
      token_vectors.append(np.zeros((200)))
    else:
      for token in sample:
        if token.has_vector:
          token_vectors.append(token.vector)
    token_vectors = torch.tensor(np.array(token_vectors))

    vectors.append(token_vectors)

  return vectors

def process_lstm(inputs):
    tokens = get_calamancy_tokens(inputs)
    vectors = get_calamancy_token_vectors(tokens)
    return vectors

# Predicting

def predict_proba_lstm(inputs: list):
    vectors = process_lstm(inputs)
    preds = []
    with torch.inference_mode():
        for sample in vectors:
            sample = sample.to(DEVICE)
            sample_pred = LEARNERS['lstm'](sample)
            preds.append(sample_pred)
    preds = torch.cat(preds)
    probabilities = nn.functional.softmax(preds, dim=0)

    return probabilities.cpu()

def predict_lstm(inputs: list):
  probabilities = predict_proba_lstm(inputs)
  discrete_probabilities = torch.argmax(
    probabilities,
    dim=0,
  )
  return discrete_probabilities

### mBERT ###

def process_bert(inputs):
    BERT_MAX_LENGTH = 250

    input_ids = []
    attention_masks = []

    for text in inputs:
        # Tokenize the text
        tokens = LEARNERS['bert_tokenizer'].tokenize(text)

        # Truncate the tokens if necessary
        if len(tokens) > BERT_MAX_LENGTH - 2:
            tokens = tokens[:BERT_MAX_LENGTH - 2]

        # Add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Convert tokens to token IDs
        token_ids = LEARNERS['bert_tokenizer'].convert_tokens_to_ids(tokens)

        # Pad the token IDs to BERT_MAX_LENGTH
        padding = [0] * (BERT_MAX_LENGTH - len(token_ids))
        token_ids += padding

        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * len(padding)

        input_ids.append(token_ids)
        attention_masks.append(attention_mask)

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids).to(DEVICE)
    attention_masks = torch.tensor(attention_masks).to(DEVICE)

    return input_ids, attention_masks

# Custom data loader
def data_loader(input_ids, attention_masks, batch_size):
    for i in range(0, len(input_ids), batch_size):
        yield input_ids[i:i+batch_size], attention_masks[i:i+batch_size]

def predict_proba_bert(inputs: list):
  with torch.inference_mode():
    input_ids, attention_masks = process_bert(inputs)

    all_predictions = []
    for batch_input_ids, batch_attention_masks in data_loader(input_ids, attention_masks, 5):
      batch_input_ids = batch_input_ids.to(DEVICE)
      batch_attention_masks = batch_attention_masks.to(DEVICE)

      predictions = LEARNERS['bert'](
        batch_input_ids,
        attention_mask=batch_attention_masks,
      ).logits

      probabilities = nn.functional.softmax(predictions, dim=1)

      all_predictions.append(probabilities)

    return torch.cat(all_predictions).cpu()

def predict_bert(inputs: list):
  probabilities = predict_proba_bert(inputs)
  discrete_probabilities = torch.argmax(
    probabilities,
    dim=1,
  )
  return discrete_probabilities

##### CONCATENATE LEARNER PREDICTIONS ######

def get_learner_single_predict(input: str):
    text = clean_text(input)
    bayes_pred = predict_bayes([text])
    lstm_pred = predict_lstm([text])
    bert_pred = predict_bert([text])
    return np.array([
        bayes_pred[0],
        lstm_pred.detach().numpy(),
        bert_pred[0].detach().numpy(),
    ])

def get_learner_single_predict_proba(input: str):
    text = clean_text(input)
    bayes_pred = predict_proba_bayes([text])
    lstm_pred = predict_proba_lstm([text])
    bert_pred = predict_proba_bert([text])
    return np.array([
        bayes_pred[0],
        lstm_pred.detach().numpy(),
        bert_pred[0].detach().numpy(),
    ])

##### ENSEMBLE METHODS #####

def predict_hard_voting(input: str):
    preds = get_learner_single_predict(input)
    return np.bincount(preds).argmax()

def predict_soft_voting(input: str):
    preds = get_learner_single_predict_proba(input)
    return np.average(preds, axis=0)

def predict_stacking(input: str):
    preds = get_learner_single_predict_proba(input)
    individual_preds = preds[:, 1:]
    transposed_preds = individual_preds.T
    return LEARNERS['logistic_regression'].predict_proba(transposed_preds)[0]

##### UI #####

def loading_popup(ensemble_string: str):
    """
    Displays a non-blocking popup indicating that an ensemble model is being loaded.

    Parameters:
    - ensemble_string (str): A string representing the type of ensemble being loaded.
    """
    sg.PopupNonBlocking(
        f'Loading {ensemble_string} ensemble...',
        button_type=sg.POPUP_BUTTONS_NO_BUTTONS,
        modal=True,
        # We do auto close to keep the loading popup active
        # while the thread execution is blocked due to
        # loading the ensemble
        auto_close=True,
        auto_close_duration=1,
    )

def default_table_values():
    """
    Defines default values for the table in the GUI.

    Returns:
    - A list of lists representing the default table values.
    """
    return [
        ['BNB', '-', '-'],
        ['LSTM', '-', '-'],
        ['mBERT', '-', '-'],
    ]

def input_column(ensemble):
    """
    Creates the GUI column for inputting text.

    Returns:
    - A PySimpleGUI Column element containing the input text field and buttons.
    """
    return sg.Column([
        [sg.Text(ensemble, font=('Helvetica', 16))],
        [sg.Text("Input text:")],
        [sg.Multiline(
            size=(40, 8), 
            key='-INPUT-',
            focus=True,
            autoscroll_only_at_bottom=True,
        )],
        [
            sg.Exit(
                button_text='Return',
            ),
            sg.Push(),
            sg.Button('Clear'),
            sg.Button('Predict'),
        ],
    ])

def output_column(table_values):
    """
    Creates the GUI column for displaying output.

    Parameters:
    - table_values: The values to display in the output table.

    Returns:
    - A PySimpleGUI Column element containing the output table.
    """
    return sg.Column([
        [sg.Table(
            values=table_values, 
            auto_size_columns=False,
            hide_vertical_scroll=True,
            justification='center',
            cols_justification=('r', 'c', 'c'),
            num_rows=3,
            key='-TABLE-',
            headings=(
                'Learner',
                '0 (Non-hate)', 
                '1 (Hate)',
            ),
            col_widths=(12, 17, 17)
        )],
        [
            sg.Text('Ensemble:', justification='r'), 
            sg.Text('-', key='-ENSEMBLE-', justification='l'),
        ],
    ], element_justification='c')

def output_column_hard(table_values):
    """
    Creates the GUI column for displaying output in hard voting.

    Parameters:
    - table_values: The values to display in the output table.

    Returns:
    - A PySimpleGUI Column element containing the output table.
    """
    return sg.Column([
        [sg.Table(
            values=table_values, 
            auto_size_columns=False,
            hide_vertical_scroll=True,
            justification='center',
            cols_justification=('r', 'c'),
            num_rows=3,
            key='-TABLE-',
            headings=(
                'Learner',
                'Vote',
            ),
            col_widths=(12, 8)
        )],
        [
            sg.Text('Ensemble:', justification='r'), 
            sg.Text('-', key='-ENSEMBLE-', justification='l'),
        ],
    ], element_justification='c')

def predict(ensemble, text: str):
    """
    Performs a language check before predicting the class of the input text.

    Parameters:
    - ensemble: The ensemble model used for prediction.
    - text (str): The input text to classify.

    Returns:
    - The prediction result and the individual learner predictions, or None if the
    language check fails.
    """
    result = {
        'hard': predict_hard_voting(text),
        'soft': predict_soft_voting(text),
        'stacking': predict_stacking(text),
    }[ensemble]

    if ensemble == 'hard':
        return result, get_learner_single_predict(text)
    else:
        return result, np.round(get_learner_single_predict_proba(text), 4)

# The following functions, `hard_voting`, `soft_voting`, `stacking`, and `event_loop`,
# define different ensemble strategies and handle the GUI event loop.

def hard_voting():
    loading_popup('hard voting')
    table_values = default_table_values()
    input_column_element = input_column('Hard Voting')
    output_column_element = output_column_hard(table_values)
    layout = [
        [
            input_column_element, 
            sg.VerticalSeparator(), 
            output_column_element,
        ]
    ]

    ensemble = 'hard'

    window = sg.Window(
        'Hard Voting Ensemble',
        layout,
        modal=True,
    )

    # Event Loop
    while True:
        event, values = window.read()

        if event == 'Predict':
            if not values['-INPUT-']:
                window['-ENSEMBLE-'].update('-')
                window['-TABLE-'].update(default_table_values())
                continue
            result, learner_predictions = predict(
                ensemble, 
                values['-INPUT-']
            )
            if result is None: continue
            table_values[0][1] = learner_predictions[0]
            table_values[1][1] = learner_predictions[1]
            table_values[2][1] = learner_predictions[2]
            window['-ENSEMBLE-'].update(
                'Non-hate' if result == 0 else 'Hate'
            )
            window['-TABLE-'].update(table_values)
        elif event == 'Clear':
            window['-INPUT-'].update('')
        elif event == sg.WIN_CLOSED or event == 'Return':
            break

    window.close()
    layout = None
    window = None
    gc.collect()

def soft_voting():
    loading_popup('soft voting')
    table_values = default_table_values()
    input_column_element = input_column('Soft Voting')
    output_column_element = output_column(table_values)
    layout = [
        [
            input_column_element, 
            sg.VerticalSeparator(), 
            output_column_element,
        ]
    ]

    ensemble = 'soft'

    window = sg.Window(
        'Soft Voting Ensemble',
        layout,
        modal=True,
    )

    # Event Loop
    while True:
        event, values = window.read()

        if event == 'Predict':
            if not values['-INPUT-']:
                window['-ENSEMBLE-'].update('-')
                window['-TABLE-'].update(default_table_values())
                continue
            result, learner_predictions = predict(
                ensemble, 
                values['-INPUT-']
            )
            if result is None: continue
            result = np.round(result, 4)
            table_values[0][1:] = learner_predictions[0]
            table_values[1][1:] = learner_predictions[1]
            table_values[2][1:] = learner_predictions[2]
            window['-ENSEMBLE-'].update(
                f"Non-hate - {result[0]} | Hate - {result[1]}"
            )
            window['-TABLE-'].update(table_values)
        elif event == 'Clear':
            window['-INPUT-'].update('')
        elif event == sg.WIN_CLOSED or event == 'Return':
            break

    window.close()
    layout = None
    window = None
    gc.collect()

def stacking():
    loading_popup('stacking')
    table_values = default_table_values()
    input_column_element = input_column('Stacking')
    output_column_element = output_column(table_values)
    layout = [
        [
            input_column_element, 
            sg.VerticalSeparator(), 
            output_column_element,
        ]
    ]

    ensemble = 'stacking'

    window = sg.Window(
        'Stacking Ensemble',
        layout,
        modal=True,
    )

    # Event Loop
    while True:
        event, values = window.read()

        if event == 'Predict':
            if not values['-INPUT-']:
                window['-ENSEMBLE-'].update('-')
                window['-TABLE-'].update(default_table_values())
                continue
            result, learner_predictions = predict(
                ensemble, 
                values['-INPUT-']
            )
            if result is None: continue
            result = np.round(result, 4)
            table_values[0][1:] = learner_predictions[0]
            table_values[1][1:] = learner_predictions[1]
            table_values[2][1:] = learner_predictions[2]
            window['-ENSEMBLE-'].update(
                f"Non-hate - {result[0]} | Hate - {result[1]}"
            )
            window['-TABLE-'].update(table_values)
        elif event == 'Clear':
            window['-INPUT-'].update('')
        elif event == sg.WIN_CLOSED or event == 'Return':
            break

    window.close()
    layout = None
    window = None
    gc.collect()

def event_loop(window: sg.Window):
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            break
        elif event == 'Hard Voting':
            hard_voting()
        elif event == 'Soft Voting':
            soft_voting()
        elif event == 'Stacking':
            stacking()

def get_resized_image(image_path, width, height):
    """
    Resizes an image to specified dimensions using PIL.

    Parameters:
    - image_path (str): Path to the image file.
    - width (int): Desired width of the resized image.
    - height (int): Desired height of the resized image.

    Returns:
    - A byte representation of the resized image. 
    """
    image = Image.open(image_path)
    image = image.resize((width, height), Image.Resampling.LANCZOS)  # High-quality downsampling filter
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()

def main_window():
    """
    Initializes and displays the main window of the GUI application.
    """
    sg.theme('LightBlue5')

    # Define a frame for the buttons arranged horizontally
    button_frame = sg.Frame(
        'Select an Ensemble:',
        layout=[
            [
                sg.Button('Hard Voting', size=(15, 1), font=('Helvetica', 12)),
                sg.Button('Soft Voting', size=(15, 1), font=('Helvetica', 12)),
                sg.Button('Stacking', size=(15, 1), font=('Helvetica', 12))
            ],
        ],
        element_justification='c',
        relief=sg.RELIEF_SUNKEN
    )

    resized_logo = get_resized_image('logo.png', width=150, height=150)

    layout = [
        [sg.Image(data=resized_logo)],
        [button_frame],
        [sg.Exit(size=(10, 1), pad=((0, 0), (10, 0)))]
    ]

    # Create the Window
    window = sg.Window('TAG-HATE-ELF', layout, element_justification='c')

    event_loop(window)

    window.close()


if __name__ == "__main__":
    sg.theme('LightBlue5')

    learners = load_learners()

    if not learners:
        exit(1)

    LEARNERS = {
        'tfidf': learners[0],
        'bayes': learners[1],
        'calamancy': learners[2],
        'lstm': learners[3],
        'bert_tokenizer': learners[4],
        'bert': learners[5],
        'logistic_regression': learners[6],
    }

    main_window()
