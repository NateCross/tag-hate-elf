import numpy as np
import PySimpleGUI as sg
import joblib
from sklearn.ensemble import VotingClassifier
from PIL import Image
import io
from src import Utils
from langdetect import detect_langs, LangDetectException, DetectorFactory
import gc

# Seed langdetect to make it more deterministic
DetectorFactory.seed = 0

def load_ensemble(filename: str):
    """
    Loads a pre-trained ensemble model from a specified file.

    Parameters:
    - filename (str): The path to the file containing the saved ensemble model.

    Returns:
    - The loaded ensemble model if the file is found, otherwise None.
    """
    try:
        ensemble = joblib.load(filename)
    except FileNotFoundError:
        sg.PopupError(
            f'ERROR: "{filename}" not found. Please train the ensemble first.'
        )
        return None
    return ensemble

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

def predict(ensemble, text: str):
    """
    Predicts the class of the input text using the ensemble model.

    Parameters:
    - ensemble: The ensemble model used for prediction.
    - text (str): The input text to classify.

    Returns:
    - The prediction result and the individual learner predictions.
    """
    text = clean_text(text)
    learner_predictions = [
        np.round(estimator.predict_proba([text]), 4)
        for estimator in ensemble.estimators_
    ]
    if isinstance(ensemble, VotingClassifier) and ensemble.voting == 'hard':
        return ensemble.predict([text]), learner_predictions
    else:
        return (
            np.round(ensemble.predict_proba([text]), 4), 
            learner_predictions,
        )

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

def input_column():
    """
    Creates the GUI column for inputting text.

    Returns:
    - A PySimpleGUI Column element containing the input text field and buttons.
    """
    return sg.Column([
        [sg.Text("Input text:")],
        [sg.Multiline(size=(40, 8), key='-INPUT-')],
        [
            sg.Exit(),
            sg.Push(),
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
            auto_size_columns=True,
            hide_vertical_scroll=True,
            justification='center',
            cols_justification=('r', 'c', 'c'),
            num_rows=3,
            key='-TABLE-',
            headings=(
                'Learner',
                '0 (Non-hate)', 
                '1 (Hate)',
            )
        )],
        [
            sg.Text('Ensemble:', justification='r'), 
            sg.Text('-', key='-ENSEMBLE-', justification='l'),
        ],
    ], element_justification='c')

def check_language(text):
    """
    Checks the language of the input text.

    Parameters:
    - text (str): The input text to check.

    Returns:
    - True if the text is in English or Tagalog, False otherwise.
    """
    try:
        # Detect the language of the text
        langs = detect_langs(text)
    except LangDetectException:
        # Return False if language detection fails
        return False
    # Return True if the text is in English or Tagalog, False otherwise
    # Implementing like this over detect to allow for text
    # where en or tl are not the most prominent language
    return any(lang.lang in ['en', 'tl'] for lang in langs)

def predict_with_language_check(ensemble, text: str):
    """
    Performs a language check before predicting the class of the input text.

    Parameters:
    - ensemble: The ensemble model used for prediction.
    - text (str): The input text to classify.

    Returns:
    - The prediction result and the individual learner predictions, or None if the
    language check fails.
    """
    if not check_language(text):
        sg.PopupError(
            'Error: Input must be in English or Tagalog.', 
            title='Language Error'
        )
        return None, None  # Return None to indicate that no prediction was made
    return predict(ensemble, text)

# The following functions, `hard_voting`, `soft_voting`, `stacking`, and `event_loop`,
# define different ensemble strategies and handle the GUI event loop.

def hard_voting():
    loading_popup('hard voting')
    table_values = default_table_values()
    input_column_element = input_column()
    output_column_element = output_column(table_values)
    layout = [
        [
            input_column_element, 
            sg.VerticalSeparator(), 
            output_column_element,
        ]
    ]

    ensemble = load_ensemble('ensemble-hard.pkl')
    if not ensemble:
        return

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
            result, learner_predictions = predict_with_language_check(
                ensemble, 
                values['-INPUT-']
            )
            if result is None: continue
            table_values[0][1:] = learner_predictions[0][0]
            table_values[1][1:] = learner_predictions[1][0]
            table_values[2][1:] = learner_predictions[2][0]
            window['-ENSEMBLE-'].update(
                'Non-hate' if result[0] == 0 else 'Hate'
            )
            window['-TABLE-'].update(table_values)
        elif event == sg.WIN_CLOSED or event == 'Exit':
            break

    window.close()
    layout = None
    window = None
    gc.collect()

def soft_voting():
    loading_popup('soft voting')
    table_values = default_table_values()
    input_column_element = input_column()
    output_column_element = output_column(table_values)
    layout = [
        [
            input_column_element, 
            sg.VerticalSeparator(), 
            output_column_element,
        ]
    ]

    ensemble = load_ensemble('ensemble-soft.pkl')
    if not ensemble:
        return

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
            result, learner_predictions = predict_with_language_check(
                ensemble, 
                values['-INPUT-']
            )
            if result is None: continue
            table_values[0][1:] = learner_predictions[0][0]
            table_values[1][1:] = learner_predictions[1][0]
            table_values[2][1:] = learner_predictions[2][0]
            window['-ENSEMBLE-'].update(
                f"Non-hate - {result[0][0]} | Hate - {result[0][1]}"
            )
            window['-TABLE-'].update(table_values)
        elif event == sg.WIN_CLOSED or event == 'Exit':
            break

    window.close()
    layout = None
    window = None
    gc.collect()

def stacking():
    loading_popup('stacking')
    table_values = default_table_values()
    input_column_element = input_column()
    output_column_element = output_column(table_values)
    layout = [
        [
            input_column_element, 
            sg.VerticalSeparator(), 
            output_column_element,
        ]
    ]

    ensemble = load_ensemble('ensemble-stacking.pkl')
    if not ensemble:
        return

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
            result, learner_predictions = predict_with_language_check(
                ensemble, 
                values['-INPUT-']
            )
            if result is None: continue
            table_values[0][1:] = learner_predictions[0][0]
            table_values[1][1:] = learner_predictions[1][0]
            table_values[2][1:] = learner_predictions[2][0]
            window['-ENSEMBLE-'].update(
                f"Non-hate - {result[0][0]} | Hate - {result[0][1]}"
            )
            window['-TABLE-'].update(table_values)
        elif event == sg.WIN_CLOSED or event == 'Exit':
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

# Function to resize an image using PIL
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
    main_window()
