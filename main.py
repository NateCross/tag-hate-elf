import warnings
warnings.filterwarnings('ignore')

import PySimpleGUI as sg
import joblib
from sklearn.ensemble import VotingClassifier
from PIL import Image, ImageTk
import io
from src import Utils
from langdetect import detect, LangDetectException

def load_ensemble(filename: str):
    try:
        ensemble = joblib.load(filename)
    except FileNotFoundError:
        sg.PopupError(
            f'ERROR: "{filename}" not found. Please train the ensemble first.'
        )
        return None
    return ensemble

def clean_text(text: str):
    text = Utils.unmark(text)
    text = Utils.remove_urls(text)
    text = Utils.remove_usernames(text)
    text = Utils.remove_emojis(text)
    text = Utils.remove_escape_sequences(text)
    text = text.rstrip()
    return text

def predict(ensemble, text: str):
    text = clean_text(text)
    learner_predictions = [
        estimator.predict_proba([text])
        for estimator in ensemble.estimators_
    ]
    if isinstance(ensemble, VotingClassifier) and ensemble.voting == 'hard':
        return ensemble.predict([text]), learner_predictions
    else:
        return ensemble.predict_proba([text]), learner_predictions

def loading_popup(ensemble_string: str):
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
    return [
        ['Bernoulli Naive Bayes', '-', '-'],
        ['LSTM', '-', '-'],
        ['mBERT', '-', '-'],
    ]

def input_column():
    return sg.Column([
        [sg.Text("Input text:")],
        [sg.Multiline(size=(None, 12), key='-INPUT-')],
        [
            sg.Exit(),
            sg.Push(),
            sg.Button('Predict'),
        ],
    ])

def output_column(table_values):
    return sg.Column([
        [sg.Table(
            values=table_values, 
            auto_size_columns=True, 
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
            result, learner_predictions = predict(
                ensemble, 
                values['-INPUT-']
            )
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
            result, learner_predictions = predict(
                ensemble, 
                values['-INPUT-']
            )
            table_values[0][1:] = learner_predictions[0][0]
            table_values[1][1:] = learner_predictions[1][0]
            table_values[2][1:] = learner_predictions[2][0]
            window['-ENSEMBLE-'].update(result[0])
            window['-TABLE-'].update(table_values)
        elif event == sg.WIN_CLOSED or event == 'Exit':
            break

    window.close()

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
            result, learner_predictions = predict(
                ensemble, 
                values['-INPUT-']
            )
            table_values[0][1:] = learner_predictions[0][0]
            table_values[1][1:] = learner_predictions[1][0]
            table_values[2][1:] = learner_predictions[2][0]
            window['-ENSEMBLE-'].update(result[0])
            window['-TABLE-'].update(table_values)
        elif event == sg.WIN_CLOSED or event == 'Exit':
            break

    window.close()

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
    image = Image.open(image_path)
    image = image.resize((width, height), Image.Resampling.LANCZOS)  # High-quality downsampling filter
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()

def main_window():
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

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event in ['-HARD_VOTING-', '-SOFT_VOTING-', '-STACKING-']:
            input_text = values['-INPUT-']
            if input_text.strip():  # Check if the input is not just whitespace
                try:
                    lang = detect(input_text)
                    if lang not in ['en', 'tl']:  # 'en' for English, 'tl' for Tagalog
                        sg.popup_error('Error: Input must be in English or Tagalog.', title='Language Error')
                        continue  # Skip further processing
                except LangDetectException:
                    sg.popup_error('Error: Language detection failed. Please ensure the input is not empty and try again.', title='Detection Error')
                    continue
                # Proceed with prediction and other logic if the input is in English or Tagalog
            else:
                sg.popup_error('Error: Please enter some text before predicting.', title='Input Error')
                
    window.close()


if __name__ == "__main__":
    main_window()