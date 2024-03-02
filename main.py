import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

import PySimpleGUI as sg
import joblib
from sklearn.ensemble import VotingClassifier

def load_ensemble(filename: str):
    try:
        ensemble = joblib.load(filename)
    except FileNotFoundError:
        sg.PopupError(
            f'ERROR: "{filename}" not found. Please train a hard voting ensemble first.'
        )
        return None
    return ensemble

def predict(ensemble, text: str):
    learner_predictions = [
        estimator.predict_proba([text])
        for estimator in ensemble.estimators_
    ]
    if isinstance(ensemble, VotingClassifier) and ensemble.voting == 'hard':
        return ensemble.predict([text]), learner_predictions
    else:
        return ensemble.predict_proba([text]), learner_predictions

def hard_voting():
    # ensemble_window = sg.Window()
    sg.PopupNonBlocking(
        'Loading hard voting ensemble...',
        button_type=sg.POPUP_BUTTONS_NO_BUTTONS,
        modal=True,
        # We do auto close to keep the loading popup active
        # while the thread execution is blocked due to
        # loading the ensemble
        auto_close=True,
        auto_close_duration=1,
    )
    ensemble = load_ensemble('ensemble-hard.pkl')
    # window.start_thread()

    print(ensemble)

def soft_voting():
    pass

def stacking():
    pass

def event_loop(window: sg.Window):
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            break
        if event == 'Hard Voting':
            hard_voting()

def main_window():
    button_frame = sg.Frame(
        'Select an Ensemble below:',
        layout=[
            [           
                sg.Button(
                    'Hard Voting',
                ),
            ],
            [
                sg.Button(
                    'Soft Voting',
                ),
            ],
            [
                sg.Button(
                    'Stacking',
                ),
            ],
        ],
        element_justification='c',
        
    )

    layout = [
        [
            sg.Text(
                'TAG-HATE-ELF',
                justification='c',
            ),
        ],
        [button_frame],
        [
            sg.Exit(),
        ],
    ]

    # Create the Window
    window = sg.Window(
        'TAG-HATE-ELF', 
        layout,
        element_justification='c',
    )

    event_loop(window)

    window.close()


if __name__ == "__main__":
    main_window()
