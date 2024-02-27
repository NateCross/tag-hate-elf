import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

from sklearn.ensemble import VotingClassifier
import joblib

if __name__ == "__main__":
    ensemble = joblib.load('ensemble.pkl')
    quotes = [
        "Gago ka putang ina", 
        'OIDAjodiajisjdai', 
        "You are a fucking bitch", 
        "NAKO  NAHIYA  YUNG  KAPAL  NG  PERA  NI  BINAY ", 
        "fuck you binay gago ka",
    ]
    for estimator in ensemble.estimators_:
        print(estimator.predict_proba(quotes))
    if isinstance(ensemble, VotingClassifier) and ensemble.voting == 'hard':
        print(ensemble.predict(quotes))
    else:
        print(ensemble.predict_proba(quotes))

    exit(0)
