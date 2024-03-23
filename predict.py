import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

from sklearn.ensemble import VotingClassifier
import joblib
import numpy as np
from src import BERT
import torch
from sklearn.metrics import accuracy_score

BertModel = BERT.BertModel

def test_bayes():
    print("Bayes")
    model = joblib.load('model_bayes/Bayes.pkl')

    preds = model.predict(["Hi", "Gago ka putangina mo", 'a'])
    y_result = [0, 0, 0]

    result = accuracy_score(y_result, preds, sample_weight=[0.9, 0.1, 0.9])

    print(result)



    # print(model['bayes'].classes_)
    #
    # neg_prob_before_log = 1 - np.exp(model['bayes'].feature_log_prob_)
    #
    # after_log = np.log(neg_prob_before_log)
    #
    # print(neg_prob_before_log)
    # print(neg_prob_before_log.shape)
    #
    # print(model['bayes'].feature_log_prob_ - after_log)

    exit()

def test_lr():
    model = joblib.load('model_lr/LR-v1.pkl')
    preds = model.predict_proba([[0.01, 0.33, 0.2]])
    print(preds)
    exit()

def test_bert():
    torch.cuda.is_available = lambda: False
    model = joblib.load('model_bert/mBERT.pkl')
    quotes = [
        "Gago gago gago gago ka putang ina", 
        "Gago ka putang ina", 
        "ka gago putang ina", 
        'OIDAjodiajisjdai', 
        "You are a fucking bitch", 
        "NAKO  NAHIYA  YUNG  KAPAL  NG  PERA  NI  BINAY ", 
        "fuck you binay gago ka",
    ]
    print(model.predict_proba(quotes))
    exit()




if __name__ == "__main__":
    # test_lr()
    test_bayes()
    # test_bert()

    model = joblib.load('model_bayes/Bayes.pkl')
    print(model['tfidf'].vocabulary_['00'])
    print(f'feature count: {model["bayes"].feature_count_.shape}')
    print(f'feature count: {len(model["tfidf"].vocabulary_)}')
    print(f'feature log prob: {model["bayes"].feature_log_prob_}')
    print(f'feature count * alpha: {model["bayes"].feature_count_ * model["bayes"].alpha}')
    print(f'class count: {model["bayes"].class_count_.reshape(2, 1)}')
    print(model['tfidf'].vocabulary_['00'])
    # print(ensemble.coef_)
    # print(ensemble.intercept_)
    quotes = [
        "Gago gago gago gago ka putang ina", 
        "ka gago putang ina", 
        'OIDAjodiajisjdai', 
        "You are a fucking bitch", 
        "NAKO  NAHIYA  YUNG  KAPAL  NG  PERA  NI  BINAY ", 
        "fuck you binay gago ka",
    ]
    print(model.predict_proba(quotes))
    # for estimator in ensemble.estimators_:
    #     print(estimator.predict_proba(quotes))
    # if isinstance(ensemble, VotingClassifier) and ensemble.voting == 'hard':
    #     print(ensemble.predict(quotes))
    # else:
    #     print(ensemble.predict_proba(quotes))

    exit(0)
