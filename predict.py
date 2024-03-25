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
    from sklearn.utils.extmath import safe_sparse_dot
    print("Bayes")
    model = joblib.load('model_bayes/Bayes.pkl')
    bayes = model['bayes']

    # print(np.log(bayes.feature_count_ + 0.3) - np.log((bayes.class_count_ + 0.3 * 2).reshape(-1, 1)))
    # print(np.log(bayes.feature_count_ + 0.3) - np.log((bayes.class_count_ + 0.3 * 2).reshape(-1, 1)))
    # print(bayes.classes_)

    # class_count = model['bayes'].class_count_
    # print(np.log(class_count) - np.log(class_count.sum()))
    # print(np.exp(bayes.class_log_prior_))

    # exit()

    print(np.log(np.exp(model['bayes'].class_log_prior_)))
    print(model['bayes'].feature_log_prob_)
    print(np.exp(model['bayes'].feature_log_prob_[0][22312]))
    feature_prob = model['bayes'].feature_log_prob_
    print("neg prob")
    neg = np.log(1 - np.exp(model['bayes'].feature_log_prob_))
    # print(neg)
    print((feature_prob - neg).T[12789])
    inverse = (feature_prob-neg).T
    print("inverse")
    result = inverse[10957] + inverse[15550] + inverse[22312] + inverse[30527]
    print(np.exp(result))
    print("clp")
    prior = model['bayes'].class_log_prior_
    print(prior)
    print('result')
    final = neg.sum(axis=1) + result + prior
    print(final)
    final_max = final.max()
    print('logsumexp')
    print(np.exp(final - final_max))
    print(np.sum(np.exp(final - final_max)))
    print(np.log(np.sum(np.exp(final - final_max))))
    logsumexp_res = final_max + (np.log(np.sum(np.exp(final - final_max))))
    print(np.exp(final - logsumexp_res))
    # print(inverse[10957] * inverse[15550] * inverse[22312] * inverse[30527])

    preds = model.predict_proba(["Hi", "Gago gago ka putangina mo mo", 'a'])
    print(preds)
    y_result = [0, 0, 0]

    # result = accuracy_score(y_result, preds, sample_weight=[0.9, 0.1, 0.9])

    # print(result)



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
    model = joblib.load('model_lr/LR.pkl')
    preds = model.predict_proba([[0.01, 0.33, 0.2]])
    print(preds)
    exit()

def test_bert():
    # BertModel = 
    torch.cuda.is_available = lambda: False
    model = joblib.load('model_bert/mBERT.pkl')
    print(model['bert'].module_.named_parameters())
    # print(model['bert'].module.parameters())
    exit()
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
    # test_bayes()
    test_bert()

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
