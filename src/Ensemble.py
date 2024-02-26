from . import Bayes, LSTM, BERT
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def initialize_estimators():
    return [
        ('nb', Bayes.BayesPipeline),
        ('lstm', LSTM.LstmPipeline),
        ('bert', BERT.BertPipeline),
    ]

_regression = LogisticRegression()

def HardVotingEnsemble():
    return VotingClassifier(
        estimators=initialize_estimators(),
        voting='hard',
    )

def SoftVotingEnsemble():
    return VotingClassifier(
        estimators=initialize_estimators(),
        voting='soft',
    )

def StackingEnsemble():
    return StackingClassifier(
        estimators=initialize_estimators(),
        final_estimator=_regression,
        cv=3,   # 3 fold cross validation
    )

