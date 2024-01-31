from . import Bayes, LSTM, BERT
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

_estimators = [
    ('nb', Bayes.BayesPipeline),
    ('lstm', LSTM.LstmPipeline),
    ('bert', BERT.BertPipeline),
]

_regression = LogisticRegression()

HardVotingEnsemble = VotingClassifier(
    estimators=_estimators,
    voting='hard',
)

SoftVotingEnsemble = VotingClassifier(
    estimators=_estimators,
    voting='soft',
)

StackingEnsemble = StackingClassifier(
    estimators=_estimators,
    final_estimator=_regression,
    cv=3,   # 3 fold cross validation
)
