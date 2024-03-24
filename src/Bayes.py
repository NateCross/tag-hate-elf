from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from os.path import dirname

BayesModel = BernoulliNB(
    alpha=0.3,
)
"""
Bernoulli Naive Bayes estimator from scikit-learn
"""

Vectorizer = CountVectorizer()
"""
Getting count of text and passing to bayes
"""

BayesPipeline = Pipeline([
    ('tfidf', Vectorizer),
    ('bayes', BayesModel),
])
"""
Create a pipeline to handle the Bernoulli Naive Bayes (BNB)
functions. First, it uses the count vectorizer to
transform text into features that can be passed to the
BNB estimator. The BNB estimator then provides the result
"""
