from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import realpath, dirname

BayesModel = BernoulliNB()
"""
Bernoulli Naive Bayes estimator from scikit-learn
"""

STOP_WORDS = open(
    f"{dirname(__file__)}/stopwords-tl.txt", 
    "r",
).read().split('\n')
"""
Import stop words for further fine tuning
with the vectorizer
These are frequent Tagalog words which do not contribute
to the text, and as such, can be omitted

To properly account for relative file location, we get the
directory of this script, Bayes.py, and find stopwords-tl.txt
This is because scripts that import this from the root directory
will fail to find the stopwords file

List taken from:
https://github.com/stopwords-iso/stopwords-tl/blob/master/stopwords-tl.txt
"""

Vectorizer = TfidfVectorizer(
    stop_words=STOP_WORDS,
)
"""
We need to transform text into numerical values
This is one of the ways to do it
"""

BayesPipeline = Pipeline([
    ('tfidf', Vectorizer),
    ('bayes', BayesModel),
])
"""
Create a pipeline to handle the Bernoulli Naive Bayes (BNB)
functions. First, it uses the TF-IDF vectorizer to
transform text into features that can be passed to the
BNB estimator. The BNB estimator then provides the result
"""
