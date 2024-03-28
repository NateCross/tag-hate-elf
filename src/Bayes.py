from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
import sklearn

text.tokenized_text = []

def patched_analyze(
    doc,
    analyzer=None,
    tokenizer=None,
    ngrams=None,
    preprocessor=None,
    decoder=None,
    stop_words=None,
):
    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if preprocessor is not None:
            doc = preprocessor(doc)
        if tokenizer is not None:
            doc = tokenizer(doc)
            text.tokenized_text.append(doc)
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)
            else:
                doc = ngrams(doc)
    return doc

text._analyze = patched_analyze

def get_tokenized_text(self):
    return text.tokenized_text

text.CountVectorizer.get_tokenized_text = get_tokenized_text

# BayesModel = BernoulliNB(
#     alpha=0.3,
# )
# """
# Bernoulli Naive Bayes estimator from scikit-learn
# """

# Vectorizer = CountVectorizer()
# """
# Getting count of text and passing to bayes
# """

# BayesPipeline = Pipeline([
#     ('tfidf', Vectorizer),
#     ('bayes', BayesModel),
# ])
# """
# Create a pipeline to handle the Bernoulli Naive Bayes (BNB)
# functions. First, it uses the count vectorizer to
# transform text into features that can be passed to the
# BNB estimator. The BNB estimator then provides the result
# """
