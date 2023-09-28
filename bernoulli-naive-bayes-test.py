from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd

bayes = BernoulliNB()

train_csv = pd.read_csv("./datasets/train.csv")
test_csv = pd.read_csv("./datasets/test.csv")

x_train = np.array([row[0] for row in train_csv.values])
y_train = np.array([row[1] for row in train_csv.values])

# We need to transform text into numerical values
# This is one of the ways to do it. Needs investigation to improve
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train).toarray()

# Some values are nan because of improper encoding as a csv
# This is a temporary fix to that. It needs proper cleaning
# print(np.isnan(y_train))
y_train[np.isnan(y_train)] = 0

result = bayes.fit(x_train, y_train)

# Current result: 0.8206
print(result.score(x_train, y_train))
