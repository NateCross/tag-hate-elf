import torch
from torch import nn
import torch.optim as optim
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from skorch import NeuralNetBinaryClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
import numpy as np
import pandas as pd

# Import and refit data to work

train_csv = pd.read_csv("./datasets/testfor100.csv")
x_train = np.array([row[0] for row in train_csv.values])
y_train = np.array([row[1] for row in train_csv.values])

# x_train = x_train.reshape(-1, 1)

# Importing naive bayes

# Stop words for vectorizer
stop_words = open("./stopwords-tl.txt", "r").read().split('\n')

with open('./naive-bayes.pickle', 'rb') as bayes:
    import pickle
    bayes_model = pickle.load(bayes)
bayes_pipeline = Pipeline([
    (
        'tfidf',
        TfidfVectorizer(stop_words=stop_words),
    ),
    (
        'bayes',
        bayes_model,
    ),
])
print('Loaded bernoulli naive bayes')

# Seed pytorch and numpy
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, **kwargs):
        print("input ids:")
        print(input_ids)
        input_ids = self.embedding(input_ids)
        # print('finish embedding')
        lstm_out, _ = self.lstm(input_ids)
        # print('finish lstm')
        lstm_out = lstm_out[:, -1, :]
        # print('finish lstm slicing')
        output = self.fc(lstm_out)
        print('output:')
        print(output.shape)
        # print('finish output')

        return output.squeeze(1)  # Squeeze the output to a single dimension


# Initialize the model, loss function, and optimizer
# _ = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

bert_tokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')
print('Loaded tokenizer')

lstm_model = torch.load("./lstm_model.pt")
lstm_model.eval()
criterion = nn.BCEWithLogitsLoss
learning_rate = 0.000001
optimizer = optim.Adam
lstm_net = NeuralNetBinaryClassifier(
    lstm_model,
    criterion=criterion,
    optimizer=optimizer,
    batch_size=10,
    device=device,
)
lstm_pipeline = Pipeline([
    (
        'tokenizer',
        bert_tokenizer,
    ),
    (
        'lstm',
        lstm_net,
    ),
])
print('Loaded lstm')

bert_model = torch.load('./mbert_model.pt')
bert_model.eval()
bert_net = NeuralNetBinaryClassifier(
    bert_model,
    batch_size=10,
    device=device,
)
bert_pipeline = Pipeline([
    (
        'tokenizer',
        bert_tokenizer,
    ),
    (
        'bert',
        bert_net,
    ),
])
print('Loaded mbert')
print(bert_pipeline)

ensemble = VotingClassifier(
    estimators=[
        ('nb', bayes_pipeline),
        # ('bert', bert_pipeline),
        ('lstm', lstm_pipeline),
    ],
    voting='hard',
)
print(ensemble)

print('before fit')
ensemble.fit(x_train, y_train)
print('after fit')

print(ensemble.predict("gago ka putangina bakla"))
