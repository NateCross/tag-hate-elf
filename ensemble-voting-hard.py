import torch
from torch import nn
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
import numpy as np
import pandas as pd

# Importing naive bayes

with open('./naive-bayes.pickle', 'rb') as bayes:
    import pickle
    bayes_model = pickle.load(bayes)
print('Loaded bernoulli naive bayes')

# Seed pytorch and numpy
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output.squeeze(1)  # Squeeze the output to a single dimension


# Initialize the model, loss function, and optimizer
# _ = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

bert_tokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')
print('Loaded tokenizer')

lstm_model = torch.load("./lstm_model.pt")
lstm_model.eval()
lstm_net = NeuralNetClassifier(
    lstm_model,
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
bert_net = NeuralNetClassifier(
    bert_model,
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
        ('nb', bayes_model),
        ('bert', bert_pipeline),
        ('lstm', lstm_pipeline),
    ],
    voting='hard',
)
print(ensemble)

train_csv = pd.read_csv("./datasets/output.csv")
x_train = np.array([row[0] for row in train_csv.values])
y_train = np.array([row[1] for row in train_csv.values])

x_train = x_train.reshape(-1, 1)

ensemble.fit(x_train, y_train)

print(ensemble.predict("gago ka putangina bakla"))
