import torch
from torch import nn
from sklearn.ensemble import VotingClassifier
from skorch import NeuralNetClassifier

# Importing naive bayes

with open('./naive-bayes.pickle', 'rb') as bayes:
    import pickle
    bayes_model = pickle.load(bayes)

# Making lstm and skorch first

torch.manual_seed(0)

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

lstm_model = torch.load("./lstm_model.pt")
lstm_model.eval()

lstm_net = NeuralNetClassifier(
    lstm_model,
    device=device,
)

ensemble = VotingClassifier(
    estimators=[
        ('nb', bayes_model),
        ('lstm', lstm_net),
    ],
    voting='hard',
)
