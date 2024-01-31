from sklearn.pipeline import Pipeline
from torch import nn, optim, device, cuda
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import Checkpoint

_device = device("cuda" if cuda.is_available() else "cpu")

class LstmModel(nn.Module):
    """
    TODO: Document each layer
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LstmModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, **_):
        input_ids = self.embedding(input_ids)
        lstm_out, _ = self.lstm(input_ids)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)

        return output

BertTokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')

Criterion = nn.L1Loss

Optimizer = optim.Adam

_checkpoint = Checkpoint(
    monitor='valid_acc_best',
    f_params='lstm_train.pt',
)
"""
Checkpoint is used to save and load training progress
"""

LstmNet = NeuralNetBinaryClassifier(
    LstmModel,
    ### TODO: Revise LSTM and the options here
    module__input_size=len(BertTokenizer.vocabulary_),
    module__hidden_size=128,
    module__output_size=1,
    module__num_layers=2,
    criterion=Criterion,
    optimizer=Optimizer,
    batch_size=10,
    device=_device,
    callbacks=[_checkpoint],
    train_split=None, # Fixes numpy.exceptions.AxisError in training
                      # Anyways, data is assumed to be already split
)

LstmPipeline = Pipeline([
    ('tokenizer', BertTokenizer),
    ('lstm', LstmNet),
])
