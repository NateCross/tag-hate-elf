import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from torch import nn, optim
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, LoadInitState
from sklearn.base import BaseEstimator, TransformerMixin
import calamancy

_device = device("cuda" if cuda.is_available() else "cpu")
# _device = "cpu"

Calamancy = calamancy.load("tl_calamancy_md-0.1.0")

class LstmModel(nn.Module):
    """
    TODO: Document each layer
    """
    def __init__(
            self, 
            input_size=200, 
            hidden_size=400, 
            output_size=2, 
            num_layers=2
        ):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size,
            num_layers, 
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.fc(lstm_out)

        return output

Criterion = nn.CrossEntropyLoss

Optimizer = optim.Adam

checkpoint = Checkpoint(
    monitor='train_loss_best',
    dirname='train_lstm',
)

load_state = LoadInitState(checkpoint)
"""
Checkpoint is used to save and load training progress
"""

class CalamancyTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, _, y=None):
        return self

    def transform(self, data):
        if isinstance(data, pd.Series):
            data = data.values
        result = [
            text.vector.reshape(1, -1)
            for text 
            in Calamancy.pipe(data)
        ]
        result = np.concatenate(result)
        return result

LstmNet = NeuralNetClassifier(
    LstmModel,
    module__hidden_size=400,
    optimizer__lr=0.00001,
    optimizer__weight_decay=0.01,
    criterion=Criterion,
    optimizer=Optimizer,
    batch_size=100,
    device=_device,
    callbacks=[
        checkpoint, 
        load_state,
    ],
    train_split=None, # Fixes numpy.exceptions.AxisError in training
                    # Anyways, data is assumed to be already split
)

LstmPipeline = Pipeline([
    ('tokenizer', CalamancyTokenizer()),
    ('lstm', LstmNet),
])
