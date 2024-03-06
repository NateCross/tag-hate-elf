import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from torch import nn, optim, device, cuda
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, LoadInitState
from sklearn.base import BaseEstimator, TransformerMixin
import calamancy

_device = device("cuda" if cuda.is_available() else "cpu")
# _device = "cpu"
"""
Set the device used by the learner.
It automatically uses the GPU if it is available, else,
it will default to the CPU.
Using the GPU is preferred because it is faster,
and can handle greater quantities of data.
"""

Calamancy = calamancy.load("tl_calamancy_md-0.1.0")
"""
Load the tokenizer and vectorizer for use in LSTM.
CalamanCy is a SpaCy-based NLP model trained on Tagalog.
As such, this made it suitable for use here as a solution for
tokenization and vectorization all in one go, allowing text
to be properly read by LSTM.
"""

class LstmModel(nn.Module):
    """
    The LSTM learner to be used in the ensemble.
    It is composed of an LSTM layer and a Linear layer.
    The LSTM layer applies the LSTM RNN to a given input,
    and the Linear layer performs a linear transformation
    on the hidden state outputs into a vector of the desired
    size.
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
            batch_first=True,   # Modifies LSTM output shape 
                                # to be better compatible
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.fc(lstm_out)

        return output

Criterion = nn.CrossEntropyLoss
"""
Loss function for multilabel classification. This is desired
so we get the right output shape to be uniform with the other
learners.
This was chosen over BCELoss because BCELoss does not have the
right output shape.
"""

Optimizer = optim.Adam
"""
Implements the Adam algorithm as the optimizer,
commonly used in text classification problems.
"""

checkpoint = Checkpoint(
    monitor='train_loss_best',
    dirname='train_lstm',
)
"""
Checkpoint is used to save and load training progress.
"""

load_state = LoadInitState(checkpoint)
"""
Create a callback that loads the checkpoint.
"""

class CalamancyTokenizer(BaseEstimator, TransformerMixin):
    """
    Implement the tokenizer as a custom scikit-learn
    estimator. This allows it to be used in a scikit-learn
    pipeline.
    """
    def __init__(self):
        pass

    def fit(self, _, y=None):
        return self

    def transform(self, data):
        # Allows it to work with both dataframes and
        # simple lists of strings.
        if isinstance(data, pd.Series):
            data = data.values

        # Pipe is a faster way of iterating through all the data.
        # We get the vector of the tokenized text and reshape them
        # to be the right output shape.
        result = [
            text.vector.reshape(1, -1)
            for text 
            in Calamancy.pipe(data)
        ]

        # Concatenate all of them to form tensors of the right
        # output shape.
        result = np.concatenate(result)

        return result

LstmNet = NeuralNetClassifier(
    LstmModel,
    module__hidden_size=400,
    optimizer__lr=0.015,
    optimizer__weight_decay=0.00001,
    max_epochs=50,
    criterion=Criterion,
    optimizer=Optimizer,
    batch_size=32,
    device=_device,
    callbacks=[
        checkpoint, 
        load_state,
    ],
    train_split=None, # Fixes numpy.exceptions.AxisError in training
                    # Anyways, data is assumed to be already split
)
"""
Define the LSTM neural network alongside parameters of it,
its optimizer, and its criterion.
"""

LstmPipeline = Pipeline([
    ('tokenizer', CalamancyTokenizer()),
    ('lstm', LstmNet),
])
"""
Pipeline for LSTM. Import this for the ensemble.
"""
