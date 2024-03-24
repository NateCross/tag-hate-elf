import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from torch import nn, optim, device, cuda
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, LoadInitState, ProgressBar
from sklearn.base import BaseEstimator, TransformerMixin
import calamancy

_device = device("cuda" if cuda.is_available() else "cpu")
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
            hidden_size=300, 
            output_size=2, 
            num_layers=1,
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
Loss function for classification
"""

Optimizer = optim.Adam

checkpoint = Checkpoint(
    monitor='train_loss_best',
    dirname='model_lstm/train_lstm',
    load_best=True,
)
"""
Checkpoint is used to save and load training progress.
"""

load_state = LoadInitState(checkpoint)
"""
Create a callback that loads the checkpoint.
"""

progress_bar = ProgressBar()

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
        result = []
        for doc in Calamancy.pipe(data):
            tokens = [
                token 
                for token 
                in doc
            ]

            if not tokens:
                doc_vector = np.zeros((1, 200))
            else:
                doc_vector = np.mean(
                    [token.vector for token in tokens], 
                    axis=0
                ).reshape(1, -1)
            
            result.append(doc_vector)

        # Concatenate all of them to form tensors of the right
        # output shape.
        result = np.concatenate(result).astype('float32')

        return result

LstmNet = NeuralNetClassifier(
    LstmModel,
    optimizer__lr=0.02,
    max_epochs=30,
    criterion=Criterion,
    optimizer=Optimizer,
    batch_size=32,
    device=_device,
    callbacks=[
        checkpoint, 
        load_state,
        progress_bar,
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
