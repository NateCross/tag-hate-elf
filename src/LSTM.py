from sklearn.pipeline import Pipeline
from torch import nn, optim, device, cuda
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import Checkpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import dirname

# _device = device("cuda" if cuda.is_available() else "cpu")
_device = "cpu"

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

STOP_WORDS = open(
    f"{dirname(__file__)}/stopwords-tl.txt", 
    "r",
).read().split('\n')

Vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)

class LstmData():
    """
    Make a class (and object) to hold the dataset to be used
    for LSTM and its training.
    This is because LSTM uses an embedding layer, wherein a
    required argument is the size of the dictionary, which can
    be obtained from the vectorizer, but only after fitting.
    The scikit-learn ensemble and pipelines do not account for this,
    so this workaround is done to essentially pass another variable
    to this script
    """

    def __init__(self, X_train = None):
        self.X_train = X_train
        self._input_size = None

    def change(self, X_train):
        """
        Execute this method to set the input size for LSTM.
        Must be run before training
        """
        vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
        x_train_tfidf = vectorizer.fit_transform(X_train)

        self.X_train = X_train
        self._input_size = x_train_tfidf.shape[1]


dataset = LstmData()

Criterion = nn.BCEWithLogitsLoss

Optimizer = optim.Adam

_checkpoint = Checkpoint(
    monitor='valid_acc_best',
    f_params='lstm_train.pt',
)
"""
Checkpoint is used to save and load training progress
"""

def LstmPipeline():
    LstmNet = NeuralNetBinaryClassifier(
        LstmModel,
        ### TODO: Revise LSTM and the options here
        module__input_size=dataset._input_size,
        # module__input_size=dataset._input_size,
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
        ('tokenizer', Vectorizer),
        ('lstm', LstmNet),
    ])

    return LstmPipeline
