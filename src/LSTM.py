from sklearn.pipeline import Pipeline
from torch import nn, optim, device, cuda, tensor, relu
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import Checkpoint, LoadInitState
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import dirname
from transformers import BertForSequenceClassification

# _device = device("cuda" if cuda.is_available() else "cpu")
_device = "cpu"

# _model_name = "bert-base-multilingual-cased"

BertTokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')

class LstmModel(nn.Module):
    """
    TODO: Document each layer
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LstmModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            # bias=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_ids, **_):
        # input_ids = input_ids.clone().detach().to(_device).long()
        # print(input_ids)
        # input_ids = tensor(input_ids).to(_device).long()
        print("NEW")
        print(input_ids)

        batch_size = input_ids.size(0)
        hidden = self.init_hidden(batch_size)

        input_ids = self.embedding(input_ids)
        print(input_ids)
        lstm_out, _ = self.lstm(input_ids, hidden)
        print(lstm_out)

        lstm_out = relu(lstm_out)

        lstm_out = lstm_out[:, -1]
        # lstm_out = lstm_out[:, -1, :]
        print(lstm_out)
        # _, (final_hidden_state, __) = self.lstm(input_ids)
        output = self.fc(lstm_out)
        print(output)

        return output

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(_device),
        weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(_device))
        return hidden

STOP_WORDS = open(
    f"{dirname(__file__)}/stopwords-tl.txt", 
    "r",
).read().split('\n')

Vectorizer = TfidfVectorizer(
    stop_words=STOP_WORDS,
    # dtype=float,
)

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

        tokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')
        x_train_tokenizer = tokenizer.fit_transform(X_train)
        # print(len(tokenizer.vocabulary_))
        # print(x_train_tokenizer.vocabulary_)
        self.X_train = X_train
        self._input_size = len(tokenizer.vocabulary_)

        # vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
        # x_train_tfidf = vectorizer.fit_transform(X_train)
        #
        # self.X_train = X_train
        # self._input_size = x_train_tfidf.shape[1]


dataset = LstmData()

# Criterion = nn.L1Loss
# Criterion = nn.BCEWithLogitsLoss
Criterion = nn.CrossEntropyLoss
# Criterion = nn.BCELoss

Optimizer = optim.Adam

checkpoint = Checkpoint(
    monitor='train_loss_best',
    dirname='train_lstm',
    # f_params='lstm_train.pt',
)

load_state = LoadInitState(checkpoint)
"""
Checkpoint is used to save and load training progress
"""

class DataFormatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        print(len(data))
        X = data[0]
        y = data[1]
        print("X:")
        print(X)
        print("y:")
        print(y)
        return X

def LstmPipeline():
    LstmNet = NeuralNetClassifier(
        LstmModel,
        ### TODO: Revise LSTM and the options here
        # module__input_size=len(BertTokenizer.vocabulary_),
        # module__input_size=517,
        module__input_size=dataset._input_size,
        module__hidden_size=128,
        module__output_size=2,
        module__num_layers=2,
        optimizer__lr=0.00001,
        optimizer__weight_decay=0.01,
        criterion=Criterion,
        # criterion__reduction='none',
        optimizer=Optimizer,
        batch_size=10,
        device=_device,
        callbacks=[
            checkpoint, 
            load_state,
        ],
        train_split=None, # Fixes numpy.exceptions.AxisError in training
                        # Anyways, data is assumed to be already split
    )

    LstmPipeline = Pipeline([
        ('tokenizer', BertTokenizer),
        # ('tokenizer', Vectorizer),
        # ('format', DataFormatTransformer()),
        ('lstm', LstmNet),
    ])

    return LstmPipeline
