from sklearn.pipeline import Pipeline
from transformers import BertForSequenceClassification
from torch import nn, device, cuda, optim
from skorch import NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import Checkpoint, LoadInitState, ProgressBar

_device = device("cuda" if cuda.is_available() else "cpu")
"""
Set the device used by the learner.
It automatically uses the GPU if it is available, else,
it will default to the CPU.
Using the GPU is preferred because it is faster,
and can handle greater quantities of data.
"""

_model_name = "bert-base-multilingual-uncased"
"""
mBERT on Huggingface
"""

_model = BertForSequenceClassification.from_pretrained(
    _model_name,
    device_map=_device,
)
_model.to(_device)
"""
Make the mBERT model and map it to the device.
This automatically downloads it from Huggingface if it is
not already on the current system.
"""

class BertModel(nn.Module):
    """
    Custom Pytorch module for mBERT.
    This simply gets the output from mBERT and returns the
    logits, allowing it to properly classify inputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = _model

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)
        return x.logits

BertTokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-uncased')
"""
Load the tokenizer for use in mBERT.
It is the tokenizer made specifically for use in mBERT,
and as such, should be utilized here to process text input.
"""

Criterion = nn.CrossEntropyLoss
"""
Loss function for classification problems
"""

Optimizer = optim.Adam

checkpoint = Checkpoint(
    monitor='train_loss_best',
    dirname='train_bert',
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

BertNet = NeuralNetClassifier(
    BertModel,
    criterion=Criterion,
    batch_size=16,
    optimizer=Optimizer,
    optimizer__lr=5e-5,
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
Define the mBERT neural network alongside parameters of it,
its optimizer, and its criterion.
"""

BertPipeline = Pipeline([
    ('tokenizer', BertTokenizer),
    ('bert', BertNet),
])
"""
Pipeline for mBERT. Import this for the ensemble.
"""

BertPipeline.set_params(
    tokenizer__max_length=255,
)
"""
Setting parameters of the tokenizer in the mBERT pipeline
so that the output shape is processable by the ensemble.
"""
