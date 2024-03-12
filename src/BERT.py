from sklearn.pipeline import Pipeline
from transformers import BertForSequenceClassification
from torch import nn, device, cuda, optim
from skorch import NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import Checkpoint, LoadInitState

# _device = "cpu"
_device = device("cuda" if cuda.is_available() else "cpu")
"""
Set the device used by the learner.
It automatically uses the GPU if it is available, else,
it will default to the CPU.
Using the GPU is preferred because it is faster,
and can handle greater quantities of data.
"""

_model_name = "bert-base-multilingual-cased"
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

BertTokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')
"""
Load the tokenizer for use in mBERT.
It is the tokenizer made specifically for use in mBERT,
and as such, should be utilized here to process text input.
"""

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

BertNet = NeuralNetClassifier(
    BertModel,
    criterion=Criterion,
    batch_size=25,
    optimizer=Optimizer,
    optimizer__lr=0.00001,
    # optimizer__weight_decay=0.01,
    device=_device,
    callbacks=[
        checkpoint, 
        load_state,
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
    tokenizer__return_attention_mask=True,
    tokenizer__return_tensors="pt",
)
"""
Setting parameters of the tokenizer in the mBERT pipeline
so that the output shape is processable by the ensemble.
"""
