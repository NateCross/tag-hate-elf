from sklearn.pipeline import Pipeline
from transformers import BertForSequenceClassification
from torch import nn, device, cuda
from skorch import NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import Checkpoint

_device = "cpu"
# _device = device("cuda" if cuda.is_available() else "cpu")

_model_name = "bert-base-multilingual-cased"

_model = BertForSequenceClassification.from_pretrained(
    _model_name,
    device_map=_device,
)
_model.to(_device)

class BertModel(nn.Module):
    """
    TODO: Document each layer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = _model

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)
        return x.logits

BertTokenizer = HuggingfacePretrainedTokenizer('bert-base-multilingual-cased')

Criterion = nn.CrossEntropyLoss

_checkpoint = Checkpoint(
    monitor='train_loss_best',
    f_params='bert_train.pt',
)
"""
Checkpoint is used to save and load training progress
"""

BertNet = NeuralNetClassifier(
    BertModel,
    criterion=Criterion,
    batch_size=10,
    device=_device,
    callbacks=[_checkpoint],
    train_split=None, # Fixes numpy.exceptions.AxisError in training
                      # Anyways, data is assumed to be already split
)

BertPipeline = Pipeline([
    ('tokenizer', BertTokenizer),
    ('bert', BertNet),
])

BertPipeline.set_params(
    tokenizer__max_length=255,
    tokenizer__return_attention_mask=True,
    tokenizer__return_tensors="pt",
)
