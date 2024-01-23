import torch

from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained BERT model for sequence classification
model_name = "bert-base-multilingual-cased"
model = BertForSequenceClassification.from_pretrained(
    model_name,
    # device_map=device,
)
# model.to(device)

tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)

# tokenizer.to(device)

# Load your CSV data
data = pd.read_csv(
    "./datasets/test.csv"
)

# Extract input text and labels
input_text = data["text"]
labels = data["label"]

# Tokenize and encode the data
input_ids = []
attention_masks = []

for text in input_text:
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=255,  # Adjust as needed
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids.append(encoded["input_ids"])
    attention_masks.append(encoded["attention_mask"])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Convert labels to tensor
labels = torch.tensor(labels.tolist())

print('before train')
# Perform inference for text classification
with torch.no_grad():
    model.train()
    outputs = model(input_ids, attention_mask=attention_masks)
print('after train')

"""
Add Functions on what you want to do to the model Now

Example below is checking it's accuracy.
"""


# Get predicted probabilities and predicted labels
logits = outputs.logits
predicted_labels = torch.argmax(logits, dim=1)

# Calculate accuracy
accuracy = torch.sum(predicted_labels == labels).item() / len(labels)

print(f"Accuracy: {accuracy * 100:.2f}%")

torch.save(
    model,
    "./mbert_model.pt",
)
