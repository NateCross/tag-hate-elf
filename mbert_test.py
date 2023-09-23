import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load the pre-trained BERT model for sequence classification
model_name = "bert-base-multilingual-cased"
model = BertForSequenceClassification.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)

# Load your CSV data
data = pd.read_csv(
    "C:/Users/Sam/Desktop/Hate Speech Detection/datasets/test100.csv"
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

# Perform inference for text classification
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)

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
