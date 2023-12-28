# 80% Training 20% Validation
# Bias from Majority
# Still needs fine tuning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV DATASET
data = pd.read_csv("./datasets/train.csv")

# Extract input text and labels
input_text = data["text"]
labels = data["label"]

# Calculate class weights for unbias
class_counts = data["label"].value_counts().to_dict()
total_samples = len(data)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

# For binary classification, we are interested in the weight for the positive class
pos_weight = class_weights[1]  # Assuming 1 is the positive class
pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)

# Tokenize and encode the data
# Still need to change tokenizer since this is using bert
# tokenizer = TfIdf
# tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

input_ids = []
attention_masks = []

for text in input_text:
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=255,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids.append(encoded["input_ids"])
    attention_masks.append(encoded["attention_mask"])

input_ids = torch.cat(input_ids, dim=0).to(device)
attention_masks = torch.cat(attention_masks, dim=0).to(device)

# Convert labels to tensor
labels = torch.tensor(labels.values, dtype=torch.float32).to(
    device
)  # Adjust dtype for BCEWithLogitsLoss

# Split the data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42
)

# Set hyperparameters
input_size = len(tokenizer.get_vocab())
hidden_size = 128
output_size = 1  # Binary classification (1 output for BCEWithLogitsLoss)
num_layers = 2
learning_rate = 0.0001
num_epochs = 1
batch_size = 32  # for batches in training the model

# Create DataLoader for training data
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for validation data
val_dataset = TensorDataset(val_inputs, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output.squeeze(1)  # Squeeze the output to a single dimension


# Initialize the model, loss function, and optimizer
lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Training loop using Batching for large dataset which in this case is 4.2k data
for epoch in range(num_epochs):
    lstm_model.train()
    for batch_start in tqdm(
        range(0, len(train_inputs), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"
    ):
        batch_end = min(batch_start + batch_size, len(train_inputs))
        inputs_batch = train_inputs[batch_start:batch_end].to(device)
        labels_batch = train_labels[batch_start:batch_end].to(device)

        optimizer.zero_grad()
        outputs = lstm_model(inputs_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    lstm_model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs_batch, labels_batch in val_loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = lstm_model(inputs_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()
    print(f"Validation loss: {val_loss / len(val_loader)}")

# Save the trained model
torch.save(
    lstm_model.state_dict(),
    "./lstm_model",
)
