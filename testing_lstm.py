import torch
from transformers import BertTokenizer
from lstm_trainor import LSTMModel

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Define model parameters from trainor
input_size = len(tokenizer.get_vocab())
hidden_size = 128
output_size = 1
num_layers = 2

# Load the trained LSTM model
lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
lstm_model.load_state_dict(
    torch.load("./lstm_model")
)

# Set the model to evaluation mode
lstm_model.eval()

# Pass the tokenizer from the trainor
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Function to predict hate speech and return probability score
def predict_hate_speech_with_prob(input_text):
    # Tokenize and encode the input text
    encoded = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        padding="max_length",
        max_length=255,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Forward pass through the model
    with torch.no_grad():
        output = lstm_model(encoded["input_ids"])

    # Apply sigmoid to get probabilities
    predicted_probabilities = torch.sigmoid(output)
    predicted_label = (predicted_probabilities > 0.75).float().item()

    return predicted_label, predicted_probabilities.item()


user_input = input("Enter a phrase: ")

# Get the prediction and probability
prediction, probability = predict_hate_speech_with_prob(user_input)


print(
    f"Prediction: {'Hate Speech Detected!' if prediction == 1 else 'Non-Hate Speech Detected!'}"
)
print(f"Probability Score: {probability}")
