import torch
from transformers import BertTokenizer
from lstm_[name] import LSTMModel


# Define model parameters
input_size = 256  # Adjust based on your model architecture
hidden_size = 128  # Adjust based on your model architecture
output_size = 2  # Adjust based on your problem
num_layers = 2

# Load the trained LSTM model
lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
lstm_model.load_state_dict(
    torch.load("C:/Users/Sam/Desktop/Hate Speech Detection/lstm_model")
)

# Set the model to evaluation mode
lstm_model.eval()

# Tokenizer for input text
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Function to predict hate speech
def predict_hate_speech(input_text):
    # Tokenize and encode the input text
    encoded = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        padding="max_length",
        max_length=255,  # Adjust as needed
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Forward pass through the model
    with torch.no_grad():
        output = lstm_model(encoded["input_ids"])

    # Get the predicted label
    predicted_label = torch.argmax(output, dim=1).item()

    return predicted_label


# User input
user_input = input("Enter a phrase: ")

# Get the prediction
prediction = predict_hate_speech(user_input)

# Display the result
if prediction == 1:
    print("Hate Speech Detected!")
else:
    print("Non-Hate Speech Detected!")
