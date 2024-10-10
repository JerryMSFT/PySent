import os
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '1'


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

print(f"PyTorch version: {torch.__version__}")
print(f"torchtext version: {torch.__version__}")

print("Starting to load the IMDB dataset...")

# Set random seed for reproducibility
torch.manual_seed(1)

# Define the LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[-1]
        output = self.fc(lstm_out)
        return output

# Fields
TEXT = Field(tokenize='basic_english', lower=True)
LABEL = Field(sequential=False, use_vocab=True, is_target=True)

# Load IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

print("Building vocabulary...")
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

# Create iterators
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=16, 
    device=device)

# Model parameters
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = 2
NUM_EPOCHS = 5

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Using device: {device}")

# Initialize the model
model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("Model initialized. Starting training...")

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_iterator:
        optimizer.zero_grad()
        text, labels = batch.text, batch.label
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_iterator):.4f}')

print("Training complete. Entering interactive mode...")

# Function to predict sentiment
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        text_tensor = torch.tensor(TEXT.process([text])).to(device)
        output = model(text_tensor)
        predicted = torch.argmax(output, dim=-1)
        sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
    return sentiment

# Interactive loop
print("\nYou can now analyze sentiments!")
print("Enter a sentence to analyze its sentiment (or type 'quit' to exit):")

while True:
    user_input = input("Enter text: ").strip()
    if user_input.lower() == 'quit':
        break

    if user_input:
        sentiment = predict_sentiment(user_input)
        print(f"Sentiment: {sentiment}")
    else:
        print("Please enter a non-empty text.")

print("Thank you for using the sentiment analyzer!")
