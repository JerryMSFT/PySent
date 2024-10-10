import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

print(f"PyTorch version: {torch.__version__}")
print(f"torchtext version: {torch.__version__}")

print("Starting to load the IMDB dataset...")

# Set random seed for reproducibility
torch.manual_seed(1)

# Define the LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return torch.sigmoid(output)

# Tokenizer and vocabulary
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Load IMDB dataset
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

print("Preparing data and building vocabulary...")

# Text pipeline
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

# Collate function for DataLoader
def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(float(_label) - 1)  # Convert labels to 0 and 1
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

# Model parameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 32
NUM_EPOCHS = 5

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model
model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

print("Model initialized. Starting training...")

# Training loop
train_iter = IMDB(split='train')
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for labels, text in train_dataloader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_dataloader):.4f}')

print("Training complete. Entering interactive mode...")

# Function to predict sentiment
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        text_tensor = torch.tensor(text_pipeline(text)).unsqueeze(0).to(device)
        output = model(text_tensor)
    return output.item()

# Interactive loop
print("\nYou can now analyze sentiments!")
print("Enter a sentence to analyze its sentiment (or type 'quit' to exit):")

while True:
    user_input = input("Enter text: ").strip()
    if user_input.lower() == 'quit':
        break
    
    if user_input:
        sentiment_score = predict_sentiment(user_input)
        sentiment = "Positive" if sentiment_score > 0.5 else "Negative"
        print(f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")
    else:
        print("Please enter a non-empty text.")

print("Thank you for using the sentiment analyzer!")
