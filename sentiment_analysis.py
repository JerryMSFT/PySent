import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import gc

print(f"PyTorch version: {torch.__version__}")
torch.manual_seed(1)

# Use MPS for M1 Macs, CUDA for NVIDIA GPUs, or fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load a subset of the IMDB dataset to reduce memory usage
dataset = load_dataset("imdb", split="train[:25%]+test[:25%]")
train_dataset = dataset.shuffle(seed=42).select(range(10000))  # Use 10,000 samples for training
test_dataset = dataset.shuffle(seed=42).select(range(10000, 12500))  # Use 2,500 samples for testing

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Use DistilBERT, a lighter version of BERT
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

class IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):  # Reduced max_length
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(item['label'])
        }

train_dataset = IMDBDataset(train_dataset, tokenizer)
test_dataset = IMDBDataset(test_dataset, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size
test_dataloader = DataLoader(test_dataset, batch_size=16)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # Use AdamW optimizer
num_epochs = 3  # Reduced number of epochs

print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Clear GPU memory
        del input_ids, attention_mask, labels, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}')

print("Training complete. Evaluating on test set...")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Clear GPU memory
        del input_ids, attention_mask, labels, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

print(f'Accuracy on the test set: {correct/total:.4f}')

print("Entering interactive mode...")

def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        sentiment_score = probabilities[:, 1].item()
        
        # Clear GPU memory
        del inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return sentiment_score

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

