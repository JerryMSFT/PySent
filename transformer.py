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

print(f"PyTorch version: {torch.__version__}")
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

class IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 5

print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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
        output = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(output.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f'Accuracy on the test set: {correct/total:.4f}')

print("Entering interactive mode...")

def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        sentiment_score = probabilities[:, 1].item()  # Probability of positive sentiment
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
