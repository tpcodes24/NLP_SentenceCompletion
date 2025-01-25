import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv("/content/drive/MyDrive/MachineLearning/Dataset.csv")


data.head()

from huggingface_hub import login

# Use the token in your script
login(token, add_to_git_credential=True)

import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from torch.optim import Adam, Adamax  # Correct import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/MachineLearning/Dataset.csv')  
data.fillna("", inplace=True)
label_encoder = LabelEncoder()
data['model_label'] = label_encoder.fit_transform(data['model_used'])

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create datasets
train_dataset = TextDataset(train_data['xj'].values, train_data['model_label'].values, tokenizer, max_length=128)
test_dataset = TextDataset(test_data['xj'].values, test_data['model_label'].values, tokenizer, max_length=128)

# Hyperparameters to experiment with
batch_sizes = [16, 32]  # Updated batch sizes
learning_rates = [1e-5, 2e-5, 5e-5]
optimizers = ['AdamW', 'Adam', 'Adamax']  # Added Adamax
schedulers = ['cosine', 'linear']
epochs = 3

# Function to train and evaluate the model
def train_and_evaluate(batch_size, learning_rate, optimizer_type, scheduler_type):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))
    optimizer = AdamW(model.parameters(), lr=learning_rate) if optimizer_type == 'AdamW' else Adam(model.parameters(), lr=learning_rate) if optimizer_type == 'Adam' else Adamax(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training
    model.train()
    total_loss = []
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss.append(avg_epoch_loss)
        print(f'Epoch: {epoch + 1}, Loss: {avg_epoch_loss}')

    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(f"Results for batch_size={batch_size}, learning_rate={learning_rate}, optimizer={optimizer_type}, scheduler={scheduler_type}")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

    # Plotting loss
    plt.plot(total_loss)
    plt.title(f'Training Loss: Batch Size {batch_size}, LR {learning_rate}, Optimizer {optimizer_type}, Scheduler {scheduler_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

# Running experiments
for batch_size, learning_rate, optimizer_type, scheduler_type in itertools.product(batch_sizes, learning_rates, optimizers, schedulers):
    train_and_evaluate(batch_size, learning_rate, optimizer_type, scheduler_type)