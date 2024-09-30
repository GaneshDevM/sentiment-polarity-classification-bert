import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load Dataset and Split
# Load positive and negative reviews from text files
positive_reviews = open('rt-polarity.pos', 'r').readlines()
negative_reviews = open('rt-polarity.neg', 'r').readlines()

# Train/Val/Test Split
# Split the data into training, validation, and test sets
train_pos, val_pos, test_pos = positive_reviews[:4000], positive_reviews[4000:4500], positive_reviews[4500:]
train_neg, val_neg, test_neg = negative_reviews[:4000], negative_reviews[4000:4500], negative_reviews[4500:]

# Create a combined dataset of texts and labels
train_data = [(text, 1) for text in train_pos] + [(text, 0) for text in train_neg]
val_data = [(text, 1) for text in val_pos] + [(text, 0) for text in val_neg]
test_data = [(text, 1) for text in test_pos] + [(text, 0) for text in test_neg]

# Custom Dataset Class
class SentimentDataset(Dataset):
    # Initialize the dataset
    def __init__(self, data, tokenizer, max_len):
        self.data = data  # Store the data
        self.tokenizer = tokenizer  # Store the tokenizer
        self.max_len = max_len  # Maximum length for padding/truncating

    def __len__(self):
        return len(self.data)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        # Get the text and label for the sample at the given index
        text, label = self.data[idx]
        # Tokenize and encode the text
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add special tokens for BERT
            max_length=self.max_len,  # Set max length for padding/truncating
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate longer texts
            return_tensors="pt"  # Return as PyTorch tensors
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),  # Flatten the input IDs
            'attention_mask': inputs['attention_mask'].flatten(),  # Flatten the attention mask
            'label': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }

# Hyperparameters
MAX_LEN = 128  # Maximum sequence length
BATCH_SIZE = 16  # Batch size for training
EPOCHS = 3  # Number of training epochs
LR = 2e-5  # Learning rate

# Initialize BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Load the BERT tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Load the BERT model for classification

# DataLoaders
# Create datasets and dataloaders for training, validation, and test sets
train_dataset = SentimentDataset(train_data, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_data, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Shuffle training data
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)  # Validation data loader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  # Test data loader

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR)  # AdamW optimizer for weight decay

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
model = model.to(device)  # Move model to the appropriate device

def train_epoch(model, data_loader, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss for this epoch
    for batch in data_loader:
        optimizer.zero_grad()  # Clear previous gradients
        # Move inputs to the specified device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Get the loss from the outputs
        total_loss += loss.item()  # Accumulate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    return total_loss / len(data_loader)  # Return average loss for the epoch

def eval_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    preds, true_labels = [], []  # Initialize lists for predictions and true labels
    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:
            # Move inputs to the specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())  # Get predicted classes
            true_labels.extend(labels.cpu().numpy())  # Get true labels

    # Return classification report and confusion matrix
    return classification_report(true_labels, preds), confusion_matrix(true_labels, preds)

# Training & Validation
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, device)  # Train for one epoch
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss}")  # Print loss

# Evaluate on Test Data
test_report, test_cm = eval_model(model, test_loader, device)  # Evaluate on test set
print("Test Report:\n", test_report)  # Print classification report
print("Confusion Matrix:\n", test_cm)  # Print confusion matrix
