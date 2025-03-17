import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Select model architecture
MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"  
# "distilbert-base-uncased-finetuned-sst-2-english" 
# "facebook/roberta-hate-speech-dynabench-r4-target"
# "GroNLP/hateBERT" 

# Load data
data = pd.read_csv('/home/liu.jiaqi10/CS7980/merged_all_conversations.csv')
# data = pd.read_csv("../Data/merged_all_conversations.csv")
data.columns = data.columns.str.strip()

# Ensure the 'message' column exists
if "message" not in data.columns:
    raise ValueError(f"Column name error, current column names: {data.columns}")

# Ensure the 'message' column is of string type
data["message"] = data["message"].astype(str)

# Split data: 80% training, 10% validation, 10% test
train_messages, test_messages, train_labels, test_labels = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)
val_messages, test_messages, val_labels, test_labels = train_test_split(
    test_messages, test_labels, test_size=0.5, random_state=42
)

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame({'message': train_messages, 'label': train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'message': val_messages, 'label': val_labels}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'message': test_messages, 'label': test_labels}))

# Tokenization function
def tokenize_function(batch):
    return tokenizer(batch["message"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
# train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set data format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Compute evaluation metrics (including F1-score & Confusion Matrix)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)
    plot_confusion_matrix(cm)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Plot Confusion Matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-hate", "Hate"], yticklabels=["Non-hate", "Hate"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Training parameters (including early stopping & gradient clipping)
training_args = TrainingArguments(
    output_dir=f"./{MODEL_NAME}_finetuned",
    per_device_train_batch_size=16,  # Choose 16 or 32
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # Choose 1e-5, 2e-5, 5e-5
    num_train_epochs=3,
    weight_decay=0.01,  # AdamW weight decay
    max_grad_norm=1.0,  # Gradient Clipping
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Key: Select the best model based on F1-score
    greater_is_better=True
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping patience=2
)

trainer.train()

# Evaluate the test set
test_results = trainer.evaluate(test_dataset)
print("Test set evaluation:", test_results)

# Get test set predictions
predictions = trainer.predict(test_dataset)
labels = predictions.label_ids
preds = predictions.predictions.argmax(-1)

# Compute Confusion Matrix
cm = confusion_matrix(labels, preds)
plot_confusion_matrix(cm)

# Save the model
model.save_pretrained(f'./{MODEL_NAME}_finetuned')
tokenizer.save_pretrained(f'./{MODEL_NAME}_finetuned')
