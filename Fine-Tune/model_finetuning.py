import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import os
import json
from datetime import datetime
import seaborn as sns

# Import your data
from data_splitting import X_train, y_train, X_val, y_val, X_test, y_test

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create dataframes
train_df = pd.DataFrame({"text": X_train, "labels": y_train}).dropna()
val_df = pd.DataFrame({"text": X_val, "labels": y_val}).dropna()
test_df = pd.DataFrame({"text": X_test, "labels": y_test}).dropna()

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Load the tokenizer
# Choose Model
# Here we are using DistilBERT
# Alternative models:
# Roberta - facebook/roberta-hate-speech-dynabench-r4-target
# HateBert - GroNLP/hateBERT
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define hyperparameter grid
learning_rates = [1e-5, 2e-5, 5e-5]
batch_sizes = [16, 32]
max_epochs = 8
patience = 2
weight_decay = 0.01
max_grad_norm = 1.0

# Create a directory to save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"hyperparameter_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Tokenize function
def tokenize_dataset(dataset, tokenizer, max_length=512):
    return tokenizer(
        dataset["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="pt"
    )

# Tokenize datasets
tokenized_train = dataset["train"].map(
    lambda examples: tokenize_dataset(examples, tokenizer),
    batched=True
)
tokenized_val = dataset["validation"].map(
    lambda examples: tokenize_dataset(examples, tokenizer),
    batched=True
)
tokenized_test = dataset["test"].map(
    lambda examples: tokenize_dataset(examples, tokenizer),
    batched=True
)

# Set format to PyTorch tensors
columns_to_return = ["input_ids", "attention_mask", "labels"]
tokenized_train.set_format(type="torch", columns=columns_to_return)
tokenized_val.set_format(type="torch", columns=columns_to_return)
tokenized_test.set_format(type="torch", columns=columns_to_return)

# For storing results
results = []

# Track the best model
best_val_f1 = 0.0
best_params = None
best_model_state = None

# Loop through all hyperparameter combinations
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\n{'-'*50}")
        print(f"Training with learning rate: {lr}, batch size: {batch_size}")
        print(f"{'-'*50}")
        
        # Create data loaders with current batch size
        train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(tokenized_val, batch_size=batch_size)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # hate or non-hate
        ).to(device)
        
        # Initialize optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        num_training_steps = len(train_dataloader) * max_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        # For tracking metrics
        train_losses = []
        train_accs = []
        val_accs = []
        val_f1s = []
        no_improvement_count = 0
        current_best_val_f1 = 0.0
        
        # Training loop
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            total_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
            for batch in train_loop:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=-1)
                correct_train += (preds == batch["labels"]).sum().item()
                total_train += batch["labels"].size(0)
                
                train_loop.set_postfix(loss=loss.item())
                
            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accs.append(train_accuracy)
            
            # Validation phase
            model.eval()
            correct_val = 0
            total_val = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]")
                for batch in val_loop:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    
                    preds = torch.argmax(outputs.logits, dim=-1)
                    correct_val += (preds == batch["labels"]).sum().item()
                    total_val += batch["labels"].size(0)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(batch["labels"].cpu().numpy())
            
            val_accuracy = correct_val / total_val
            val_f1 = f1_score(val_true, val_preds, average='weighted')
            
            val_accs.append(val_accuracy)
            val_f1s.append(val_f1)
            
            print(f"Epoch {epoch+1}/{max_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Check if we have a new best model
            if val_f1 > current_best_val_f1:
                current_best_val_f1 = val_f1
                epoch_model_state = model.state_dict().copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping check
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Update best model if current model is better
        if current_best_val_f1 > best_val_f1:
            best_val_f1 = current_best_val_f1
            best_params = {'lr': lr, 'batch_size': batch_size}
            best_model_state = epoch_model_state
        
        # Save metrics for this hyperparameter set
        run_result = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs,
            'val_f1_scores': val_f1s,
            'best_val_f1': current_best_val_f1,
            'epochs_trained': len(train_losses),
        }
        
        results.append(run_result)
        
        # Plot and save learning curves for this run
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title(f'Training Loss (lr={lr}, batch={batch_size})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(val_f1s)
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'learning_curves_lr{lr}_bs{batch_size}.png'))
        plt.close()

# Save all results to a JSON file
with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
    # Convert numpy floats to Python floats for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, list) and isinstance(value[0], (np.float32, np.float64)):
                serializable_result[key] = [float(v) for v in value]
            elif isinstance(value, (np.float32, np.float64)):
                serializable_result[key] = float(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    json.dump(serializable_results, f, indent=2)

# Print results summary
print("\n=== Hyperparameter Tuning Results ===")
for result in results:
    print(f"Learning Rate: {result['learning_rate']}, Batch Size: {result['batch_size']}")
    print(f"  Best Val F1: {result['best_val_f1']:.4f}")
    print(f"  Epochs Trained: {result['epochs_trained']}")
    print()

print(f"\nBest Hyperparameters: Learning Rate = {best_params['lr']}, Batch Size = {best_params['batch_size']}")
print(f"Best Validation F1 Score: {best_val_f1:.4f}")

# Plot comparison of best F1 scores
plt.figure(figsize=(10, 6))
hyperparams = [f"lr={r['learning_rate']}\nbs={r['batch_size']}" for r in results]
best_f1s = [r['best_val_f1'] for r in results]

plt.bar(hyperparams, best_f1s)
plt.title('Best Validation F1 Score by Hyperparameter Combination')
plt.ylabel('F1 Score')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'hyperparameter_comparison.png'))

# Train final model with best hyperparameters
print("\n=== Training Final Model with Best Hyperparameters ===")

# Create data loaders with best batch size
train_dataloader = DataLoader(tokenized_train, batch_size=best_params['batch_size'], shuffle=True)
val_dataloader = DataLoader(tokenized_val, batch_size=best_params['batch_size'])
test_dataloader = DataLoader(tokenized_test, batch_size=best_params['batch_size'])

# Initialize model with best params
final_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
).to(device)

# Load best model state
final_model.load_state_dict(best_model_state)

# Evaluate on test set
final_model.eval()
test_preds = []
test_true = []

with torch.no_grad():
    test_loop = tqdm(test_dataloader, desc="Testing")
    for batch in test_loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = final_model(**batch)
        
        preds = torch.argmax(outputs.logits, dim=-1)
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(batch["labels"].cpu().numpy())

# Calculate test metrics
test_accuracy = accuracy_score(test_true, test_preds)
test_f1 = f1_score(test_true, test_preds, average='weighted')

print(f"\nTest Set Results:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  F1 Score: {test_f1:.4f}")

# Generate and save detailed classification report
report = classification_report(test_true, test_preds, target_names=['Non-Hate', 'Hate'], output_dict=True)
with open(os.path.join(results_dir, 'classification_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(test_true, test_preds, target_names=['Non-Hate', 'Hate']))

# Generate and save confusion matrix
cm = confusion_matrix(test_true, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))

# Save the best model
final_model_path = os.path.join(results_dir, "best_model")
final_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nBest model saved to {final_model_path}")

# Save a summary of results
summary = {
    "best_hyperparameters": {
        "learning_rate": float(best_params['lr']),
        "batch_size": best_params['batch_size']
    },
    "best_validation_f1": float(best_val_f1),
    "test_results": {
        "accuracy": float(test_accuracy),
        "f1_score": float(test_f1)
    }
}

with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to directory: {results_dir}")