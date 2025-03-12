import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

<<<<<<< Updated upstream
# Load data
data = pd.read_csv('/home/liu.jiaqi10/CS7980/fine_tune/fine_tune_dataset.csv')
=======
# 加载数据
data = pd.read_csv('../Data/merged_all_conversations.csv')
>>>>>>> Stashed changes

# Split dataset: 80% train, 10% validation, 10% test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    test_texts, test_labels, test_size=0.5, random_state=42
)

# Load HateBERT tokenizer and model
model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create Hugging Face Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'text': test_texts, 'label': test_labels}))

# Tokenization function
def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format (Trainer requires PyTorch format)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./hatebert_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
<<<<<<< Updated upstream
    per_device_train_batch_size=16,  # Adjust batch size
    per_device_eval_batch_size=10,   # Evaluation batch size
    num_train_epochs=4,
=======
    per_device_train_batch_size=16,  # 调整 batch size
    per_device_eval_batch_size=10,   # 评估 batch size
    num_train_epochs=10,
>>>>>>> Stashed changes
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start fine-tuning
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("Test set evaluation:", test_results)

# Save the fine-tuned model
model.save_pretrained('./hatebert_finetuned')
tokenizer.save_pretrained('./hatebert_finetuned')

