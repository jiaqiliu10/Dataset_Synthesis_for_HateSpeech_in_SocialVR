# -*- coding: utf-8 -*-

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# check if GPU is available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
# Use HateBERT to classify the message column to determine if it contains hate speech
data = pd.read_csv('/home/liu.jiaqi10/CS7980/merged_all_conversations.csv')

# data = pd.read_csv('/home/liu.jiaqi10/CS7980/merged_hate_non_hate_speech.csv')
# data = pd.read_csv('/home/liu.jiaqi10/CS7980/merged_without_vr.csv')

# Load HateBERT model and tokenizer
model_name = "GroNLP/hateBERT"
# model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
# Convert the text into a format the model can understand, 
# and pass it through HateBERT for classification.

# Classify each message in the dataset
predictions = []
for message in data['message']:
    inputs = tokenizer(str(message), return_tensors='pt', truncation=True, padding=True).to(device)
    # inputs = tokenizer(message, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    predictions.append(prediction)
# The model outputs a result, and we use argmax to pick the most likely category—
# 0 for non-hate speech and 1 for hate speech.

# Add predictions to the dataset
data['predicted_label'] = predictions

# Save classification results to a new file
data.to_csv('merge_all_classified_results.csv', index=False)
print("Classification completed. Results saved to 'merge_all_classified_results.csv'")

# data.to_csv('hate_speech_classified_results.csv', index=False)
# print("Classification completed. Results saved to 'hate_speech_classified_results.csv'")

# data.to_csv('result_without_vr.csv', index=False)
# print("Classification completed. Results saved to 'result_without_vr.csv'")
