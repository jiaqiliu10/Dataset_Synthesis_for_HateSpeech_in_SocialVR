import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Load the Facebook hate speech detection model
model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    attn_implementation="eager"  # Explicitly use eager implementation to avoid warning
)

# Function to predict hate speech
def predict_hate_speech(text):
    # Handle invalid inputs
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # This model is binary: 0 = not hate, 1 = hate
    return probs[0][1].item()  # Probability of hate speech

# Load CSV data
df = pd.read_csv('../../Data/merged_all_conversations.csv')

# Apply the model to the messages
print("Calculating hate speech scores...")
df['hate_score'] = df['message'].apply(predict_hate_speech)

# Convert to binary prediction using threshold
threshold = 0.5 
df['predicted_label'] = (df['hate_score'] > threshold).astype(int)
df['matches_ground_truth'] = (df['predicted_label'] == df['label']).astype(int)

# Calculate accuracy and other metrics
accuracy = df['matches_ground_truth'].mean()
print(f"Accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(df['label'], df['predicted_label']))

print("\nConfusion Matrix:")
print(confusion_matrix(df['label'], df['predicted_label']))

# Save results
df.to_csv('facebook_roberta_evaluation_results.csv', index=False)
print(f"Results saved to facebook_roberta_evaluation_results.csv")