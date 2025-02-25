import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict sentiment score (can be adapted for toxicity detection)
def predict_sentiment(text):
    # Handle NaN values or non-string inputs
    if pd.isna(text) or not isinstance(text, str):
        return 0.0  # Return default score for invalid inputs
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # For this model, index 1 typically represents positive sentiment
    # Return negative sentiment score (can be interpreted as toxicity proxy)
    return probs[0][0].item() 

# Load your CSV data
df = pd.read_csv('../Data/merged_hate_non_hate_speech.csv')

# Apply the model to the messages
print("Calculating scores...")
df['toxicity_score'] = df['message'].apply(predict_sentiment)

# Compare model predictions with your labels
# 'label' column 1 for toxic and 0 for non-toxic
threshold = 0.5
df['predicted_label'] = (df['toxicity_score'] > threshold).astype(int)
df['matches_ground_truth'] = (df['predicted_label'] == df['label']).astype(int)

# Calculate accuracy
accuracy = df['matches_ground_truth'].mean()
print(f"Accuracy: {accuracy:.4f}")

# Generate detailed evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(df['label'], df['predicted_label']))

print("\nConfusion Matrix:")
print(confusion_matrix(df['label'], df['predicted_label']))

# Save results to a new CSV
df.to_csv('distilbert_evaluation_results.csv', index=False)

print(f"Results saved to distilbert_evaluation_results.csv")