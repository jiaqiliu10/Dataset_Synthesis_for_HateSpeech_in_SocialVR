from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned HateBERT model and tokenizer
model_name = "/home/liu.jiaqi10/CS7980/fine_tune/hatebert_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True).to(device)
# Load the dataset for testing
data = pd.read_csv('/home/liu.jiaqi10/CS7980/fine_tune/fine_tune_dataset.csv')

# Tokenize the text
inputs = tokenizer(list(data['text']), padding=True, truncation=True, return_tensors="pt").to(device)

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# Add predictions to the dataframe
data['predicted_label'] = predictions

# Save the results to a new CSV file
data.to_csv('/home/liu.jiaqi10/CS7980/fine_tune/hate_speech_classified_results_ft.csv', index=False)

print("Classification completed. Results saved to '/home/liu.jiaqi10/CS7980/fine_tune/hate_speech_classified_results_ft.csv'")
