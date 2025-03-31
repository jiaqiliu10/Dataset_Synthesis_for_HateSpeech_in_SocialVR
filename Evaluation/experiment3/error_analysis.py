import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"error_analysis_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

def get_predictions(model, dataloader):
    """Get model predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Making predictions"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    return all_preds, all_labels

def analyze_misclassifications(texts, true_labels, pred_labels, stratified_purpose=None):
    """Analyze misclassification patterns focusing on explicit vs implicit hate speech"""
    
    # Find misclassifications
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true != pred]
    
    # Separate false positives and false negatives
    false_positives = [i for i in misclassified_indices if true_labels[i] == 0 and pred_labels[i] == 1]
    false_negatives = [i for i in misclassified_indices if true_labels[i] == 1 and pred_labels[i] == 0]
    
    print(f"Total misclassifications: {len(misclassified_indices)}")
    print(f"False positives (non-hate classified as hate): {len(false_positives)}")
    print(f"False negatives (hate classified as non-hate): {len(false_negatives)}")
    
    # Create dataframes for misclassifications
    misclassified_df = pd.DataFrame({
        'text': [texts[i] for i in misclassified_indices],
        'true_label': [true_labels[i] for i in misclassified_indices],
        'pred_label': [pred_labels[i] for i in misclassified_indices],
        'error_type': ['false_positive' if true_labels[i] == 0 else 'false_negative' for i in misclassified_indices]
    })
    
    # Add stratified purpose for analyzing explicit vs implicit hate (0=non-hate, 2=explicit, 3=implicit)
    if stratified_purpose is not None:
        misclassified_df['stratified_purpose'] = [stratified_purpose[i] for i in misclassified_indices]
        misclassified_df['hate_type'] = misclassified_df['stratified_purpose'].map({
            0: 'non-hate',
            2: 'explicit', 
            3: 'implicit'
        })
        
        # Analyze error rates by hate type (focus on false negatives - missed hate speech)
        explicit_indices = [i for i, purpose in enumerate(stratified_purpose) if purpose == 2]
        implicit_indices = [i for i, purpose in enumerate(stratified_purpose) if purpose == 3]
        
        explicit_errors = [i for i in explicit_indices if true_labels[i] != pred_labels[i]]
        implicit_errors = [i for i in implicit_indices if true_labels[i] != pred_labels[i]]
        
        # Show detailed counts for each hate type
        explicit_total = len(explicit_indices)
        implicit_total = len(implicit_indices)
        print(f"\nExplicit vs Implicit Hate Speech Analysis:")
        print(f"Total explicit hate instances: {explicit_total}")
        print(f"Misclassified explicit hate instances: {len(explicit_errors)} ({len(explicit_errors)/max(1, explicit_total):.2%})")
        print(f"Total implicit hate instances: {implicit_total}")
        print(f"Misclassified implicit hate instances: {len(implicit_errors)} ({len(implicit_errors)/max(1, implicit_total):.2%})")
    
    return misclassified_df

def create_visualizations(misclassified_df):
    """Create visualizations for misclassification analysis"""
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix([0, 1], [0, 1])  # Placeholder for actual confusion matrix
    sns.heatmap([[misclassified_df['error_type'].value_counts().get('false_negative', 0), 
                  misclassified_df['error_type'].value_counts().get('false_positive', 0)]], 
                annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False Negative', 'False Positive'],
                yticklabels=['Misclassifications'])
    plt.title('Misclassification Types')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'misclassification_types.png'))
    plt.close()
    
    # Visualize explicit vs implicit errors
    if 'hate_type' in misclassified_df.columns:
        # Focus on false negatives (missed hate speech)
        hate_errors_df = misclassified_df[(misclassified_df['error_type'] == 'false_negative') & 
                                        (misclassified_df['hate_type'].isin(['explicit', 'implicit']))]
        
        if not hate_errors_df.empty:
            # Count of explicit vs implicit errors
            plt.figure(figsize=(10, 6))
            hate_type_counts = hate_errors_df['hate_type'].value_counts()
            sns.barplot(x=hate_type_counts.index, y=hate_type_counts.values)
            plt.title('False Negatives by Hate Speech Type')
            plt.xlabel('Hate Speech Type')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'hate_type_errors.png'))
            plt.close()
            
            # Percentage of errors for each hate type
            plt.figure(figsize=(10, 6))
            explicit_total = misclassified_df['stratified_purpose'].value_counts().get(2, 0)
            implicit_total = misclassified_df['stratified_purpose'].value_counts().get(3, 0)
            
            error_percentages = []
            if explicit_total > 0:
                explicit_errors = hate_errors_df[hate_errors_df['hate_type'] == 'explicit'].shape[0]
                error_percentages.append(explicit_errors / explicit_total * 100)
            else:
                error_percentages.append(0)
                
            if implicit_total > 0:
                implicit_errors = hate_errors_df[hate_errors_df['hate_type'] == 'implicit'].shape[0]
                error_percentages.append(implicit_errors / implicit_total * 100)
            else:
                error_percentages.append(0)
                
            sns.barplot(x=['explicit', 'implicit'], y=error_percentages)
            plt.title('Error Rate by Hate Speech Type')
            plt.xlabel('Hate Speech Type')
            plt.ylabel('Error Rate (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'hate_type_error_rates.png'))
            plt.close()

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Error analysis for hate speech detection')
    parser.add_argument('--model_path', type=str, 
                        # Fine-tuned HateBert model
                        default='../../Results/experiment 2/hyperparameter_results_20250324_hatebert/best_model', 
                        help='Path to the fine-tuned model directory')
    parser.add_argument('--test_data', type=str, 
                        default='../../Results/split_data/test_data.csv',
                        help='Path to test data CSV file')
    args = parser.parse_args()
    
    # Load test data from CSV
    try:
        print(f"Loading test data from {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        
        # Handle column name differences if any
        if 'text' not in test_df.columns and 'message' in test_df.columns:
            test_df['text'] = test_df['message']
        
        if 'labels' not in test_df.columns and 'label' in test_df.columns:
            test_df['labels'] = test_df['label']
        
        if 'text' not in test_df.columns or 'labels' not in test_df.columns:
            raise ValueError("CSV must have 'text'/'message' and 'labels'/'label' columns")
        
        # Create the dataset
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
        print(f"Loaded {len(test_df)} test samples")
        print(f"Class distribution: {test_df['labels'].value_counts().to_dict()}")
        
        # Check for stratified_purpose column
        has_stratified = 'stratified_purpose' in test_df.columns
        if has_stratified:
            explicit_count = sum(test_df['stratified_purpose'] == 2)
            implicit_count = sum(test_df['stratified_purpose'] == 3)
            print(f"Hate speech types: {explicit_count} explicit, {implicit_count} implicit")
    except Exception as e:
        print(f"Error loading test data: {e}")
        import sys
        sys.exit(1)
    
    # Load the model and tokenizer
    try:
        model_path = args.model_path
        print(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model.to(device)
        print("Successfully loaded model and tokenizer")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model path and make sure it contains a valid model.")
        import sys
        sys.exit(1)
    
    # Tokenize the test data
    def tokenize_dataset(dataset, tokenizer, max_length=512):
        return tokenizer(
            dataset["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )
    
    tokenized_test = test_dataset.map(
        lambda examples: tokenize_dataset(examples, tokenizer),
        batched=True
    )
    
    # Set format to PyTorch tensors
    columns_to_return = ["input_ids", "attention_mask", "labels"]
    tokenized_test.set_format(type="torch", columns=columns_to_return)
    
    # Create test dataloader
    test_dataloader = DataLoader(tokenized_test, batch_size=32)
    
    # Make predictions
    predictions, true_labels = get_predictions(model, test_dataloader)
    
    # Generate basic classification metrics
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Generate and save classification report
    report = classification_report(true_labels, predictions, target_names=['Non-Hate', 'Hate'], output_dict=True)
    with open(os.path.join(results_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Non-Hate', 'Hate']))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Analyze misclassifications
    stratified_purpose = test_df.get('stratified_purpose', None)
    misclassified_df = analyze_misclassifications(
        test_df['text'].tolist(), true_labels, predictions, stratified_purpose
    )
    
    # Create visualizations
    create_visualizations(misclassified_df)

    # Save explicit and implicit hate misclassifications
    if 'hate_type' in misclassified_df.columns:
        explicit_errors_df = misclassified_df[(misclassified_df['error_type'] == 'false_negative') & 
                                            (misclassified_df['hate_type'] == 'explicit')]
        implicit_errors_df = misclassified_df[(misclassified_df['error_type'] == 'false_negative') & 
                                            (misclassified_df['hate_type'] == 'implicit')]
        
        explicit_errors_df.to_csv(os.path.join(results_dir, 'explicit_hate_errors.csv'), index=False)
        implicit_errors_df.to_csv(os.path.join(results_dir, 'implicit_hate_errors.csv'), index=False)
        
        # Print summary of explicit vs implicit errors
        print("\nExplicit vs Implicit Hate Speech Misclassifications:")
        print(f"Explicit hate errors: {len(explicit_errors_df)}")
        print(f"Implicit hate errors: {len(implicit_errors_df)}")
    
    print(f"\nError analysis results saved to directory: {results_dir}")

if __name__ == "__main__":
    main()