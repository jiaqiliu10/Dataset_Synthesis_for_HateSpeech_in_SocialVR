import pandas as pd
from sklearn.model_selection import train_test_split

file_path = '../Data/merged_all_conversations.csv'

# Load data
data = pd.read_csv(file_path)

# Split data into X and y
X = data.loc[:, 'message'].values
y = data.loc[:, 'label'].values
strat_y = data.loc[:, 'stratified_purpose'].values  # Use stratified_purpose for better stratification

# firstly divide 80% training，20% temp
X_train, X_temp, y_train, y_temp, strat_y_train, strat_y_temp = train_test_split(
    X, y, strat_y, test_size=0.2, random_state=42, stratify=strat_y
)

# then from 20% divide 10% validation，10% test
X_val, X_test, y_val, y_test, strat_y_val, strat_y_test = train_test_split(
    X_temp, y_temp, strat_y_temp, test_size=0.5, random_state=42, stratify=strat_y_temp
)

# print size of dataset
print(f"Training Set: {len(X_train)}")
print(f"Validation Set: {len(X_val)}")
print(f"Test set: {len(X_test)}")

# Count distribution in training set
train_nonhate = sum(y_train == 0)
train_hate = sum(y_train == 1)
train_explicit = sum(strat_y_train == 2)
train_implicit = sum(strat_y_train == 3)

# Count distribution in validation set
val_nonhate = sum(y_val == 0)
val_hate = sum(y_val == 1)
val_explicit = sum(strat_y_val == 2)
val_implicit = sum(strat_y_val == 3)

# Count distribution in test set
test_nonhate = sum(y_test == 0)
test_hate = sum(y_test == 1)
test_explicit = sum(strat_y_test == 2)
test_implicit = sum(strat_y_test == 3)

# Print detailed statistics
print(f"\nIn total {sum(y == 0)} non-hate speech, {sum(y == 1)} hate speech ({sum(strat_y == 2)} explicit hate, {sum(strat_y == 3)} implicit hate)")
print(f"Train set ({len(X_train)}), validation set ({len(X_val)}), test set ({len(X_test)})")
print(f"For train set, {train_nonhate} non-hate speech, {train_hate} hate speech ({train_explicit} explicit hate, {train_implicit} implicit hate)")
print(f"For validation set, {val_nonhate} non-hate speech, {val_hate} hate speech ({val_explicit} explicit hate, {val_implicit} implicit hate)")
print(f"For test set, {test_nonhate} non-hate speech, {test_hate} hate speech ({test_explicit} explicit hate, {test_implicit} implicit hate)")

# Save the datasets to CSV files
print("\nSaving datasets to CSV files...")

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), 'split_data')
os.makedirs(output_dir, exist_ok=True)

# Save train set
train_df = pd.DataFrame({
    "text": X_train,
    "labels": y_train,
    "stratified_purpose": strat_y_train
})
train_path = os.path.join(output_dir, 'train_data.csv')
train_df.to_csv(train_path, index=False)
print(f"Train data saved to {train_path}")

# Save validation set
val_df = pd.DataFrame({
    "text": X_val,
    "labels": y_val,
    "stratified_purpose": strat_y_val
})
val_path = os.path.join(output_dir, 'validation_data.csv')
val_df.to_csv(val_path, index=False)
print(f"Validation data saved to {val_path}")

# Save test set
test_df = pd.DataFrame({
    "text": X_test,
    "labels": y_test,
    "stratified_purpose": strat_y_test
})
test_path = os.path.join(output_dir, 'test_data.csv')
test_df.to_csv(test_path, index=False)
print(f"Test data saved to {test_path}")
