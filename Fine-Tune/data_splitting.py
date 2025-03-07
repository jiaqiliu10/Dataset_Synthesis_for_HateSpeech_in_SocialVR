import pandas as pd
from sklearn.model_selection import train_test_split

file_path = '../Data/merged_all_conversations.csv'

# Load data
data = pd.read_csv(file_path)  # TODO: replace this set with the total hate speech dataset

# 把数据集拆分为数据 X 和标签 y
X, y = data.loc[:, ['video', 'message']].values, data.loc[:, 'label'].values

# firstly divide 80% training，20% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# then from 20% divide 10% validation，10% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

# print size of dataset
print(f"Training Set: {len(X_train)}")
print(f"Validation Set: {len(X_val)}")
print(f"Test set: {len(X_test)}")
