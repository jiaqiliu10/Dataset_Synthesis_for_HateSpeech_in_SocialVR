import pandas as pd

# Load the classified results dataset
# df = pd.read_csv('/home/liu.jiaqi10/CS7980/hate_speech_classified_results.csv')
# df = pd.read_csv('/home/liu.jiaqi10/CS7980/result_without_vr.csv')
df = pd.read_csv('/home/liu.jiaqi10/CS7980/merge_all_classified_results.csv')
# Overall accuracy
total_correct = (df['label'] == df['predicted_label']).sum()
total_samples = len(df)
overall_accuracy = (total_correct / total_samples) * 100

# Hate Speech accuracy (label == 1)
hate_df = df[df['label'] == 1]
hate_correct = (hate_df['label'] == hate_df['predicted_label']).sum()
hate_total = len(hate_df)
hate_accuracy = (hate_correct / hate_total) * 100 if hate_total > 0 else 0

# Non-Hate Speech accuracy (label == 0)
non_hate_df = df[df['label'] == 0]
non_hate_correct = (non_hate_df['label'] == non_hate_df['predicted_label']).sum()
non_hate_total = len(non_hate_df)
non_hate_accuracy = (non_hate_correct / non_hate_total) * 100 if non_hate_total > 0 else 0

# Display detailed results
print(f"Total samples: {total_samples}")
print(f"Correctly classified samples: {total_correct}")
print(f"Model accuracy: {overall_accuracy:.2f}%\n")

print(f"Hate Speech samples: {hate_total}")
print(f"Correctly classified Hate Speech samples: {hate_correct}")
print(f"Hate Speech accuracy: {hate_accuracy:.2f}%\n")

print(f"Non-Hate Speech samples: {non_hate_total}")
print(f"Correctly classified Non-Hate Speech samples: {non_hate_correct}")
print(f"Non-Hate Speech accuracy: {non_hate_accuracy:.2f}%")
