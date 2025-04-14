## Project Structure

```
DATASET_SYNTHESIS_FOR_HATESPEECH/
├── Data/
│   ├── HateMM_conversation.json      # Transcribed conversations from HateMM dataset
│   └── merged_all_conversations.csv  # Final processed dataset
├── Evaluation/
│   ├── experiment1/                  # Experiment 1: Pre-trained Model Evaluation
│   │   ├── distilbert_test.py        # Evaluation using DistilBERT (no fine-tuning)
│   │   └── roberta_test.py           # Evaluation using RoBERTa (no fine-tuning)
│   └── experiment3/                  # Experiment 3: VR Hate Speech Characteristics Analysis
│       └── error_analysis.py         # In-depth error analysis on detection results
├── Fine-Tune/                        # Experiment 2: Dataset Trainability Assessment
│   ├── data_splitting.py             # Script to split dataset into train/val/test
│   └── model_finetuning.py           # Model fine-tuning with hyperparameter optimization
├── Generation/
│   ├── Whisper_HateMM/
│   │   └── Whisper_HateMM_Transcription.py  # Audio transcription using Whisper
│   ├── filterConversations.py        # Filter and clean conversations
│   ├── generateHS.py                 # Generate hate speech examples
│   └── generateNonHS.py              # Generate non-hate speech examples
└── node_modules/                     # Node.js dependencies
```

## Team

- **Kehan Yan** - Khoury College of Computer Sciences, Northeastern University
- **Jiaqi Liu** - Khoury College of Computer Sciences, Northeastern University
- **Jiangmeng Zhou** - Khoury College of Computer Sciences, Northeastern University

## Mentors

- **Dr. Aanchan Mohan**
- **Dr. Mirjana Prpa**

## Project Description

This project develops a specialized dataset and detection system for identifying hate speech in Social Virtual Reality (Social VR) environments. With the rapid advancement of social VR platforms like VRChat, there's an increasing need for effective content moderation. Our work addresses the gap in VR-specific hate speech detection by creating a tailored dataset that captures the unique characteristics of harmful content in immersive social environments, particularly focusing on implicit hate speech that develops across conversation turns.

Our dataset (2,594 total samples) was created by combining transcribed audio from the Modified HateMM dataset and synthetic VR scenarios generated using GPT-4o. It contains 1,453 non-hate speech instances (56.01%) and 1,141 hate speech instances (43.99%), with a notable 86.5% of hate speech being implicit rather than explicit, reflecting the subtle nature of toxic language in VR.

## Instructions for Running the Project

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers library
- Node.js (for certain utilities)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jiaqiliu10/Dataset_Synthesis_for_HateSpeech_in_SocialVR.git
   ```

2. Install Python dependencies:
   ```bash
   pip install torch transformers datasets tqdm matplotlib seaborn scikit-learn pandas numpy whisper openai
   ```

3. Install Node.js dependencies (if needed):
   ```bash
   npm install
   ```

### Data Preparation

```bash
# Transcribe audio files
python Generation/Whisper_HateMM/Whisper_HateMM_Transcription.py

# Generate synthetic data
python Generation/generateHS.py
python Generation/generateNonHS.py

# Filter and merge conversations
python Generation/filterConversations.py
```

### Running Experiments

```bash
# Experiment 1: Pre-trained Model Evaluation
python Evaluation/experiment1/distilbert_test.py
python Evaluation/experiment1/roberta_test.py

# Experiment 2: Dataset Trainability Assessment
# Then fine-tune the model of your choice by modifying the model_name variable in model_finetuning.py
# Options include:
# - "distilbert-base-uncased-finetuned-sst-2-english" for DistilBERT
# - "facebook/roberta-hate-speech-dynabench-r4-target" for RoBERTa-hate-speech
# - "GroNLP/hateBERT" for HateBERT
python Fine-Tune/model_finetuning.py

# Experiment 3: VR Hate Speech Characteristics Analysis
python Evaluation/experiment3/error_analysis.py
```

## Expected Output and Summary Results

### Experiment 1: Pre-trained Model Evaluation
Running the scripts will evaluate the performance of pre-trained models on our VR hate speech dataset without any fine-tuning. Expected outputs:

- **DistilBERT**: 82.07% accuracy, F1-score: 0.80
- **RoBERTa-hate-speech**: 59.56% accuracy, F1-score: 0.15

Output files: Classification reports, confusion matrices, and evaluation metrics in CSV format.

### Experiment 2: Dataset Trainability Assessment
The fine-tuning process will create a timestamped directory (e.g., `hyperparameter_results_20250414_123456`) containing:

- Model checkpoints for the best-performing configurations
- Confusion matrices and classification reports
- Learning curve visualizations for each hyperparameter combination
- Summary JSON file with optimal hyperparameters

Expected performance after fine-tuning:
- **DistilBERT**: 83.46% accuracy, F1-score: 0.835
- **RoBERTa-hate-speech**: 80.77% accuracy, F1-score: 0.808
- **HateBERT**: 84.23% accuracy, F1-score: 0.842

### Experiment 3: Error Analysis
Running the error analysis script will generate:

- Detailed breakdown of classification errors
- Analysis of detection performance by hate speech type (explicit vs. implicit)
- Visualization of error distributions
- Examples of misclassified conversations

Key findings:
- 100% accuracy on explicit hate speech
- 94.95% accuracy on implicit hate speech
- False positives (87.8% of errors) more common than false negatives (12.2%)

## Links to Models and Generated Data

### Pre-trained Models Used
- **DistilBERT**: [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- **RoBERTa-hate-speech**: [facebook/roberta-hate-speech-dynabench-r4-target](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target)
- **HateBERT**: [GroNLP/hateBERT](https://huggingface.co/GroNLP/hateBERT)

### Fine-tuned Models
- **Fine-tuned HateBERT**: [https://huggingface.co/northeastern-khoury/hatebert-vr](https://huggingface.co/northeastern-khoury/hatebert-vr)
- **Fine-tuned RoBERTa**: [https://huggingface.co/northeastern-khoury/roberta-hate-speech-vr](https://huggingface.co/northeastern-khoury/roberta-hate-speech-vr)
- **Fine-tuned DistilBERT**: [https://huggingface.co/northeastern-khoury/distilbert-vr-hate-speech](https://huggingface.co/northeastern-khoury/distilbert-vr-hate-speech)

### Generated Dataset
- **VR Hate Speech Dataset**: [https://huggingface.co/datasets/northeastern-khoury/vr-hate-speech](https://huggingface.co/datasets/northeastern-khoury/vr-hate-speech)

## Conclusion
This study attempted to create and evaluate a specialized dataset for hate speech detection in social VR environments, with fine-tuned models showing promising performance. Our analysis suggests that VR hate speech may have some unique characteristics, with implicit expressions developing across multiple conversation turns appearing to be more common.

The best model achieved:
- 100% accuracy in detecting explicit hate speech
- 94.95% accuracy in detecting implicit hate speech
- 84.23% overall accuracy (HateBERT)

## Acknowledgments

- HateMM dataset creators
- OpenAI for Whisper transcription technology and GPT-4o
- Hugging Face for the Transformers library

## License

This project is licensed under the MIT License - see the LICENSE file for details.