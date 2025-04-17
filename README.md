# Dataset Synthesis for Hate Speech in Social VR
*A specialized dataset and model evaluation framework for detecting hate speech in immersive VR environments.*

## Project Description

This project develops a specialized dataset and detection system for identifying hate speech in Social Virtual Reality (Social VR) environments. With the rapid advancement of social VR platforms like VRChat, there's an increasing need for effective content moderation. Our work addresses the gap in VR-specific hate speech detection by creating a tailored dataset that captures the unique characteristics of harmful content in immersive social environments, particularly focusing on implicit hate speech that develops across conversation turns.

Our dataset (2,594 total samples) was created by combining transcribed audio from the Modified HateMM dataset and synthetic VR scenarios generated using GPT-4o. It contains 1,453 non-hate speech instances (56.01%) and 1,141 hate speech instances (43.99%), with a notable 86.5% of hate speech being implicit rather than explicit, reflecting the subtle nature of toxic language in VR.

## Team

- **Kehan Yan** - Khoury College of Computer Sciences, Northeastern University
- **Jiaqi Liu** - Khoury College of Computer Sciences, Northeastern University
- **Jiangmeng Zhou** - Khoury College of Computer Sciences, Northeastern University

## Mentors

- **Dr. Aanchan Mohan**
- **Dr. Mirjana Prpa**

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

## Instructions for Running the Project

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers library
- Node.js (for certain utilities)
- `.env` file with OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jiaqiliu10/Dataset_Synthesis_for_HateSpeech_in_SocialVR.git
   ```

2. Install Python dependencies:
   ```bash
   pip install torch transformers datasets tqdm matplotlib seaborn scikit-learn pandas numpy whisper openai dotenv
   ```

3. Install Node.js dependencies (if needed):
   ```bash
   npm install
   ```
4. Create a .env file in the root directory and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   ⚠️ Keep your API key secure and make sure .env is listed in .gitignore.

### Data Preparation

Navigate to the `Generation` folder:
```bash
# Transcribe audio files
python Whisper_HateMM/Whisper_HateMM_Transcription.py

# Generate synthetic data
python generateHS.py
python generateNonHS.py

# Filter and merge conversations
python filterConversations.py
```

### Running Experiments

To run the three core experiments, please navigate to the appropriate subfolders and execute the scripts from there.

#### Experiment 1: Pre-trained Model Evaluation
Evaluate performance of models without fine-tuning:

Navigate to the `experiment1` folder:
```bash
cd Evaluation/experiment1
python distilbert_test.py
python roberta_test.py
cd ../..  # Return to project root if needed
```

#### Experiment 2: Dataset Trainability Assessment
Fine-tune and evaluate models on the VR dataset. This script handles data splitting automatically (80/10/10 split).

Navigate to the Fine-Tune folder:
```bash
cd Fine-Tune
python model_finetuning.py
cd ..  # Return to project root if needed
```
This script automatically splits the dataset (80/10/10) and performs model fine-tuning.
Modify the `model_name` variable in `model_finetuning.py` to choose which model to train:

```python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # or use "facebook/roberta-hate-speech-dynabench-r4-target"
# or "GroNLP/hateBERT"
```

#### Experiment 3: VR Hate Speech Characteristics Analysis
Analyze model misclassifications and implicit/explicit hate speech breakdown:

Navigate to the experiment3 folder:
```bash
cd Evaluation/experiment3
python error_analysis.py
cd ../..  # Return to project root if needed
```

### Configuration and Hyperparameters
You can adjust key hyperparameters in model_finetuning.py:
```python
learning_rates = [1e-5, 2e-5, 5e-5]
batch_sizes = [16, 32]
epochs = 3–8
optimizer = AdamW
early_stopping_patience = 2
weight_decay = 0.01
max_grad_norm = 1.0
seed = 42
```
The main entry point for training is model_finetuning.py. Modify model_name inside this script to choose which model to fine-tune.

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
- **Fine-tuned HateBERT**: [https://huggingface.co/spaces/Josh09122/CS7980-HS-hatebert/tree/main](https://huggingface.co/spaces/Josh09122/CS7980-HS-hatebert/tree/main)
- **Fine-tuned RoBERTa**: [https://huggingface.co/spaces/Josh09122/CS7980-HS-roberta/tree/main](https://huggingface.co/spaces/Josh09122/CS7980-HS-roberta/tree/main)
- **Fine-tuned DistilBERT**: [https://huggingface.co/spaces/Josh09122/CS7980-HS-distillbert/tree/main](https://huggingface.co/spaces/Josh09122/CS7980-HS-distillbert/tree/main)

### Generated Dataset
- **VR Hate Speech Dataset**: [https://huggingface.co/spaces/Josh09122/CS7980-HS-Dataset/tree/main](https://huggingface.co/spaces/Josh09122/CS7980-HS-Dataset/tree/main)


## Reproducibility Notes

- All experiments are reproducible using fixed random seeds (seed=42)
- Data splits are stratified to maintain class distribution
- Results in the report can be reproduced by running the listed commands

## Future Work
- Integrate multimodal features (e.g., audio tone, gestures)
- Enable real-time inference for live VR moderation
- Extend dataset to include multilingual scenarios

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
