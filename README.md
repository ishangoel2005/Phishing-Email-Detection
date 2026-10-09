# Cross-Channel Threat Detection: Phishing Email Classification

> A Robust Machine Learning framework for detecting phishing emails using a **Hybrid Feature Engineering approach** combining TF-IDF NLP text representations with security indicator rules and Gradient Boosted Decision Trees (LightGBM).

---

## Project Overview

This project focuses on Cross Channel Threat Detection by identifying and classifying phishing email attempts. By analyzing both textual content (subject line and email body) and structural/heuristic security indicators (URL presence, credential request patterns, urgency keywords, casing ratios), the model provides high-accuracy, low-latency detection of deceptive email communications.

### Key Highlights
- **Processed & Transformed Raw Email Datasets**: Cleaned and standardized raw email text into high-dimensional structured features for model training and evaluation.
- **Hybrid Feature Engineering**: Fused TF-IDF text representations with domain-specific security rules (suspicious URLs, IP tokens, punctuation density, urgent language patterns, sender domain signals).
- **LightGBM Gradient Boosting**: Trained a tuned LightGBM classifier with early stopping and a 70-15-15 train-validation-test split (along with 5-fold Stratified CV) to prevent overfitting and ensure strong generalization.
- **Semantic Text Analytics**: Extracted features from email subject lines and body text using TF-IDF vectorization, enhancing detection of subtle semantic phishing patterns.
- **Exceptional Performance**: Achieved **99.46% Accuracy** and **99.53% F1-Score** on test evaluation sets.

---

## Key Results & Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.46%** |
| **F1-Score** | **99.53%** |
| **Precision** | **99.50%** |
| **Recall** | **99.56%** |

---

## System Architecture & Workflow

```
┌────────────────────────┐
│   Raw Email Datasets   │
│ (Subject, Body, Domain)│
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Data Preprocessing    │
│  - Text Normalization  │
│  - Token Extraction    │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│              Hybrid Feature Engineering                │
├────────────────────────────┬───────────────────────────┤
│    TF-IDF Vectorization    │   Rule-based Indicators   │
│ (Subject & Body Analytics) │ (URLs, Urgency, Capital)  │
└────────────────────────────┴───────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│              LightGBM Classifier Training              │
│          (70-15-15 Split w/ Early Stopping)            │
└───────────┬────────────────────────────┬───────────────┘
            │                            │
            ▼                            ▼
┌────────────────────────┐  ┌────────────────────────────┐
│ Model Evaluation & SHAP│  │ Inference Engine / API     │
│  Feature Interpretability│ │ Real-Time Threat Detection │
└────────────────────────┘  └────────────────────────────┘
```

---

## Methodology & Feature Engineering

### 1. Textual NLP Features (TF-IDF)
- Concatenated cleaned subject lines and body text.
- Fitted TF-IDF Vectorizer with `ngram_range=(1, 2)` and `max_features=30,000` to capture semantic phishing phrases (e.g., *"account suspended"*, *"verify identity"*).

### 2. Rule-Based Security Indicators
- **URL & Token Detection**: Counts of `[URL]`, `[IP]`, `[EMAIL]` placeholders.
- **Urgency & Manipulation Cues**: Frequency of high-risk keywords (*urgent*, *verify*, *password*, *click*, *bank*, *security*).
- **Stylistic & Formatting Signals**: Uppercase-to-lowercase ratio, exclamation/question mark counts, punctuation density, word length, character count.
- **Domain Signals**: Categorization and top-K domain encoding for sender email headers.

### 3. Model Training & Optimization
- **Classifier**: LightGBM (Gradient Boosted Trees).
- **Hyperparameters**: `learning_rate=0.05`, `num_leaves=64`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_alpha=0.1`, `reg_lambda=1.0`.
- **Validation**: 5-Fold Stratified K-Fold CV & 70-15-15 Split with Early Stopping to guarantee strong out-of-fold generalization.

---

## Repository Structure

```
.
├── MODEL 5_SYN/                  # Main Hybrid LightGBM Model Architecture
│   ├── preprocessing.py          # Data cleaning and tokenization pipeline
│   ├── features.py               # Hybrid feature extraction (TF-IDF + Security Rules)
│   ├── model.py                  # LightGBM training & Stratified K-Fold evaluation
│   ├── testing.py                # Model validation and metrics calculation
│   └── prediction.py             # Inference pipeline for new/unseen emails
├── data_clean.py                 # Dataset cleaning utilities
├── combining_csv.py              # Dataset consolidation scripts
├── .gitignore                    # Git ignore rules for datasets and model weights
└── README.md                     # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Recommended: Virtual Environment (`venv` or `conda`)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate  
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

*(Required core libraries: `scikit-learn`, `lightgbm`, `pandas`, `numpy`, `scipy`, `joblib`, `matplotlib`, `tqdm`)*

---

## Usage

### 1. Feature Extraction
Run the feature pipeline to extract TF-IDF vectors and security features:
```bash
python "MODEL 5_SYN/features.py"
```

### 2. Train the Model
Train the LightGBM classifier with early stopping:
```bash
python "MODEL 5_SYN/model.py"
```

### 3. Evaluate & Test
Evaluate on test datasets:
```bash
python "MODEL 5_SYN/testing.py"
```

---

## Feature Importance & Interpretability
The model utilizes **SHAP (SHapley Additive exPlanations)** to provide transparent, interpretable decisions. Top features influencing classification decisions include:
1. TF-IDF unigrams/bigrams relating to financial or security urgency.
2. Frequency of embedded URLs and IP tokens.
3. Ratio of uppercase text and exclamation counts.
4. Sender domain reputation indicators.

---

## Contributing

Contributions, issues and feature requests are welcome! Feel free to check the issues page.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.