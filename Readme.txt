# Email Spam Detection System

Binary text classifier achieving 99% F1-score with zero false positives using Linear SVC and custom NLP preprocessing.

## Problem

Email spam filters must:
- Catch spam reliably (high recall) - users hate spam in inbox
- Never block legitimate emails (perfect precision) - false positives are unacceptable
- Handle diverse text patterns (URLs, numbers, headers)
- Run fast enough for real-time filtering

## Solution

Linear SVC with TF-IDF vectorization, custom text preprocessing, and optimized decision threshold.

## Results

**Test Set Performance:**
- F1-Score: 0.99
- Precision (Spam): 1.00 (zero false positives)
- Recall (Spam): 0.98
- Accuracy: 0.99

**Key Achievement:** 100% precision on spam detection - no legitimate emails blocked.

## Technical Approach

**Pipeline:**
1. Custom Email Preprocessor: Strip headers, replace URLs/numbers, optional stemming
2. TF-IDF Vectorization: max_features=20000, bigrams (1,2), English stop words
3. Linear SVC: Tuned C parameter, hinge loss, L2 regularization
4. Threshold Optimization: ROC analysis to achieve <0.5% false positive rate

**Model Selection:**
- Compared SGD Classifier vs Linear SVC
- Linear SVC achieved slightly better precision/recall balance
- Hyperparameter search: 100 iterations testing preprocessing + model params

**Key Decisions:**
- Custom threshold (via ROC curve) prioritizes precision over recall
- TF-IDF over Count Vectorizer: better for spam detection

## Installation
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib
```

## Usage
```python
import joblib

# Load trained model
model = joblib.load('SCV_Spam_Classifier.joblib')


## Model Architecture
```
Raw Email Text
    ↓
Email Preprocessor (headers, URLs, numbers)
    ↓
TF-IDF Vectorizer (20K features, bigrams)
    ↓
Linear SVC (C=10, hinge loss, L2 penalty)
    ↓
Optimized Threshold (FPR < 0.5%)
    ↓
Spam/Ham Classification
```



## Project Structure
```
spam-detection/
├── datasets/
│   ├── spam_2/
│   ├── easy_ham/
│   └── hard_ham/
├── spam_detection.ipynb
├── SCV_Spam_Classifier.joblib
├── requirements.txt
└── README.md
```

## Performance Analysis

**ROC Analysis:**
- AUC Score: 0.995
- Optimized threshold for <0.5% FPR
- Trade-off: 98% recall for perfect precision

**Cross-Validation:**
- 3-fold CV during hyperparameter tuning
- Consistent performance across folds
- Low variance indicates stable model

## Future Work

- Real-time API deployment (Flask/FastAPI)
- Multilingual support
- Adaptive learning from user feedback
- Integration with email clients

## Technologies

Python, scikit-learn, NLTK, TF-IDF, Linear SVC, pandas, matplotlib

## Contact

**Rafiou Diallo**  
rafioudiallo03@gmail.com  
[GitHub](https://github.com/D-Rafiou)
[Linkedin](https://www.linkedin.com/in/rafiou-diallo-004522260/)