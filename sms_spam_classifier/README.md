# SMS Spam Classifier

Classifies SMS messages as **spam** or **legitimate (ham)** using TF-IDF features
and three classical classifiers: Multinomial Naive Bayes, Logistic Regression and Linear SVM.

## Dataset
UCI **SMS Spam Collection** (5,574 messages; 5,169 after removing duplicates; ~12.6% spam).
Stored at `data/spam.csv` (Kaggle format: `v1` = label, `v2` = message; latin-1 encoded).

## Pipeline
1. **Preprocessing** – lowercase; replace URLs, phone numbers, currency amounts and numbers
   with special tokens (`urltoken`, `phonetoken`, `moneytoken`, `numtoken`); strip punctuation.
2. **Features** – TF-IDF (sublinear TF, English stop-words, unigrams/bigrams).
3. **Models** – Naive Bayes, Logistic Regression, Linear SVM.
4. **Tuning** – 5-fold stratified `GridSearchCV` optimising **F1 on the spam class**
   (accuracy alone is misleading because the data is imbalanced).
5. **Evaluation** – held-out 20% stratified test set: accuracy, precision, recall, F1,
   confusion matrix, ROC/AUC.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Naive Bayes** | **0.9874** | **0.9758** | 0.9237 | **0.9490** |
| Logistic Regression | 0.9855 | 0.9677 | 0.9160 | 0.9412 |
| Linear SVM | 0.9865 | 0.9535 | 0.9389 | 0.9462 |

(Exact numbers are in `results.csv` after you run training.)

## How to run
```bash
pip install -r requirements.txt
python train.py                       # trains, evaluates, saves model + plots
python predict.py "You won a free prize! Call now"
python predict.py                     # interactive mode
```

## Output files
`spam_model.joblib` (best model), `results.csv`, `confusion_matrices.png`,
`roc_curves.png`, `model_comparison.png`, `top_words.png`.

## Possible extensions
- Word embeddings (Word2Vec/GloVe averaged vectors) as features
- Character n-gram TF-IDF to catch obfuscated words ("fr33", "w1n")
- Deep models (LSTM / BERT), and a Flask/Streamlit web demo
