"""
SMS Spam Classifier - training & evaluation
Compares TF-IDF + {Naive Bayes, Logistic Regression, Linear SVM}
with a small hyper-parameter search, saves the best model and plots.

Run:  python train.py
"""
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
DATA_PATH = "data/spam.csv"     # Kaggle format: v1 = label, v2 = message


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    # The Kaggle file is latin-1 encoded and has 3 empty trailing columns
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df["y"] = (df["label"] == "spam").astype(int)      # spam = 1, ham = 0
    return df


# --------------------------------------------------------------------------
# 2. Text preprocessing
#    Replace URLs / phone numbers / money / numbers with tokens. These
#    patterns are strong spam signals even when the exact value differs.
# --------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"[£$€]\s?\d+(?:[.,]\d+)?", " moneytoken ", text)
    text = re.sub(r"\b\d{10,}\b", " phonetoken ", text)
    text = re.sub(r"\b\d+\b", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# 3. Models: TF-IDF (word 1-2 grams) + classifier
# --------------------------------------------------------------------------
def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, stop_words="english",
                                  sublinear_tf=True)),
        ("clf", clf),
    ])


MODELS = {
    "Naive Bayes": (
        make_pipeline(MultinomialNB()),
        {"tfidf__ngram_range": [(1, 1), (1, 2)],
         "tfidf__min_df": [1, 2],
         "clf__alpha": [0.05, 0.1, 0.5, 1.0]},
    ),
    "Logistic Regression": (
        make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
        {"tfidf__ngram_range": [(1, 1), (1, 2)],
         "tfidf__min_df": [1, 2],
         "clf__C": [1, 10, 50]},
    ),
    "Linear SVM": (
        make_pipeline(LinearSVC(class_weight="balanced")),
        {"tfidf__ngram_range": [(1, 1), (1, 2)],
         "tfidf__min_df": [1, 2],
         "clf__C": [0.1, 0.5, 1, 5]},
    ),
}


def get_scores(model, X):
    """Continuous spam score for ROC curve (works for all 3 model types)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    df = load_data()
    print(f"Messages after de-duplication: {len(df)}")
    print(df["label"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"],
        random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted, rocs = [], {}, {}

    for name, (pipe, grid) in MODELS.items():
        print(f"Tuning {name} ...")
        # optimise F1 on spam class: accuracy alone is misleading (imbalanced data)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        pred = best.predict(X_test)

        results.append({
            "Model": name,
            "CV F1": round(gs.best_score_, 4),
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred), 4),
            "Recall": round(recall_score(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
        })
        fitted[name] = (best, pred)
        fpr, tpr, _ = roc_curve(y_test, get_scores(best, X_test))
        rocs[name] = (fpr, tpr, auc(fpr, tpr))
        print(f"  best params: {gs.best_params_}")

    res = pd.DataFrame(results).set_index("Model")
    print("\n=== Test-set comparison ===")
    print(res.to_string())
    res.to_csv("results.csv")

    # ---- pick best by F1 (tie-break: precision, since false positives hurt) --
    best_name = res.sort_values(["F1", "Precision"], ascending=False).index[0]
    best_model, best_pred = fitted[best_name]
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, best_pred, target_names=["ham", "spam"], digits=4))

    # ---- Final model: make sure it can output a probability ---------------
    final = best_model
    if not hasattr(final, "predict_proba"):
        # Calibrate SVM so predict.py can show a confidence %
        tfidf = final.named_steps["tfidf"]
        svm = CalibratedClassifierCV(final.named_steps["clf"], cv=5)
        final = Pipeline([("tfidf", tfidf), ("clf", svm)]).fit(X_train, y_train)
    joblib.dump(final, "spam_model.joblib")
    json.dump({"best_model": best_name}, open("model_info.json", "w"))

    # ---- Plots -------------------------------------------------------------
    # confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, (_, pred)) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                               display_labels=["ham", "spam"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150); plt.close()

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, a) in rocs.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={a:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150); plt.close()

    # model comparison bar chart
    res[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(figsize=(8, 4.5), ylim=(0.8, 1.0))
    plt.title("Model comparison (test set)"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig("model_comparison.png", dpi=150); plt.close()

    # top spam / ham words (use logistic regression coefficients)
    lr_pipe = MODELS["Logistic Regression"][0]
    lr_pipe.set_params(**{k: v for k, v in fitted["Logistic Regression"][0].get_params().items()
                          if k.startswith(("tfidf__ngram", "tfidf__min_df", "clf__C"))})
    lr_pipe.fit(X_train, y_train)
    vocab = np.array(lr_pipe.named_steps["tfidf"].get_feature_names_out())
    coef = lr_pipe.named_steps["clf"].coef_[0]
    top_spam, top_ham = np.argsort(coef)[-15:], np.argsort(coef)[:15]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(vocab[top_spam], coef[top_spam], color="crimson"); axes[0].set_title("Top SPAM indicators")
    axes[1].barh(vocab[top_ham], -coef[top_ham], color="seagreen"); axes[1].set_title("Top LEGITIMATE indicators")
    plt.tight_layout(); plt.savefig("top_words.png", dpi=150); plt.close()

    print("\nSaved: spam_model.joblib, results.csv, confusion_matrices.png, "
          "roc_curves.png, model_comparison.png, top_words.png")


if __name__ == "__main__":
    main()
