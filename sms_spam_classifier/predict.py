"""
Classify SMS messages with the trained model.

Usage:
  python predict.py "WINNER!! Claim your free prize now"   # single message
  python predict.py                                         # interactive mode
"""
import sys
import joblib
from train import preprocess   # needed so joblib can unpickle the pipeline

model = joblib.load("spam_model.joblib")


def classify(msg: str):
    p_spam = model.predict_proba([msg])[0][1]
    return ("SPAM" if p_spam >= 0.5 else "LEGITIMATE"), p_spam


if __name__ == "__main__":
    if len(sys.argv) > 1:
        label, p = classify(" ".join(sys.argv[1:]))
        print(f"{label}  (spam probability: {p:.1%})")
    else:
        print("SMS spam detector - type a message (empty line to quit)")
        while True:
            msg = input("\n> ").strip()
            if not msg:
                break
            label, p = classify(msg)
            print(f"  -> {label}  (spam probability: {p:.1%})")
