"""
Customer Churn Prediction - training & evaluation
Models: Logistic Regression, Random Forest, Gradient Boosting

Run:  python train.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, precision_recall_curve)

from churn_utils import (load_data, add_features, make_preprocessor, TARGET,
                         FEATURES)

SEED = 42
DATA_PATH = "data/Churn_Modelling.csv"
N_ITER = 10          # random-search candidates per model (raise for a deeper search)


def build(clf, scale):
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", make_preprocessor(scale)),
        ("clf", clf),
    ])


# model -> (pipeline, hyper-parameter search space)
MODELS = {
    "Logistic Regression": (
        build(LogisticRegression(max_iter=3000, class_weight="balanced"), scale=True),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        build(RandomForestClassifier(class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=-1), scale=False),
        {"clf__n_estimators": [150, 250],
         "clf__max_depth": [6, 8, 10, 14],
         "clf__min_samples_leaf": [3, 5, 10, 20],
         "clf__max_features": ["sqrt", 0.5]},
    ),
    "Gradient Boosting": (
        build(GradientBoostingClassifier(random_state=SEED), scale=False),
        {"clf__n_estimators": [100, 200, 300],
         "clf__learning_rate": [0.03, 0.05, 0.1],
         "clf__max_depth": [2, 3, 4],
         "clf__subsample": [0.7, 0.85, 1.0],
         "clf__min_samples_leaf": [5, 20, 50]},
    ),
}


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def main():
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Customers: {len(df)} | churn rate: {y.mean():.1%}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted, probas = [], {}, {}
    for name, (pipe, grid) in MODELS.items():
        print(f"Tuning {name} ...")
        n_iter = min(N_ITER, int(np.prod([len(v) for v in grid.values()])))
        # optimise ROC-AUC: threshold independent, robust for imbalanced data
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=cv,
                                    scoring="roc_auc", n_jobs=-1,
                                    random_state=SEED, refit=True)
        search.fit(X_tr, y_tr)
        model = search.best_estimator_
        print(f"  best params: {search.best_params_}")

        # choose the decision threshold on TRAIN out-of-fold predictions
        # (never on the test set -> no leakage)
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_tr, oof)

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= thr).astype(int)
        rows.append({
            "Model": name,
            "CV AUC": round(search.best_score_, 4),
            "Test AUC": round(roc_auc_score(y_te, proba), 4),
            "PR-AUC": round(average_precision_score(y_te, proba), 4),
            "Threshold": round(thr, 3),
            "Accuracy": round(accuracy_score(y_te, pred), 4),
            "Precision": round(precision_score(y_te, pred), 4),
            "Recall": round(recall_score(y_te, pred), 4),
            "F1": round(f1_score(y_te, pred), 4),
        })
        fitted[name], probas[name] = (model, thr, pred), proba

    res = pd.DataFrame(rows).set_index("Model")
    print("\n=== Test-set comparison ===")
    print(res.to_string())
    res.to_csv("results.csv")

    best_name = res["Test AUC"].idxmax()
    model, thr, pred = fitted[best_name]
    print(f"\nBest model (highest AUC): {best_name}  | decision threshold = {thr:.3f}\n")
    print(classification_report(y_te, pred, target_names=["Stayed", "Churned"], digits=3))

    # refit the winner on ALL data for deployment; keep threshold from CV
    final = build(MODELS[best_name][0].named_steps["clf"].__class__(
        **MODELS[best_name][0].named_steps["clf"].get_params()), scale=(best_name == "Logistic Regression"))
    final.set_params(**{k: v for k, v in model.get_params().items()
                        if k.startswith("clf__") and k != "clf"})
    final.fit(X, y)
    joblib.dump({"model": final, "threshold": thr, "name": best_name}, "churn_model.joblib")

    # ------------------------------ plots ---------------------------------
    plt.figure(figsize=(6, 5))
    for n, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for n, p in probas.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        plt.plot(rc, pr, label=f"{n} (AP={average_precision_score(y_te, p):.3f})")
    plt.axhline(y_te.mean(), color="k", ls="--", alpha=.4, label="baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves")
    plt.legend(); plt.tight_layout(); plt.savefig("pr_curves.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (n, (_, _, pr_)) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_te, pr_),
                               display_labels=["Stayed", "Churned"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(n)
    plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150); plt.close()

    # permutation importance on the test set (works for every model type,
    # reported on the ORIGINAL columns, not one-hot fragments)
    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc",
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    plt.figure(figsize=(7, 5))
    imp.plot.barh(color="steelblue", xerr=pd.Series(pi.importances_std, index=X_te.columns)[imp.index])
    plt.xlabel("Drop in AUC when feature is shuffled")
    plt.title(f"Feature importance ({best_name})"); plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150); plt.close()
    imp.sort_values(ascending=False).round(4).to_csv("feature_importance.csv")

    # EDA: churn rate by key segments
    eda = pd.concat([X, y], axis=1)
    eda["AgeGroup"] = pd.cut(eda["Age"], [17, 30, 40, 50, 60, 100],
                             labels=["<=30", "31-40", "41-50", "51-60", "60+"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), ["Geography", "Gender", "AgeGroup",
                                      "NumOfProducts", "IsActiveMember", "HasCrCard"]):
        (eda.groupby(col, observed=True)[TARGET].mean() * 100).plot.bar(ax=ax, color="indianred")
        ax.set_ylabel("Churn rate (%)"); ax.set_title(f"Churn by {col}")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig("churn_by_segment.png", dpi=150); plt.close()

    # model comparison chart
    res[["Test AUC", "Precision", "Recall", "F1"]].plot.bar(figsize=(8, 4.5), ylim=(0, 1))
    plt.title("Model comparison (test set)"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150); plt.close()

    json.dump({"best_model": best_name, "threshold": thr}, open("model_info.json", "w"))
    print("\nTop drivers of churn:\n", imp.sort_values(ascending=False).head(6).round(4).to_string())
    print("\nSaved: churn_model.joblib, results.csv, feature_importance.csv, *.png")


if __name__ == "__main__":
    main()
