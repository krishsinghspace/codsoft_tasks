"""
Predict churn for one customer or a whole CSV.

Single customer:
  python predict.py --CreditScore 600 --Geography Germany --Gender Female --Age 45 \
      --Tenure 3 --Balance 120000 --NumOfProducts 1 --HasCrCard 1 \
      --IsActiveMember 0 --EstimatedSalary 90000

Batch (CSV with the same columns as Churn_Modelling.csv; Exited optional):
  python predict.py --csv data/Churn_Modelling.csv --out scored_customers.csv
"""
import argparse
import joblib
import pandas as pd
from churn_utils import risk_tier

art = joblib.load("churn_model.joblib")
model, thr = art["model"], art["threshold"]

ap = argparse.ArgumentParser()
ap.add_argument("--csv"); ap.add_argument("--out", default="scored_customers.csv")
for c, t in [("CreditScore", int), ("Geography", str), ("Gender", str), ("Age", int),
             ("Tenure", int), ("Balance", float), ("NumOfProducts", int),
             ("HasCrCard", int), ("IsActiveMember", int), ("EstimatedSalary", float)]:
    ap.add_argument(f"--{c}", type=t)
a = ap.parse_args()

if a.csv:
    df = pd.read_csv(a.csv)
    p = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["ChurnProbability"] = p.round(4)
    out["PredictedChurn"] = (p >= thr).astype(int)
    out["RiskTier"] = [risk_tier(x) for x in p]
    out.sort_values("ChurnProbability", ascending=False).to_csv(a.out, index=False)
    print(f"Scored {len(out)} customers -> {a.out}")
    print(out["RiskTier"].value_counts().to_string())
else:
    fields = {k: v for k, v in vars(a).items() if k not in ("csv", "out")}
    if any(v is None for v in fields.values()):
        ap.error("give all customer fields, or use --csv")
    p = model.predict_proba(pd.DataFrame([fields]))[0, 1]
    verdict = "WILL LIKELY CHURN" if p >= thr else "likely to stay"
    print(f"Churn probability: {p:.1%}  |  risk tier: {risk_tier(p)}  |  {verdict}")
