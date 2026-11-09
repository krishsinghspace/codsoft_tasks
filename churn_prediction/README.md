# Customer Churn Prediction

Predicts which bank customers are likely to leave (`Exited = 1`) using Logistic Regression,
Random Forest and Gradient Boosting.

## Dataset
`data/Churn_Modelling.csv` - 10,000 customers, 20.4% churned, no missing values.
Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary.  `RowNumber`, `CustomerId`, `Surname` are dropped (identifiers).

## Pipeline
1. **Feature engineering** - BalanceSalaryRatio, TenureByAge, HasBalance.
2. **Preprocessing** - scaling (Logistic Regression only) + one-hot encoding, all inside a
   scikit-learn `Pipeline` so there is no train/test leakage.
3. **Imbalance** - class weights (`balanced`) instead of discarding data.
4. **Tuning** - 5-fold stratified `RandomizedSearchCV`, optimising ROC-AUC.
5. **Decision threshold** - chosen on out-of-fold *training* predictions to maximise F1
   (the default 0.5 is poor when only ~20% churn). The test set is never used for tuning.
6. **Evaluation** - 20% stratified hold-out: ROC-AUC, PR-AUC, precision, recall, F1.
7. **Interpretation** - permutation feature importance + churn rate by segment.

## Results (hold-out test set, 2,000 customers)
| Model | AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.777 | 0.460 | 0.434 | 0.619 | 0.510 |
| Random Forest | 0.864 | 0.709 | 0.634 | 0.627 | 0.630 |
| **Gradient Boosting** | **0.871** | **0.723** | 0.604 | 0.673 | **0.637** |

(Exact values are written to `results.csv` when you run training.)

## Key findings
- **Age** and **NumOfProducts** are the strongest churn drivers, then **IsActiveMember**, **Balance**, **Geography**.
- Churn peaks at ages 51-60 (~56%); customers with 3-4 products churn at >80%; German customers churn
  about twice as often as French/Spanish; inactive members churn ~2x more than active ones.
- Logistic Regression underperforms because these relationships are non-linear
  (e.g. 2 products = safest, 1 or 3+ = risky), which tree ensembles capture automatically.
- Credit card ownership, tenure and salary have almost no predictive value.

## Run (Windows PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python predict.py --CreditScore 600 --Geography Germany --Gender Female --Age 52 --Tenure 3 --Balance 120000 --NumOfProducts 1 --HasCrCard 1 --IsActiveMember 0 --EstimatedSalary 90000
python predict.py --csv data/Churn_Modelling.csv --out scored_customers.csv
```
Increase `N_ITER` at the top of `train.py` for a deeper hyper-parameter search.

## Outputs
`churn_model.joblib`, `results.csv`, `feature_importance.csv`, `roc_curves.png`, `pr_curves.png`,
`confusion_matrices.png`, `feature_importance.png`, `churn_by_segment.png`, `model_comparison.png`.

## Notes
- The saved model is refit on all 10,000 rows, so scoring that same file gives optimistic
  (in-sample) probabilities. Use the hold-out metrics above to judge real performance.
- Risk tiers: High >= 60%, Medium 30-60%, Low < 30%.
