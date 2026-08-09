"""
train_model.py
-----------------
Trains the credit default model on the Spark-produced feature table, in
TWO variants that are the heart of this project's fairness argument:

  WITH_PROXY    - includes neighborhood_risk_score (and the neighborhood
                  aggregate features derived from geography)
  WITHOUT_PROXY - identical model, identical hyperparameters, identical
                  split, with those geographic features removed

Neither variant ever sees demographic_group. That column is held out
entirely from training and used only by the fairness audit. This matters:
the whole point is that a model with NO protected attribute among its
inputs can still discriminate through a proxy, which is exactly what
FL-200-3 in the policy corpus describes.

Ground truth (data/ground_truth.json) says neighborhood_risk_score has a
TRUE causal effect of 0.0 on default. So the honest expectation is:
dropping it should cost little or no predictive accuracy while removing a
large chunk of the disparity. Whether that actually holds is measured in
fairness/audit.py, not assumed here.

Models: logistic regression (interpretable baseline) and gradient
boosting (the stronger model the bank would actually want to ship).

Output: risk_model/artifacts/*.pkl, reports/model_performance.json/.md
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "database"
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "risk_model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SEED = 42
TARGET = "defaulted"

# Features common to both variants - purely financial
BASE_FEATURES = [
    "debt_to_income", "num_delinquencies_2y", "credit_history_length_years",
    "log_income", "employment_length_years", "loan_to_income",
    "payment_to_income", "loan_amount", "loan_term_months",
    "has_delinquency", "thin_file",
]
# Geography-derived features - present ONLY in the with-proxy variant
PROXY_FEATURES = ["neighborhood_risk_score", "nbh_median_income", "nbh_avg_dti", "income_vs_nbh"]

# Never a model input under any variant.
EXCLUDED_ALWAYS = ["demographic_group", "latitude", "longitude", "neighborhood_id", "group_b_share"]


def load_features() -> pd.DataFrame:
    """Read the Spark-written partitioned Parquet (never .toPandas())."""
    df = pd.read_parquet(DB_DIR / "features")
    return df


def fit_variant(df: pd.DataFrame, features: list[str], label: str, seed: int = SEED) -> dict:
    X = df[features].astype(float).values
    y = df[TARGET].astype(int).values
    idx = np.arange(len(df))

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idx, test_size=0.25, stratify=y, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    logreg = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr_s, y_tr)
    gbm = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6, random_state=seed).fit(X_tr, y_tr)

    out = {"variant": label, "n_features": len(features), "features": features,
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "models": {}}

    for name, model, Xte in (("logistic_regression", logreg, X_te_s),
                              ("gradient_boosting", gbm, X_te)):
        p = model.predict_proba(Xte)[:, 1]
        out["models"][name] = {
            "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
            "pr_auc": round(float(average_precision_score(y_te, p)), 4),
            "brier": round(float(brier_score_loss(y_te, p)), 5),
        }

    with open(ARTIFACTS / f"{label}_bundle.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "logistic_regression": logreg, "gradient_boosting": gbm,
                      "features": features, "test_index": idx_te, "seed": seed}, f)

    return out


def main():
    df = load_features()
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols from Spark feature table")

    with_proxy = fit_variant(df, BASE_FEATURES + PROXY_FEATURES, "with_proxy")
    without_proxy = fit_variant(df, BASE_FEATURES, "without_proxy")

    gb_with = with_proxy["models"]["gradient_boosting"]["roc_auc"]
    gb_without = without_proxy["models"]["gradient_boosting"]["roc_auc"]

    summary = {
        "n_rows": int(len(df)),
        "target": TARGET,
        "excluded_always": EXCLUDED_ALWAYS,
        "note": "demographic_group is NEVER a model input in either variant; it is used only by "
                "the fairness audit.",
        "with_proxy": with_proxy,
        "without_proxy": without_proxy,
        "auc_cost_of_removing_proxy": round(gb_with - gb_without, 5),
    }
    with open(REPORTS_DIR / "model_performance.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "model_performance.md", "w", encoding="utf-8") as f:
        f.write("# Credit Risk Model Performance\n\n")
        f.write("Neither variant uses `demographic_group` as an input.\n\n")
        f.write("| Variant | Model | ROC AUC | PR AUC | Brier |\n|---|---|---|---|---|\n")
        for v in (with_proxy, without_proxy):
            for m, s in v["models"].items():
                f.write(f"| {v['variant']} | {m} | {s['roc_auc']} | {s['pr_auc']} | {s['brier']} |\n")
        f.write(f"\nAUC cost of removing the geographic proxy features (gradient boosting): "
                f"**{summary['auc_cost_of_removing_proxy']:+.5f}**\n")

    print(f"\nWITH proxy    : GB ROC AUC = {gb_with:.4f}")
    print(f"WITHOUT proxy : GB ROC AUC = {gb_without:.4f}")
    print(f"AUC cost of dropping the proxy: {summary['auc_cost_of_removing_proxy']:+.5f}")
    print(f"Saved {REPORTS_DIR / 'model_performance.json'}")


if __name__ == "__main__":
    main()
