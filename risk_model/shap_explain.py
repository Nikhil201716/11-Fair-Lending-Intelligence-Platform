"""
shap_explain.py
------------------
SHAP explanations for the credit model, at two levels:

  GLOBAL - mean |SHAP| per feature, then CHECKED AGAINST THE KNOWN
           data-generating process in data/ground_truth.json. Most SHAP
           write-ups stop at "here is the importance plot"; because this
           dataset's true coefficients are known, the attribution can
           actually be validated instead of admired.

           Honest caveat stated up front: mean |SHAP| and a logistic
           coefficient are NOT the same quantity. SHAP magnitude depends
           on the feature's spread in the data as well as its effect size,
           so an exact rank match is not expected and is not the test.
           The checkable claims are directional:
             * debt_to_income (true coeff 3.10, the dominant driver)
               should rank at or near the top
             * neighborhood_risk_score (true coeff 0.0) should rank near
               the BOTTOM despite correlating 0.54 with the protected
               group - if it ranked high, the model would be leaning on a
               feature with no real predictive content

  LOCAL  - per-applicant SHAP reason codes for declined applications.
           These are what the adverse-action notice generator consumes:
           Regulation B requires the specific principal reasons for THAT
           applicant, which global importance cannot provide (see policy
           section MG-500-2).

Output: reports/shap_analysis.json/.md, risk_model/artifacts/shap_reasons.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "database"
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "risk_model" / "artifacts"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SEED = 42
N_SHAP_SAMPLE = 3000      # SHAP on 400k rows is needless; sample for the global view
N_LOCAL_EXAMPLES = 25     # declined applicants to generate reason codes for

# Human-readable reason phrasing for adverse-action notices.
REASON_LABELS = {
    "debt_to_income": "Debt-to-income ratio too high",
    "num_delinquencies_2y": "Recent delinquencies on credit obligations",
    "has_delinquency": "Presence of a delinquency in the last 24 months",
    "credit_history_length_years": "Length of credit history insufficient",
    "thin_file": "Insufficient credit file depth",
    "log_income": "Income relative to requested amount",
    "employment_length_years": "Length of employment",
    "loan_to_income": "Requested loan amount high relative to income",
    "payment_to_income": "Estimated payment high relative to income",
    "loan_amount": "Requested loan amount",
    "loan_term_months": "Requested loan term",
    "neighborhood_risk_score": "GEOGRAPHIC FEATURE - must not be disclosed (escalate)",
    "nbh_median_income": "GEOGRAPHIC FEATURE - must not be disclosed (escalate)",
    "nbh_avg_dti": "GEOGRAPHIC FEATURE - must not be disclosed (escalate)",
    "income_vs_nbh": "GEOGRAPHIC FEATURE - must not be disclosed (escalate)",
}


def main():
    import shap

    with open(ARTIFACTS / "with_proxy_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(DATA_DIR / "ground_truth.json", encoding="utf-8") as f:
        truth = json.load(f)

    features = bundle["features"]
    model = bundle["gradient_boosting"]

    df = pd.read_parquet(DB_DIR / "features")
    test_df = df.iloc[bundle["test_index"]].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_df), size=min(N_SHAP_SAMPLE, len(test_df)), replace=False)
    sample = test_df.iloc[sample_idx].reset_index(drop=True)
    X = sample[features].astype(float).values

    print(f"Computing SHAP values on {len(X):,} sampled test rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):          # older API returns per-class list
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:                   # (n, features, classes)
        shap_values = shap_values[:, :, -1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_importance = [{"feature": features[i], "mean_abs_shap": round(float(mean_abs[i]), 6),
                           "rank": int(r + 1)} for r, i in enumerate(order)]

    # ---------- validate against the known DGP ----------
    ranks = {g["feature"]: g["rank"] for g in global_importance}
    n_feat = len(features)
    proxy = truth["proxy_feature"]
    checks = {
        "dti_is_top_2": {
            "claim": "debt_to_income (true coeff 3.10) ranks in the top 2",
            "rank": ranks.get("debt_to_income"), "passed": ranks.get("debt_to_income", 99) <= 2,
        },
        "proxy_ranks_low": {
            "claim": f"{proxy} (TRUE causal effect 0.0) ranks in the bottom half",
            "rank": ranks.get(proxy), "n_features": n_feat,
            "passed": ranks.get(proxy, 0) > n_feat / 2,
        },
        "delinquency_above_proxy": {
            "claim": "num_delinquencies_2y (true coeff 0.62) outranks the zero-effect proxy",
            "delinquency_rank": ranks.get("num_delinquencies_2y"), "proxy_rank": ranks.get(proxy),
            "passed": ranks.get("num_delinquencies_2y", 99) < ranks.get(proxy, 0),
        },
    }

    # ---------- local reason codes for declined applicants ----------
    proba = model.predict_proba(sample[features].astype(float).values)[:, 1]
    decline_idx = np.argsort(proba)[::-1][:N_LOCAL_EXAMPLES]
    reasons = []
    for i in decline_idx:
        contribs = sorted(zip(features, shap_values[i]), key=lambda kv: -kv[1])
        top = [{"feature": f, "shap": round(float(v), 6), "label": REASON_LABELS.get(f, f),
                 "is_geographic": REASON_LABELS.get(f, "").startswith("GEOGRAPHIC")}
               for f, v in contribs[:4] if v > 0]
        reasons.append({
            "application_id": sample.iloc[i]["application_id"],
            "default_probability": round(float(proba[i]), 4),
            "demographic_group": sample.iloc[i]["demographic_group"],  # audit only, not an input
            "top_reasons": top,
        })

    n_with_geo = sum(1 for r in reasons if any(t["is_geographic"] for t in r["top_reasons"]))

    summary = {
        "n_shap_sample": int(len(X)),
        "model": "gradient_boosting (with_proxy variant)",
        "global_importance": global_importance,
        "ground_truth_coefficients": truth["true_coefficients_log_odds"],
        "validation_checks": checks,
        "all_checks_passed": all(c["passed"] for c in checks.values()),
        "n_declined_examples": len(reasons),
        "n_declined_with_geographic_reason": n_with_geo,
        "geographic_reason_rate": round(n_with_geo / len(reasons), 4) if reasons else 0.0,
    }
    with open(REPORTS_DIR / "shap_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ARTIFACTS / "shap_reasons.json", "w", encoding="utf-8") as f:
        json.dump(reasons, f, indent=2)

    with open(REPORTS_DIR / "shap_analysis.md", "w", encoding="utf-8") as f:
        f.write("# SHAP Analysis, Validated Against Ground Truth\n\n")
        f.write("| Rank | Feature | Mean \\|SHAP\\| | True coefficient |\n|---|---|---|---|\n")
        tc = truth["true_coefficients_log_odds"]
        for g in global_importance:
            f.write(f"| {g['rank']} | {g['feature']} | {g['mean_abs_shap']:.6f} | "
                    f"{tc.get(g['feature'], '(derived)')} |\n")
        f.write("\n## Validation checks\n\n")
        for k, c in checks.items():
            f.write(f"- **{'PASS' if c['passed'] else 'FAIL'}** — {c['claim']}\n")

    print("\nTop 6 features by mean |SHAP|:")
    for g in global_importance[:6]:
        print(f"  {g['rank']}. {g['feature']:<28} {g['mean_abs_shap']:.6f}")
    print(f"\n  ...{proxy} ranked {ranks.get(proxy)}/{n_feat} (true causal effect: 0.0)")
    print("\nValidation checks:")
    for k, c in checks.items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['claim']}")
    print(f"\nDeclined examples with a geographic feature in their top reasons: "
          f"{n_with_geo}/{len(reasons)}")
    print(f"Saved {REPORTS_DIR / 'shap_analysis.json'}")


if __name__ == "__main__":
    main()
