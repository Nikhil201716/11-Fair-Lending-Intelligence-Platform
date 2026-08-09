"""
audit.py
-----------
The fair-lending audit. Scores both model variants (with and without the
geographic proxy) on the standard metrics, then measures what each
mitigation actually costs.

Metrics implemented directly rather than pulled from a fairness library,
so every number is inspectable:

  DISPARATE IMPACT RATIO - approval rate of the lower group divided by
      approval rate of the higher group. Below 0.80 fails the four-fifths
      screen (policy FL-300-1).

  DEMOGRAPHIC PARITY DIFF - absolute gap in approval rates. Does not
      condition on the actual outcome, so it treats a genuine difference
      in creditworthiness as unfairness.

  EQUALIZED ODDS - gap in true positive rate and false positive rate
      between groups, conditioned on actual default. Under this
      definition a model may legitimately approve groups at different
      rates if their actual default rates differ.

Why both parity and equalized odds are reported: this dataset is built so
they DISAGREE. Group B has a genuinely higher default rate (14.77% vs
12.54%) caused by real injected economic disadvantage, so a model can be
perfectly calibrated and still fail demographic parity. Reporting only the
metric that flatters a conclusion is the most common way fairness analysis
goes wrong, so both are reported and the conflict is named (policy
FL-300-2 describes exactly this).

THE KEY COMPARISON: ground truth says the geographic proxy has a TRUE
causal effect of 0.0 on default, and train_model.py measured that removing
it costs +0.0004 AUC. So if removing it also closes much of the disparity,
that is a "less discriminatory alternative achieving the same business
objective" - the legal standard in FL-200-3 - established with evidence
rather than asserted.

Output: reports/fairness_audit.json/.md
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

GROUP_COL = "demographic_group"
TARGET = "defaulted"
# Approve when predicted default probability is below the threshold.
# 0.15 is near the overall base rate (13.5%), i.e. a realistic operating point.
APPROVE_THRESHOLD = 0.15
FOUR_FIFTHS = 0.80


def rates(y_true: np.ndarray, approved: np.ndarray, group: np.ndarray) -> dict:
    out = {}
    for g in sorted(set(group)):
        m = group == g
        yt, ap = y_true[m], approved[m]
        # "positive" = approved. defaulted==1 means bad outcome.
        good = yt == 0
        bad = yt == 1
        out[g] = {
            "n": int(m.sum()),
            "approval_rate": round(float(ap.mean()), 5),
            "actual_default_rate": round(float(yt.mean()), 5),
            # TPR here = share of GOOD applicants correctly approved
            "tpr_good_approved": round(float(ap[good].mean()), 5) if good.any() else None,
            # FPR here = share of BAD applicants wrongly approved
            "fpr_bad_approved": round(float(ap[bad].mean()), 5) if bad.any() else None,
        }
    return out


def fairness_metrics(per_group: dict) -> dict:
    groups = sorted(per_group)
    a, b = per_group[groups[0]], per_group[groups[1]]
    ar_a, ar_b = a["approval_rate"], b["approval_rate"]
    lo, hi = min(ar_a, ar_b), max(ar_a, ar_b)
    return {
        "disparate_impact_ratio": round(lo / hi, 5) if hi > 0 else None,
        "passes_four_fifths": bool((lo / hi) >= FOUR_FIFTHS) if hi > 0 else None,
        "demographic_parity_difference": round(abs(ar_a - ar_b), 5),
        "equalized_odds_tpr_gap": round(abs(a["tpr_good_approved"] - b["tpr_good_approved"]), 5),
        "equalized_odds_fpr_gap": round(abs(a["fpr_bad_approved"] - b["fpr_bad_approved"]), 5),
    }


def evaluate_variant(df: pd.DataFrame, label: str, model_name="gradient_boosting") -> dict:
    with open(ARTIFACTS / f"{label}_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    features = bundle["features"]
    test = df.iloc[bundle["test_index"]].reset_index(drop=True)

    X = test[features].astype(float).values
    if model_name == "logistic_regression":
        X = bundle["scaler"].transform(X)
    proba = bundle[model_name].predict_proba(X)[:, 1]

    approved = (proba < APPROVE_THRESHOLD).astype(int)
    y = test[TARGET].astype(int).values
    group = test[GROUP_COL].values

    per_group = rates(y, approved, group)
    return {"variant": label, "model": model_name, "threshold": APPROVE_THRESHOLD,
            "per_group": per_group, "metrics": fairness_metrics(per_group),
            "overall_approval_rate": round(float(approved.mean()), 5)}


def group_threshold_mitigation(df: pd.DataFrame, label="with_proxy",
                                model_name="gradient_boosting") -> dict:
    """Third mitigation option from policy FL-300-3: group-specific
    thresholds tuned so approval rates match. Reported WITH its cost, and
    with the legal caveat that using group membership at decision time is
    itself a form of disparate treatment - which is why it is presented as
    an option that was measured, not a recommendation."""
    with open(ARTIFACTS / f"{label}_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    features = bundle["features"]
    test = df.iloc[bundle["test_index"]].reset_index(drop=True)
    proba = bundle[model_name].predict_proba(test[features].astype(float).values)[:, 1]
    group = test[GROUP_COL].values
    y = test[TARGET].astype(int).values

    base_rate = float((proba < APPROVE_THRESHOLD).mean())
    thresholds = {}
    for g in sorted(set(group)):
        m = group == g
        # threshold that gives this group the overall approval rate
        thresholds[g] = float(np.quantile(proba[m], base_rate))

    approved = np.zeros(len(proba), dtype=int)
    for g, t in thresholds.items():
        m = group == g
        approved[m] = (proba[m] < t).astype(int)

    per_group = rates(y, approved, group)
    bad_approved = int(((approved == 1) & (y == 1)).sum())
    return {"strategy": "group_specific_thresholds", "thresholds": {k: round(v, 5)
                                                                     for k, v in thresholds.items()},
            "per_group": per_group, "metrics": fairness_metrics(per_group),
            "overall_approval_rate": round(float(approved.mean()), 5),
            "defaulters_approved": bad_approved,
            "caveat": "Applying different thresholds by group uses the protected attribute at "
                      "decision time, which is itself disparate treatment. Measured for "
                      "completeness, not recommended."}


def main():
    df = pd.read_parquet(DB_DIR / "features")
    with open(DATA_DIR / "ground_truth.json", encoding="utf-8") as f:
        truth = json.load(f)
    with open(REPORTS_DIR / "model_performance.json", encoding="utf-8") as f:
        perf = json.load(f)

    with_proxy = evaluate_variant(df, "with_proxy")
    without_proxy = evaluate_variant(df, "without_proxy")
    mitigated = group_threshold_mitigation(df)

    di_with = with_proxy["metrics"]["disparate_impact_ratio"]
    di_without = without_proxy["metrics"]["disparate_impact_ratio"]

    summary = {
        "approve_threshold": APPROVE_THRESHOLD,
        "ground_truth": {
            "proxy_feature": truth["proxy_feature"],
            "proxy_true_causal_effect": truth["proxy_true_causal_effect"],
            "proxy_correlation_with_group_b": round(truth["proxy_correlation_with_group_b"], 4),
            "actual_default_rate_group_a": truth["observed_default_rate_group_a"],
            "actual_default_rate_group_b": truth["observed_default_rate_group_b"],
        },
        "with_proxy": with_proxy,
        "without_proxy": without_proxy,
        "group_threshold_mitigation": mitigated,
        "headline": {
            "disparate_impact_with_proxy": di_with,
            "disparate_impact_without_proxy": di_without,
            "disparity_closed": round(di_without - di_with, 5),
            "auc_cost_of_removing_proxy": perf["auc_cost_of_removing_proxy"],
            "less_discriminatory_alternative_exists": bool(
                di_without > di_with and abs(perf["auc_cost_of_removing_proxy"]) < 0.005),
        },
    }
    with open(REPORTS_DIR / "fairness_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "fairness_audit.md", "w", encoding="utf-8") as f:
        f.write("# Fair Lending Audit\n\n")
        f.write(f"Approve when predicted default probability < {APPROVE_THRESHOLD}.\n\n")
        f.write("| Variant | Approval A | Approval B | Disparate Impact | 4/5 rule | "
                "Parity diff | EO TPR gap |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for v in (with_proxy, without_proxy):
            g = sorted(v["per_group"])
            m = v["metrics"]
            f.write(f"| {v['variant']} | {v['per_group'][g[0]]['approval_rate']:.3f} | "
                    f"{v['per_group'][g[1]]['approval_rate']:.3f} | "
                    f"{m['disparate_impact_ratio']:.4f} | "
                    f"{'PASS' if m['passes_four_fifths'] else 'FAIL'} | "
                    f"{m['demographic_parity_difference']:.4f} | "
                    f"{m['equalized_odds_tpr_gap']:.4f} |\n")
        f.write(f"\n- Removing the geographic proxy moved the disparate impact ratio "
                f"**{di_with:.4f} -> {di_without:.4f}** "
                f"at an AUC cost of **{perf['auc_cost_of_removing_proxy']:+.5f}**.\n")

    print(f"Disparate impact ratio  WITH proxy: {di_with:.4f} "
          f"({'PASS' if with_proxy['metrics']['passes_four_fifths'] else 'FAIL'} four-fifths)")
    print(f"Disparate impact ratio  W/O  proxy: {di_without:.4f} "
          f"({'PASS' if without_proxy['metrics']['passes_four_fifths'] else 'FAIL'} four-fifths)")
    print(f"AUC cost of removing it           : {perf['auc_cost_of_removing_proxy']:+.5f}")
    print(f"\nApproval rates WITH proxy : "
          f"{ {g: v['approval_rate'] for g, v in with_proxy['per_group'].items()} }")
    print(f"Approval rates W/O  proxy : "
          f"{ {g: v['approval_rate'] for g, v in without_proxy['per_group'].items()} }")
    print(f"\nEqualized-odds TPR gap  WITH: {with_proxy['metrics']['equalized_odds_tpr_gap']:.4f}"
          f"  W/O: {without_proxy['metrics']['equalized_odds_tpr_gap']:.4f}")
    print(f"Group-threshold mitigation -> DI "
          f"{mitigated['metrics']['disparate_impact_ratio']:.4f}, "
          f"defaulters approved: {mitigated['defaulters_approved']}")
    print(f"\nSaved {REPORTS_DIR / 'fairness_audit.json'}")


if __name__ == "__main__":
    main()
