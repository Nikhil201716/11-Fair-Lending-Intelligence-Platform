"""
proxy_mechanism.py
---------------------
Why this file exists: the headline fairness audit produced a result that
CONTRADICTED the working hypothesis, and the contradiction was worth
chasing rather than smoothing over.

The hypothesis was: a geographic feature correlated 0.54 with the
protected group would drive disparate impact, and removing it would close
the gap. What actually happened (reports/fairness_audit.json):

    disparate impact WITH proxy    0.8895
    disparate impact WITHOUT proxy 0.8870   (essentially unchanged)

The explanation is that the proxy carries no real signal (true causal
effect 0.0) AND the model already has verified income, which is a
strictly better predictor of the same thing. With 400k rows, gradient
boosting simply learned to mostly ignore it - SHAP ranked it 10th of 15.
Correlation with a protected class is NOT sufficient on its own to
produce measurable harm.

That raises the sharper question this script answers: geography is the
classic redlining proxy in the real world, so WHEN does it actually bite?
The mechanism should be information substitution - a model leans on
geography exactly when the legitimate signals are weak or missing. So the
test is to stratify by how much genuine information the applicant file
carries and re-measure:

    thin file      - credit_history_length_years < 2 (little real signal)
    established    - everyone else

If the mechanism is real, the proxy's SHAP influence and the disparity
gap between the two model variants should both be larger among thin-file
applicants. If it is not, that is also a finding and is reported as one.

RESULTS (measured) - the hypothesis is REFUTED on its main claim:

  geographic share of total SHAP attribution
      thin_file    5.15%
      established  5.94%      <- the model leans on geography slightly
                                 LESS for thin files, not more

  disparate impact, with proxy -> without proxy
      established  0.8910 -> 0.8892  (n=97,254)
      thin_file    0.8050 -> 0.7608  (n=2,746)

The established stratum's change (-0.0019) is negligible and its bootstrap
CIs overlap almost completely, so removing the proxy does essentially
nothing there.

The thin-file movement looks dramatic and crosses the 0.80 regulatory
line, which is exactly why it was bootstrapped before being believed:

      with proxy     DI 0.8050  CI95 [0.736, 0.877]  P(DI<0.80)=0.45
      without proxy  DI 0.7608  CI95 [0.695, 0.836]  P(DI<0.80)=0.85

Those intervals overlap heavily. On 2,746 rows the "removing the proxy
pushes thin-file applicants below four-fifths" story is SUGGESTIVE BUT
NOT ESTABLISHED, and it is reported that way. Reporting the point
estimates alone would have manufactured a regulatory finding out of
sampling noise.

The directionally consistent (if statistically weak) result is still
interesting: removing geography made disparity WORSE, not better. The
likely reason is that income_vs_nbh - the highest-ranked geographic
feature at SHAP rank 5 - normalizes an applicant's income against local
context, partially offsetting Group B's lower ABSOLUTE income. Not every
geographic feature is a redlining proxy; some carry legitimate context,
and "delete all geographic features" is not automatically the fair
choice. Establishing that properly would need a larger thin-file sample.

Output: reports/proxy_mechanism.json/.md
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "database"
ARTIFACTS = ROOT / "risk_model" / "artifacts"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

GROUP_COL = "demographic_group"
TARGET = "defaulted"
APPROVE_THRESHOLD = 0.15
SEED = 42
N_SHAP = 2000


N_BOOTSTRAP = 2000


def _di_point(approved: np.ndarray, group: np.ndarray, gs: list) -> float | None:
    ars = []
    for g in gs:
        m = group == g
        if not m.any():
            return None
        ars.append(float(approved[m].mean()))
    lo, hi = min(ars), max(ars)
    return lo / hi if hi > 0 else None


def disparate_impact(approved: np.ndarray, group: np.ndarray, bootstrap: bool = False,
                      seed: int = 0) -> dict:
    """Disparate impact ratio, optionally with a bootstrap CI.

    The CI matters because the thin-file stratum has only ~2.7k rows, and
    a point estimate that straddles the 0.80 regulatory threshold there
    would be irresponsible to report as a pass/fail without showing how
    much of the movement is sampling noise."""
    gs = sorted(set(group))
    ars = {g: float(approved[group == g].mean()) for g in gs}
    lo, hi = min(ars.values()), max(ars.values())
    out = {"approval_rates": {g: round(v, 5) for g, v in ars.items()},
            "disparate_impact_ratio": round(lo / hi, 5) if hi > 0 else None,
            "parity_difference": round(abs(ars[gs[0]] - ars[gs[1]]), 5)}

    if bootstrap:
        rng = np.random.default_rng(seed)
        n = len(approved)
        vals = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            v = _di_point(approved[idx], group[idx], gs)
            if v is not None:
                vals.append(v)
        vals = np.array(vals)
        out["bootstrap_ci95"] = [round(float(np.percentile(vals, 2.5)), 5),
                                  round(float(np.percentile(vals, 97.5)), 5)]
        out["bootstrap_p_below_0.80"] = round(float((vals < 0.80).mean()), 4)
        out["n_bootstrap"] = int(len(vals))
    return out


def proba_for(label: str, test: pd.DataFrame) -> np.ndarray:
    with open(ARTIFACTS / f"{label}_bundle.pkl", "rb") as f:
        b = pickle.load(f)
    return b["gradient_boosting"].predict_proba(test[b["features"]].astype(float).values)[:, 1]


def main():
    import shap

    df = pd.read_parquet(DB_DIR / "features")
    with open(ARTIFACTS / "with_proxy_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    test = df.iloc[bundle["test_index"]].reset_index(drop=True)

    p_with = proba_for("with_proxy", test)
    p_without = proba_for("without_proxy", test)

    strata = {
        "thin_file": test["thin_file"].astype(int).values == 1,
        "established": test["thin_file"].astype(int).values == 0,
    }

    results = {}
    for name, mask in strata.items():
        sub = test[mask]
        di_w = disparate_impact((p_with[mask] < APPROVE_THRESHOLD).astype(int),
                                 sub[GROUP_COL].values, bootstrap=True, seed=SEED)
        di_wo = disparate_impact((p_without[mask] < APPROVE_THRESHOLD).astype(int),
                                  sub[GROUP_COL].values, bootstrap=True, seed=SEED + 1)
        results[name] = {
            "n": int(mask.sum()),
            "actual_default_rate": round(float(sub[TARGET].mean()), 5),
            "with_proxy": di_w,
            "without_proxy": di_wo,
            "di_change_from_removing_proxy": round(
                di_wo["disparate_impact_ratio"] - di_w["disparate_impact_ratio"], 5),
        }

    # --- how much does the model LEAN on the proxy within each stratum? ---
    features = bundle["features"]
    model = bundle["gradient_boosting"]
    explainer = shap.TreeExplainer(model)
    rng = np.random.default_rng(SEED)
    proxy_share = {}
    for name, mask in strata.items():
        sub = test[mask]
        take = min(N_SHAP, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        X = sub.iloc[idx][features].astype(float).values
        sv = explainer.shap_values(X)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        total = mean_abs.sum()
        geo_idx = [features.index(f) for f in
                   ("neighborhood_risk_score", "nbh_median_income", "nbh_avg_dti", "income_vs_nbh")
                   if f in features]
        proxy_share[name] = {
            "n_shap_sample": int(take),
            "geographic_share_of_total_attribution": round(float(mean_abs[geo_idx].sum() / total), 5),
            "neighborhood_risk_score_mean_abs_shap": round(
                float(mean_abs[features.index("neighborhood_risk_score")]), 6),
        }

    summary = {
        "approve_threshold": APPROVE_THRESHOLD,
        "hypothesis": "the geographic proxy matters more where legitimate signal is weak",
        "strata": results,
        "shap_reliance_by_stratum": proxy_share,
        "verdict": {
            "thin_file_di_change": results["thin_file"]["di_change_from_removing_proxy"],
            "established_di_change": results["established"]["di_change_from_removing_proxy"],
            "geo_attribution_thin_file":
                proxy_share["thin_file"]["geographic_share_of_total_attribution"],
            "geo_attribution_established":
                proxy_share["established"]["geographic_share_of_total_attribution"],
        },
    }
    with open(REPORTS_DIR / "proxy_mechanism.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "proxy_mechanism.md", "w", encoding="utf-8") as f:
        f.write("# When Does a Geographic Proxy Actually Cause Harm?\n\n")
        f.write("| Stratum | n | DI with proxy | DI without proxy | Change | Geo share of SHAP |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['n']:,} | {r['with_proxy']['disparate_impact_ratio']:.4f} | "
                    f"{r['without_proxy']['disparate_impact_ratio']:.4f} | "
                    f"{r['di_change_from_removing_proxy']:+.4f} | "
                    f"{proxy_share[name]['geographic_share_of_total_attribution']:.1%} |\n")

    for name, r in results.items():
        print(f"{name:<13} n={r['n']:>7,}  DI with={r['with_proxy']['disparate_impact_ratio']:.4f}  "
              f"DI without={r['without_proxy']['disparate_impact_ratio']:.4f}  "
              f"change={r['di_change_from_removing_proxy']:+.4f}")
    print()
    for name, s in proxy_share.items():
        print(f"{name:<13} geographic features = "
              f"{s['geographic_share_of_total_attribution']:.2%} of total SHAP attribution")
    print(f"\nSaved {REPORTS_DIR / 'proxy_mechanism.json'}")


if __name__ == "__main__":
    main()
