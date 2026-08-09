"""
generate_lending_data.py
---------------------------
Generates a synthetic loan-application dataset for a fictional regional
lender ("Meridian Lending", fictional metro area) with a DELIBERATELY
INJECTED, PRECISELY-KNOWN unfairness mechanism - so the fairness audit
downstream can be scored against ground truth instead of just producing
plausible-looking metrics nobody can check.

THE INJECTED MECHANISM (this is the whole point of the project):

  * Applicants live in one of 60 neighborhoods. Neighborhoods are
    RESIDENTIALLY SEGREGATED - some are ~78% group B, others ~8% - which
    mirrors the real-world pattern that makes geography a usable proxy
    for a protected attribute.

  * `neighborhood_risk_score` is derived from the neighborhood and is
    therefore strongly correlated with `demographic_group`. It has
    **exactly ZERO causal effect on whether an applicant actually
    defaults**. It is a pure proxy: a feature a lender plausibly has,
    that looks predictive only because of who lives where.

  * TRUE default probability depends ONLY on legitimate financial
    factors: debt-to-income, delinquencies, credit history length, and
    income.

  * Group B additionally has a modest, GENUINE economic disadvantage
    (lower average income, higher average DTI). This is deliberate: it
    means a perfectly "fair" model still shows SOME approval-rate gap,
    driven by real risk factors. Distinguishing that legitimate gap from
    the proxy-driven gap is the actual hard problem in fair lending, and
    a dataset where removing one feature drops disparity to zero would
    be a toy that teaches the wrong lesson.

On the protected attribute: this project uses abstract labels ("Group A"
/ "Group B") rather than simulating real ethnic, racial, or religious
demographics. The fairness mathematics is identical, and it avoids
generating synthetic data that could be misread as an empirical claim
about real communities. See README.

Ground truth (the exact coefficients used, and the fact that the proxy's
true coefficient is 0.0) is written to data/ground_truth.json - the
fairness and SHAP steps never read it except to SCORE themselves.

Output: data/loan_applications.csv, data/neighborhoods.csv,
        data/ground_truth.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 41
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

N_APPLICATIONS = 400_000
N_NEIGHBORHOODS = 60

# Fictional metro area bounding box (arbitrary but realistic spacing so
# H3 hexagon aggregation downstream behaves like it would on real data).
LAT_MIN, LAT_MAX = 41.72, 42.08
LON_MIN, LON_MAX = -87.88, -87.52

# --- TRUE data-generating coefficients (log-odds of default) ---------
# These are the ground truth the SHAP step is checked against.
TRUE_COEFFS = {
    "intercept": -2.55,
    "debt_to_income": 3.10,          # strongest legitimate driver
    "num_delinquencies_2y": 0.62,
    "credit_history_length_years": -0.075,
    "log_income_centered": -0.55,
    "neighborhood_risk_score": 0.0,  # <-- THE PROXY. Zero true effect.
}

LOAN_PURPOSES = ["debt_consolidation", "home_improvement", "auto", "medical", "education", "other"]


def build_neighborhoods() -> pd.DataFrame:
    """60 neighborhoods, deliberately segregated, each with a risk score
    that is a function of composition + noise (i.e. a proxy)."""
    nbh_ids = [f"NBH{i:03d}" for i in range(1, N_NEIGHBORHOODS + 1)]

    center_lat = rng.uniform(LAT_MIN, LAT_MAX, N_NEIGHBORHOODS)
    center_lon = rng.uniform(LON_MIN, LON_MAX, N_NEIGHBORHOODS)

    # Segregation: a third of neighborhoods are heavily group B, a third
    # heavily group A, a third mixed.
    group_b_share = np.concatenate([
        rng.uniform(0.68, 0.88, N_NEIGHBORHOODS // 3),
        rng.uniform(0.04, 0.14, N_NEIGHBORHOODS // 3),
        rng.uniform(0.30, 0.55, N_NEIGHBORHOODS - 2 * (N_NEIGHBORHOODS // 3)),
    ])
    rng.shuffle(group_b_share)

    # The proxy: "neighborhood risk score" a lender might buy from a data
    # vendor. Driven mostly by composition, plus noise so it isn't a
    # perfect one-to-one giveaway of the protected attribute.
    nbh_risk = 0.72 * group_b_share + 0.28 * rng.uniform(0, 1, N_NEIGHBORHOODS)
    nbh_risk = (nbh_risk - nbh_risk.min()) / (nbh_risk.max() - nbh_risk.min())

    population_weight = rng.uniform(0.5, 1.5, N_NEIGHBORHOODS)

    return pd.DataFrame({
        "neighborhood_id": nbh_ids,
        "center_lat": np.round(center_lat, 6),
        "center_lon": np.round(center_lon, 6),
        "group_b_share": np.round(group_b_share, 4),
        "neighborhood_risk_score": np.round(nbh_risk, 4),
        "population_weight": np.round(population_weight, 4),
    })


def main():
    neighborhoods = build_neighborhoods()
    neighborhoods.to_csv(DATA_DIR / "neighborhoods.csv", index=False)

    # --- assign applicants to neighborhoods ---
    probs = neighborhoods.population_weight.values / neighborhoods.population_weight.values.sum()
    nbh_idx = rng.choice(N_NEIGHBORHOODS, size=N_APPLICATIONS, p=probs)

    nbh_risk = neighborhoods.neighborhood_risk_score.values[nbh_idx]
    b_share = neighborhoods.group_b_share.values[nbh_idx]

    # --- protected attribute, determined by where you live ---
    is_group_b = rng.random(N_APPLICATIONS) < b_share
    demographic_group = np.where(is_group_b, "Group B", "Group A")

    # --- applicant coordinates: jitter around neighborhood center ---
    latitude = neighborhoods.center_lat.values[nbh_idx] + rng.normal(0, 0.0075, N_APPLICATIONS)
    longitude = neighborhoods.center_lon.values[nbh_idx] + rng.normal(0, 0.0075, N_APPLICATIONS)

    # --- financial features ---
    # Group B carries a modest GENUINE economic disadvantage. This is what
    # makes the fairness analysis non-trivial (see module docstring).
    income_base = rng.lognormal(mean=11.05, sigma=0.42, size=N_APPLICATIONS)
    income_penalty = np.where(is_group_b, 0.88, 1.0)
    annual_income = np.clip(income_base * income_penalty, 14_000, 400_000)

    employment_length = np.clip(rng.gamma(2.2, 2.6, N_APPLICATIONS), 0, 40)

    dti_base = rng.beta(2.6, 6.2, N_APPLICATIONS) * 0.85
    debt_to_income = np.clip(dti_base + np.where(is_group_b, 0.031, 0.0), 0.01, 0.85)

    credit_history_length = np.clip(rng.gamma(3.0, 3.1, N_APPLICATIONS), 0.5, 45)

    num_delinquencies = rng.poisson(
        np.clip(0.32 + debt_to_income * 0.85, 0.05, 4.0), N_APPLICATIONS)

    existing_debt = np.round(annual_income * debt_to_income * rng.uniform(0.55, 1.15, N_APPLICATIONS), 2)
    loan_amount = np.round(np.clip(
        rng.lognormal(9.5, 0.6, N_APPLICATIONS), 1_000, 120_000), 2)
    loan_term_months = rng.choice([24, 36, 48, 60, 72], size=N_APPLICATIONS,
                                    p=[0.10, 0.34, 0.26, 0.22, 0.08])
    loan_purpose = rng.choice(LOAN_PURPOSES, size=N_APPLICATIONS,
                                p=[0.35, 0.18, 0.17, 0.12, 0.10, 0.08])

    # --- TRUE default probability: legitimate factors ONLY ---
    log_income_centered = np.log(annual_income) - np.log(60_000)
    logit = (TRUE_COEFFS["intercept"]
             + TRUE_COEFFS["debt_to_income"] * debt_to_income
             + TRUE_COEFFS["num_delinquencies_2y"] * num_delinquencies
             + TRUE_COEFFS["credit_history_length_years"] * credit_history_length
             + TRUE_COEFFS["log_income_centered"] * log_income_centered
             + TRUE_COEFFS["neighborhood_risk_score"] * nbh_risk)  # coefficient is 0.0
    default_prob = 1.0 / (1.0 + np.exp(-logit))
    defaulted = (rng.random(N_APPLICATIONS) < default_prob).astype(int)

    # --- application dates spread over 2 years ---
    day_offset = rng.integers(0, 730, N_APPLICATIONS)
    application_date = (pd.Timestamp("2024-01-01") + pd.to_timedelta(day_offset, unit="D"))

    applications = pd.DataFrame({
        "application_id": [f"APP{1_000_000 + i}" for i in range(N_APPLICATIONS)],
        "application_date": application_date.strftime("%Y-%m-%d"),
        "neighborhood_id": neighborhoods.neighborhood_id.values[nbh_idx],
        "latitude": np.round(latitude, 6),
        "longitude": np.round(longitude, 6),
        "demographic_group": demographic_group,
        "annual_income": np.round(annual_income, 2),
        "employment_length_years": np.round(employment_length, 2),
        "debt_to_income": np.round(debt_to_income, 4),
        "credit_history_length_years": np.round(credit_history_length, 2),
        "num_delinquencies_2y": num_delinquencies,
        "existing_debt": existing_debt,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose,
        "neighborhood_risk_score": np.round(nbh_risk, 4),
        "defaulted": defaulted,
    })

    applications.to_csv(DATA_DIR / "loan_applications.csv", index=False)

    ground_truth = {
        "seed": SEED,
        "n_applications": N_APPLICATIONS,
        "true_coefficients_log_odds": TRUE_COEFFS,
        "proxy_feature": "neighborhood_risk_score",
        "proxy_true_causal_effect": 0.0,
        "proxy_correlation_with_group_b": float(np.corrcoef(nbh_risk, is_group_b.astype(float))[0, 1]),
        "legitimate_disadvantage": {
            "group_b_income_multiplier": 0.88,
            "group_b_dti_absolute_increase": 0.031,
            "note": "A genuinely fair model should still show SOME approval gap because of these "
                    "real economic differences. The proxy-driven gap is the part that is unfair.",
        },
        "observed_default_rate_overall": float(defaulted.mean()),
        "observed_default_rate_group_a": float(defaulted[~is_group_b].mean()),
        "observed_default_rate_group_b": float(defaulted[is_group_b].mean()),
    }
    with open(DATA_DIR / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {len(applications):,} applications across {N_NEIGHBORHOODS} neighborhoods.")
    print(f"Group split: {(~is_group_b).sum():,} Group A / {is_group_b.sum():,} Group B")
    print(f"Overall default rate: {defaulted.mean():.4f}")
    print(f"  Group A: {defaulted[~is_group_b].mean():.4f}")
    print(f"  Group B: {defaulted[is_group_b].mean():.4f}")
    print(f"Proxy (neighborhood_risk_score) correlation with Group B: "
          f"{ground_truth['proxy_correlation_with_group_b']:.4f}")
    print(f"  ...and its TRUE causal effect on default: {TRUE_COEFFS['neighborhood_risk_score']}")
    print(f"\nWrote {DATA_DIR / 'loan_applications.csv'}")


if __name__ == "__main__":
    main()
