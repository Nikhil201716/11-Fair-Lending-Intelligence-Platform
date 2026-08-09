"""
redlining_analysis.py
------------------------
Geospatial fair-lending analysis using H3 hexagons.

Why hexagons rather than the neighborhood boundaries already in the data:
neighborhoods are the unit the BIAS WAS INJECTED ON, so aggregating by
neighborhood would just replay the generator's own structure back at us.
H3 imposes an independent, uniform spatial grid that does not know where
neighborhood lines are - so any spatial pattern it finds has to survive
re-bucketing into cells that cut across those boundaries. That is the
difference between measuring geography and measuring the generator.

Three questions, in order:

  1. WHERE are approvals concentrated? Approval rate per H3 cell.

  2. Is geography a PROXY for the protected attribute? Correlation
     between a cell's Group B share and its approval rate. Policy FL-200-2
     sets the internal review trigger at |r| > 0.30.

  3. Does the map change when the proxy features are removed? The same
     cells scored by both model variants. If removing geography from the
     model flattens the spatial disparity, that is the redlining
     signature; if it does not, the spatial pattern is being driven by
     something else - which is what the non-spatial audit already
     suggested.

Output: reports/redlining_analysis.json/.md, data/h3_cells.csv
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

H3_RESOLUTION = 7          # ~5km edge; coarse enough for stable per-cell rates
MIN_APPS_PER_CELL = 100    # cells below this are excluded as too noisy to score
APPROVE_THRESHOLD = 0.15
GROUP_COL = "demographic_group"
PROXY_CORR_TRIGGER = 0.30  # policy FL-200-2


def proba_for(label: str, test: pd.DataFrame) -> np.ndarray:
    with open(ARTIFACTS / f"{label}_bundle.pkl", "rb") as f:
        b = pickle.load(f)
    return b["gradient_boosting"].predict_proba(test[b["features"]].astype(float).values)[:, 1]


def main():
    import h3

    df = pd.read_parquet(DB_DIR / "features")
    with open(ARTIFACTS / "with_proxy_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    test = df.iloc[bundle["test_index"]].reset_index(drop=True)

    print(f"Assigning H3 cells (resolution {H3_RESOLUTION}) to {len(test):,} test applications...")
    test["h3_cell"] = [h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
                       for lat, lon in zip(test["latitude"], test["longitude"])]

    test["approved_with"] = (proba_for("with_proxy", test) < APPROVE_THRESHOLD).astype(int)
    test["approved_without"] = (proba_for("without_proxy", test) < APPROVE_THRESHOLD).astype(int)
    test["is_group_b"] = (test[GROUP_COL] == "Group B").astype(int)

    cells = (test.groupby("h3_cell")
             .agg(n_applications=("application_id", "count"),
                   group_b_share=("is_group_b", "mean"),
                   approval_rate_with=("approved_with", "mean"),
                   approval_rate_without=("approved_without", "mean"),
                   actual_default_rate=("defaulted", "mean"),
                   avg_income=("annual_income", "mean"),
                   avg_dti=("debt_to_income", "mean"))
             .reset_index())

    total_cells = len(cells)
    cells = cells[cells.n_applications >= MIN_APPS_PER_CELL].reset_index(drop=True)
    cells["approval_delta"] = cells.approval_rate_without - cells.approval_rate_with

    # add cell centroids for mapping
    centroids = [h3.cell_to_latlng(c) for c in cells.h3_cell]
    cells["cell_lat"] = [c[0] for c in centroids]
    cells["cell_lon"] = [c[1] for c in centroids]
    cells.to_csv(DATA_DIR / "h3_cells.csv", index=False)

    def corr(a, b):
        return float(np.corrcoef(cells[a], cells[b])[0, 1])

    corr_with = corr("group_b_share", "approval_rate_with")
    corr_without = corr("group_b_share", "approval_rate_without")
    corr_default = corr("group_b_share", "actual_default_rate")
    corr_income = corr("group_b_share", "avg_income")

    # spatial disparity: gap between most- and least-approving cells
    def spread(col):
        return round(float(cells[col].max() - cells[col].min()), 5)

    summary = {
        "h3_resolution": H3_RESOLUTION,
        "min_apps_per_cell": MIN_APPS_PER_CELL,
        "n_cells_total": int(total_cells),
        "n_cells_scored": int(len(cells)),
        "n_applications_scored": int(cells.n_applications.sum()),
        "correlations_across_cells": {
            "group_b_share_vs_approval_rate_with_proxy": round(corr_with, 5),
            "group_b_share_vs_approval_rate_without_proxy": round(corr_without, 5),
            "group_b_share_vs_actual_default_rate": round(corr_default, 5),
            "group_b_share_vs_avg_income": round(corr_income, 5),
        },
        "policy_trigger": {
            "threshold_abs_corr": PROXY_CORR_TRIGGER,
            "with_proxy_triggers_review": bool(abs(corr_with) > PROXY_CORR_TRIGGER),
            "without_proxy_triggers_review": bool(abs(corr_without) > PROXY_CORR_TRIGGER),
            "policy_reference": "FL-200-2",
        },
        "spatial_spread": {
            "approval_rate_with_proxy": spread("approval_rate_with"),
            "approval_rate_without_proxy": spread("approval_rate_without"),
        },
        "mean_abs_approval_change_from_removing_proxy":
            round(float(cells.approval_delta.abs().mean()), 5),
        "interpretation_note": (
            "A strong negative correlation between a cell's Group B share and its approval rate "
            "does NOT by itself prove the model is redlining: Group B also has genuinely lower "
            "income and higher default in this data by construction. The diagnostic that "
            "separates the two is whether the correlation weakens when the geographic features "
            "are removed from the model."
        ),
    }
    with open(REPORTS_DIR / "redlining_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "redlining_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Geospatial Redlining Analysis (H3)\n\n")
        f.write(f"{len(cells)} H3 cells at resolution {H3_RESOLUTION} with at least "
                f"{MIN_APPS_PER_CELL} applications.\n\n")
        f.write("| Correlation across cells | r |\n|---|---|\n")
        for k, v in summary["correlations_across_cells"].items():
            f.write(f"| {k} | {v:+.4f} |\n")
        f.write(f"\nPolicy FL-200-2 review trigger is |r| > {PROXY_CORR_TRIGGER}: "
                f"with-proxy model **{'TRIGGERS' if summary['policy_trigger']['with_proxy_triggers_review'] else 'does not trigger'}**, "
                f"without-proxy model **{'TRIGGERS' if summary['policy_trigger']['without_proxy_triggers_review'] else 'does not trigger'}**.\n")

    print(f"\nScored {len(cells)} of {total_cells} H3 cells "
          f"(>= {MIN_APPS_PER_CELL} applications each)")
    print(f"  corr(group_b_share, approval WITH proxy)    = {corr_with:+.4f}")
    print(f"  corr(group_b_share, approval WITHOUT proxy) = {corr_without:+.4f}")
    print(f"  corr(group_b_share, ACTUAL default rate)    = {corr_default:+.4f}")
    print(f"  corr(group_b_share, avg income)             = {corr_income:+.4f}")
    print(f"\n  Policy FL-200-2 trigger (|r| > {PROXY_CORR_TRIGGER}): "
          f"with={summary['policy_trigger']['with_proxy_triggers_review']}, "
          f"without={summary['policy_trigger']['without_proxy_triggers_review']}")
    print(f"  Mean |approval change| from removing proxy: "
          f"{summary['mean_abs_approval_change_from_removing_proxy']:.5f}")
    print(f"\nSaved {REPORTS_DIR / 'redlining_analysis.json'} and {DATA_DIR / 'h3_cells.csv'}")


if __name__ == "__main__":
    main()
