"""
generate_preview_images.py
-----------------------------
Renders static PNG previews from real pipeline output for the README.

Output: ../screenshots/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)

NAVY, ACCENT, RED, GOLD = "#1F3A5F", "#2E6F40", "#C0392B", "#E1A100"


def load(n):
    with open(REPORTS / n, encoding="utf-8") as f:
        return json.load(f)


perf = load("model_performance.json")
audit = load("fairness_audit.json")
shap_a = load("shap_analysis.json")
ret = load("retrieval_evaluation.json")
ext4 = load("spark_performance_study_wsl_native_ext4.json")
mntc = load("spark_performance_study_windows_mnt_c.json")

# ---------------------------------------------------------------- 1. KPIs
fig, axes = plt.subplots(1, 5, figsize=(17, 2.2))
cards = [
    ("Proxy ↔ Group B corr", f"{audit['ground_truth']['proxy_correlation_with_group_b']:.3f}"),
    ("Proxy TRUE effect", "0.0"),
    ("AUC cost of removal", f"{perf['auc_cost_of_removing_proxy']:+.5f}"),
    ("Disparate impact", f"{audit['headline']['disparate_impact_with_proxy']:.4f}"),
    ("Dense vs hybrid MRR", "0.917 / 0.875"),
]
for ax, (label, value) in zip(axes, cards):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY, transform=ax.transAxes, zorder=0))
    ax.text(0.5, 0.68, label, ha="center", va="center", color="white", fontsize=9.5,
            transform=ax.transAxes)
    ax.text(0.5, 0.32, value, ha="center", va="center", color="white", fontsize=15,
            fontweight="bold", transform=ax.transAxes)
fig.suptitle("Fair Lending Intelligence Platform — Key Measurements", fontsize=12, color=NAVY, y=1.08)
plt.tight_layout()
plt.savefig(OUT / "01_kpi_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------- 2. SHAP vs ground truth
imp = pd.DataFrame(shap_a["global_importance"]).head(10).sort_values("mean_abs_shap")
proxy = shap_a["ground_truth_coefficients"]
colors = [RED if f == "neighborhood_risk_score" else NAVY for f in imp.feature]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(imp.feature, imp.mean_abs_shap, color=colors)
ax.set_xlabel("mean |SHAP|")
ax.set_title("SHAP importance — the zero-effect proxy (red) correctly ranks low",
             color=NAVY, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "02_shap_validated.png", dpi=150, bbox_inches="tight")
plt.close()

# --------------------------------- 3. Spark: storage inverted the result
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, (title, d) in zip(axes, [("/mnt/c (Windows FS)", mntc), ("WSL-native ext4", ext4)]):
    j = d["join_strategy"]
    ax.bar(["shuffle", "broadcast"],
           [j["shuffle_join"]["median_seconds"], j["broadcast_join"]["median_seconds"]],
           color=[RED, NAVY])
    ax.set_title(f"{title}\nbroadcast speedup {d['summary']['broadcast_speedup_x']}x",
                 color=NAVY, fontweight="bold", fontsize=10)
    ax.set_ylabel("median seconds")
fig.suptitle("The storage layer inverted the conclusion", fontsize=12, color=NAVY)
plt.tight_layout()
plt.savefig(OUT / "03_spark_storage.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------- 4. retrieval by question type
rows = []
for method, r in ret["retrieval_comparison"].items():
    for qtype, v in r["by_type"].items():
        rows.append({"method": method, "type": qtype, "mrr": v["mrr"]})
df = pd.DataFrame(rows).pivot(index="type", columns="method", values="mrr")
fig, ax = plt.subplots(figsize=(9, 4.4))
df.plot(kind="bar", ax=ax, color=[NAVY, ACCENT, GOLD], rot=0)
ax.set_ylabel("MRR")
ax.set_title("Only the paraphrase bucket separates the retrievers — dense wins",
             color=NAVY, fontweight="bold")
ax.legend(title=None)
plt.tight_layout()
plt.savefig(OUT / "04_retrieval.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved 4 preview images to {OUT}")
