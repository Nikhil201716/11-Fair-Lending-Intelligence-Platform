"""
streamlit_app.py
-------------------
Unified dashboard for the Fair Lending Intelligence Platform. Six tabs,
all reading live from real pipeline output in reports/.

  Overview        - what the platform is and the headline findings
  Spark & Scale   - feature pipeline + the performance study, including
                     the storage-layer measurement that inverted a result
  Model & SHAP    - model performance and SHAP validated against the
                     known data-generating process
  Fairness Audit  - four-fifths, parity, equalized odds, mitigation costs
  Geospatial      - H3 redlining map and proxy-correlation diagnostics
  RAG Assistant   - policy Q&A, retrieval evaluation, red-team results

Run with:
    streamlit run dashboard/streamlit_app.py --server.port 8506
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
DATA = ROOT / "data"

st.set_page_config(page_title="Fair Lending Intelligence Platform", layout="wide", page_icon="⚖️")


def load(name, default=None):
    p = REPORTS / name
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_retriever():
    """Build the dense retriever once per session.

    Without caching, every click re-loads the sentence-transformers model
    and re-embeds the corpus, which made the Ask button take tens of
    seconds each time. Note also that this app must be launched with
    --server.fileWatcherType none: Streamlit's source watcher otherwise
    walks the entire `transformers` package on every rerun, emitting
    hundreds of torchvision import errors and stalling the UI.
    """
    from rag.answer import get_default_retriever
    return get_default_retriever()


st.title("⚖️ Fair Lending Intelligence Platform")
st.caption("PySpark feature engineering at scale · credit risk modeling with SHAP explainability · "
           "fair lending audit against injected ground truth · H3 geospatial redlining analysis · "
           "hybrid-retrieval policy assistant with red-teaming")

tabs = st.tabs(["📋 Overview", "⚡ Spark & Scale", "🧠 Model & SHAP",
                "⚖️ Fairness Audit", "🗺️ Geospatial", "💬 RAG Assistant"])

# ======================================================================
# Overview
# ======================================================================
with tabs[0]:
    truth = None
    p = DATA / "ground_truth.json"
    if p.exists():
        truth = json.load(open(p, encoding="utf-8"))
    audit = load("fairness_audit.json")
    perf = load("model_performance.json")

    st.subheader("The setup")
    st.markdown("""
A fictional lender's data with a **deliberately injected, precisely-known** unfairness mechanism, so
every fairness claim can be scored against ground truth instead of merely sounding plausible:

- Neighborhoods are **residentially segregated**, and `neighborhood_risk_score` is derived from
  neighborhood — so it correlates strongly with the protected group while having **exactly zero true
  causal effect** on default.
- Group B additionally carries a **genuine** economic disadvantage (lower income, higher DTI), so a
  perfectly fair model *still* shows some approval gap. Separating that legitimate gap from a
  proxy-driven one is the actual hard problem.
- The protected attribute is **never** a model input, in either variant.
    """)

    if truth and audit and perf:
        c = st.columns(4)
        c[0].metric("Proxy ↔ Group B correlation", f"{truth['proxy_correlation_with_group_b']:.3f}")
        c[1].metric("Proxy TRUE causal effect", f"{truth['proxy_true_causal_effect']:.1f}")
        c[2].metric("Disparate impact (with proxy)",
                     f"{audit['headline']['disparate_impact_with_proxy']:.4f}")
        c[3].metric("AUC cost of removing proxy",
                     f"{perf['auc_cost_of_removing_proxy']:+.5f}")

    st.info("**Headline finding — the hypothesis was wrong, and that is the result.** "
            "The expectation was that a geography feature correlated 0.54 with the protected group "
            "would drive disparate impact. It did not: removing it moved the disparate impact ratio "
            "only 0.8895 → 0.8870, because the feature carries no real signal and the model already "
            "had verified income, a strictly better predictor. **Correlation with a protected class "
            "is not sufficient to produce measurable harm.**")

# ======================================================================
# Spark & Scale
# ======================================================================
with tabs[1]:
    feat = load("spark_feature_summary.json")
    ext4 = load("spark_performance_study_wsl_native_ext4.json")
    mntc = load("spark_performance_study_windows_mnt_c.json")

    if feat:
        c = st.columns(4)
        c[0].metric("Rows", f"{feat['n_rows']:,}")
        c[1].metric("Feature columns", feat["n_columns"])
        c[2].metric("Month partitions", feat["n_partitions_written"])
        c[3].metric("Spark", feat["spark_version"])

    st.subheader("Performance study — and why the storage layer nearly produced a false conclusion")
    if ext4 and mntc:
        rows = []
        for label, d in (("/mnt/c (Windows FS via WSL 9p)", mntc), ("WSL-native ext4", ext4)):
            rows.append({"storage": label, "comparison": "shuffle join",
                          "median_s": d["join_strategy"]["shuffle_join"]["median_seconds"]})
            rows.append({"storage": label, "comparison": "broadcast join",
                          "median_s": d["join_strategy"]["broadcast_join"]["median_seconds"]})
        fig = px.bar(pd.DataFrame(rows), x="comparison", y="median_s", color="storage",
                      barmode="group", log_y=True)
        fig.update_layout(height=360, yaxis_title="median seconds (log scale)")
        st.plotly_chart(fig, use_container_width=True)

        c = st.columns(2)
        c[0].metric("Broadcast speedup on /mnt/c",
                     f"{mntc['summary']['broadcast_speedup_x']}x",
                     "measurement noise — wrong direction", delta_color="inverse")
        c[1].metric("Broadcast speedup on ext4",
                     f"{ext4['summary']['broadcast_speedup_x']}x", "clean signal")
        st.warning(f"On `/mnt/c` the same shuffle-join query took "
                   f"{mntc['join_strategy']['shuffle_join']['runs']} seconds across runs — "
                   f"variance larger than the effect being measured, producing a "
                   f"**{mntc['summary']['broadcast_speedup_x']}x** 'speedup' that says broadcast "
                   f"join is *slower*. Re-running on ext4 collapsed the variance to "
                   f"{ext4['join_strategy']['shuffle_join']['runs']} and recovered the correct "
                   f"**{ext4['summary']['broadcast_speedup_x']}x**.")

    if ext4:
        st.caption(f"Partition pruning on ext4: "
                   f"{ext4['summary']['partition_pruning_speedup_x']}x "
                   f"(selectivity {ext4['partition_pruning']['selectivity']:.3f}). The gain is "
                   f"modest because Parquet row-group min/max statistics already skip much of the "
                   f"data even without partition pruning.")

# ======================================================================
# Model & SHAP
# ======================================================================
with tabs[2]:
    perf = load("model_performance.json")
    shap_a = load("shap_analysis.json")

    if perf:
        st.subheader("Model performance — with vs. without the geographic proxy")
        rows = []
        for variant in ("with_proxy", "without_proxy"):
            for m, s in perf[variant]["models"].items():
                rows.append({"variant": variant, "model": m, **s})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.metric("AUC cost of removing the proxy (gradient boosting)",
                   f"{perf['auc_cost_of_removing_proxy']:+.5f}",
                   "essentially free to remove")

    if shap_a:
        st.subheader("SHAP importance, validated against the known data-generating process")
        imp = pd.DataFrame(shap_a["global_importance"])
        fig = px.bar(imp.head(10).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                      orientation="h")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Validation checks against ground truth**")
        for k, c in shap_a["validation_checks"].items():
            st.write(("✅ " if c["passed"] else "❌ ") + c["claim"])
        st.caption("Most SHAP write-ups stop at the importance plot. Because this dataset's true "
                   "coefficients are known, the attribution is checked rather than admired — note "
                   "the zero-effect proxy correctly lands in the bottom half.")

        st.metric("Declined applicants whose top reasons include a geographic feature",
                   f"{shap_a['n_declined_with_geographic_reason']}/{shap_a['n_declined_examples']}",
                   "must be escalated, not disclosed (policy AA-400-3)")

# ======================================================================
# Fairness Audit
# ======================================================================
with tabs[3]:
    audit = load("fairness_audit.json")
    mech = load("proxy_mechanism.json")
    aa = load("adverse_action_results.json")

    if audit:
        st.subheader("Four-fifths rule, parity, and equalized odds")
        rows = []
        for v in ("with_proxy", "without_proxy"):
            m = audit[v]["metrics"]
            pg = audit[v]["per_group"]
            rows.append({
                "variant": v,
                "approval Group A": pg["Group A"]["approval_rate"],
                "approval Group B": pg["Group B"]["approval_rate"],
                "disparate impact": m["disparate_impact_ratio"],
                "passes 4/5": m["passes_four_fifths"],
                "parity diff": m["demographic_parity_difference"],
                "EO TPR gap": m["equalized_odds_tpr_gap"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        gt = audit["ground_truth"]
        st.caption(f"Actual default rates differ genuinely: Group A "
                   f"{gt['actual_default_rate_group_a']:.4f} vs Group B "
                   f"{gt['actual_default_rate_group_b']:.4f}. That is injected economic "
                   f"disadvantage, not model bias — which is exactly why demographic parity and "
                   f"equalized odds disagree here (policy FL-300-2).")

        mit = audit["group_threshold_mitigation"]
        st.warning(f"**Group-specific thresholds** reach a perfect disparate impact ratio of "
                   f"{mit['metrics']['disparate_impact_ratio']:.4f} — by construction — while "
                   f"approving {mit['defaulters_approved']:,} applicants who then defaulted. "
                   f"{mit['caveat']}")

    if mech:
        st.subheader("When does a geographic proxy actually cause harm?")
        rows = []
        for name, r in mech["strata"].items():
            rows.append({
                "stratum": name, "n": r["n"],
                "DI with proxy": r["with_proxy"]["disparate_impact_ratio"],
                "DI 95% CI": str(r["with_proxy"].get("bootstrap_ci95")),
                "DI without proxy": r["without_proxy"]["disparate_impact_ratio"],
                "DI without 95% CI": str(r["without_proxy"].get("bootstrap_ci95")),
                "geo share of SHAP":
                    mech["shap_reliance_by_stratum"][name]["geographic_share_of_total_attribution"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("The thin-file point estimates appear to cross the 0.80 regulatory line when the "
                "proxy is removed (0.805 → 0.761), but the bootstrap intervals overlap heavily on "
                "only ~2.7k rows. Reported as **suggestive, not established** — point estimates "
                "alone would have manufactured a regulatory finding out of sampling noise.")

    if aa:
        st.subheader("Adverse action notices — LLM vs. deterministic template")
        llm = aa.get("llm_generation", {})
        tpl = aa.get("deterministic_template", {})

        c = st.columns(4)
        c[0].metric("Declined processed", aa["n_declined"])
        c[1].metric("Escalated (geographic reason)", aa["n_escalated_geographic"],
                     "policy AA-400-3")
        c[2].metric("LLM passed guardrails",
                     f"{llm.get('n_passed_all_guardrails', 0)}/{llm.get('n_notices_generated', 0)}")
        c[3].metric("Template passed guardrails",
                     f"{tpl.get('n_passed_all_guardrails', 0)}/{tpl.get('n_notices_generated', 0)}")

        if llm.get("example_invented_numbers"):
            st.error(f"**The local model fabricated regulatory thresholds.** Numbers appearing in "
                     f"generated notices that were never supplied: "
                     f"`{llm['example_invented_numbers']}` — e.g. *“delinquencies (last 6 months)”* "
                     f"when policy CP-100-3 sets the look-back at **24 months**, and "
                     f"*“debt-to-income greater than 50%”* when policy CP-100-1 sets the decline "
                     f"threshold at **0.45**. A notice stating the wrong threshold is materially "
                     f"false, which is exactly what AA-400-2 prohibits.")

        t_example = next((r for r in aa.get("template_results", [])
                          if not r["escalated"] and r["guardrails"]["all_passed"]), None)
        if t_example:
            st.markdown("**Deterministic template notice (shipped path)**")
            st.write(f"Reasons supplied (from SHAP): `{t_example['disclosable_reasons']}`")
            st.success(t_example["notice"])
        st.caption("SHAP chooses the reasons in both paths; only the wording differs. The LLM was "
                   "given those exact reasons and told not to invent details — and invented them "
                   "anyway. For a document with legal force, the deterministic assembler is the "
                   "correct engineering answer, and this comparison is the evidence for that call.")

# ======================================================================
# Geospatial
# ======================================================================
with tabs[4]:
    red = load("redlining_analysis.json")
    cells_path = DATA / "h3_cells.csv"

    if red and cells_path.exists():
        cells = pd.read_csv(cells_path)
        c = st.columns(3)
        cc = red["correlations_across_cells"]
        c[0].metric("corr(Group B share, approval) — with proxy",
                     f"{cc['group_b_share_vs_approval_rate_with_proxy']:+.3f}")
        c[1].metric("corr(Group B share, approval) — without proxy",
                     f"{cc['group_b_share_vs_approval_rate_without_proxy']:+.3f}")
        c[2].metric("corr(Group B share, avg income)",
                     f"{cc['group_b_share_vs_avg_income']:+.3f}")

        st.subheader("Approval rate by H3 cell")
        fig = px.scatter(cells, x="cell_lon", y="cell_lat", color="approval_rate_with",
                          size="n_applications", color_continuous_scale="RdYlGn",
                          hover_data=["h3_cell", "group_b_share", "actual_default_rate"])
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(cells, x="group_b_share", y="approval_rate_with",
                           size="n_applications", trendline="ols",
                           labels={"group_b_share": "Group B share of cell",
                                   "approval_rate_with": "approval rate"})
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.warning(f"Both model variants trigger the policy FL-200-2 review threshold "
                   f"(|r| > {red['policy_trigger']['threshold_abs_corr']}). But the correlation does "
                   f"**not** weaken when geographic features are removed "
                   f"({cc['group_b_share_vs_approval_rate_with_proxy']:+.3f} → "
                   f"{cc['group_b_share_vs_approval_rate_without_proxy']:+.3f}) — so this spatial "
                   f"pattern is the map of a genuine economic disparity "
                   f"(income correlates {cc['group_b_share_vs_avg_income']:+.3f} with Group B "
                   f"share), not model-driven redlining. The screen correctly flags it for review; "
                   f"a screen is not a verdict.")

# ======================================================================
# RAG Assistant
# ======================================================================
with tabs[5]:
    ret = load("retrieval_evaluation.json")
    rt = load("red_team_results.json")
    ans = load("answerability_experiment.json")

    if ret:
        st.subheader("Retrieval evaluation")
        rows = []
        for method, r in ret["retrieval_comparison"].items():
            row = {"method": method, **{f"MRR ({t})": v["mrr"]
                                          for t, v in r["by_type"].items()}}
            row["MRR (overall)"] = r["overall"]["mrr"]
            row["Recall@1"] = r["overall"]["recall@1"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("**Hybrid retrieval was measurably worse than pure dense here** "
                "(MRR 0.875 vs 0.917). RRF fuses ranks, so when dense is confidently right at "
                "rank 1 and BM25 is wrong at rank 3–4, fusion drags the correct answer down to "
                "rank 2. Hybrid pays off when retrievers have complementary strengths; on this "
                "corpus BM25 had none that dense lacked. The 'hard' question bucket turned out "
                "not to be hard for retrieval — every method scored 1.000 on it — so only the "
                "paraphrase bucket separates the methods.")

    if rt:
        st.subheader("Red-team results")
        c = st.columns(len(rt["by_family"]) + 1)
        c[0].metric("Overall", f"{rt['n_passed']}/{rt['n_attacks']}")
        for i, (fam, v) in enumerate(sorted(rt["by_family"].items()), start=1):
            c[i].metric(fam, f"{v['passed']}/{v['total']}")
        failed = [r for r in rt["results"] if not r["passed"]]
        for f in failed:
            st.error(f"**Real failure found:** “{f['attack']}” → the assistant answered "
                     f"*“{f.get('answer')}”* — a confident fabrication on a topic the corpus does "
                     f"not cover.")

    if ans:
        st.subheader("Three attempted mitigations, all rejected with data")
        st.markdown(f"""
| Mitigation | Result | Verdict |
|---|---|---|
| Retrieval-score floor | legitimate question scores **0.218**, out-of-scope scores **0.305** | structurally cannot separate them |
| Out-of-vocabulary refusal | would refuse **19/20** legitimate questions | rejected |
| LLM answerability gate | refuses **{ans['n_false_refusals']}/{ans['n_legitimate']}** legitimate questions ({ans['false_refusal_rate']:.0%}) while catching {ans['n_correctly_refused']}/{ans['n_out_of_scope']} out-of-scope | rejected — it says NO to nearly everything |
        """)
        st.caption("Residual risk is documented rather than papered over: with a 0.5B local model, "
                   "a question borrowing corpus vocabulary but asking about an uncovered topic can "
                   "still draw a fabrication. The honest fix is a stronger judge model or a trained "
                   "relevance classifier — neither fits the free/local/6GB constraint.")

    st.subheader("Ask the policy assistant")
    st.caption("Retrieval + local Ollama generation, grounded in the policy corpus only.")
    q = st.text_input("Question", value="How quickly must an adverse action notice be delivered?")
    if st.button("Ask"):
        with st.spinner("Retrieving and generating locally (Ollama qwen2.5:0.5b)..."):
            from rag.answer import ask
            r = ask(q, get_retriever())
        if r["refused"]:
            st.error(f"Refused: {r['refusal_reason']}")
        else:
            st.success(r["answer"])
            st.caption(f"Grounding overlap: {r['grounding_score']} "
                       f"({'grounded' if r['grounded'] else 'FLAGGED as ungrounded'})")
        st.dataframe(pd.DataFrame(r["retrieved"]), use_container_width=True, hide_index=True)
