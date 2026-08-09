"""
run_pipeline.py
------------------
One-command orchestrator for the Fair Lending Intelligence Platform.

IMPORTANT - the Spark stage runs under WSL, not Windows. Spark's Hadoop
write path requires winutils.exe on Windows (reads and computation work
natively; every write fails). Rather than install an unofficial
third-party binary, spark_layer/ runs inside WSL Ubuntu. This script
detects the platform and shells out to WSL for that stage automatically.
Pass --skip-spark to reuse an existing feature table.

LLM stages (RAG answering, red-teaming, the answerability experiment, and
adverse-action generation) are opt-in via --include-llm-steps: together
they make ~60 local Ollama calls.

Usage:
    python scripts/run_pipeline.py [--skip-spark] [--include-llm-steps]
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

WSL_PROJECT = "/mnt/c/Users/nikhi/OneDrive/Desktop/Projects/11-Fair-Lending-Intelligence-Platform"

CORE_STEPS = [
    ("1/8 Generating synthetic lending data (400k applications, injected proxy)",
     ["scripts/generate_lending_data.py"]),
    ("2/8 Generating the policy document corpus", ["rag/generate_policy_corpus.py"]),
    ("4/8 Training credit risk models (with vs. without the geographic proxy)",
     ["risk_model/train_model.py"]),
    ("5/8 SHAP explainability, validated against ground truth",
     ["risk_model/shap_explain.py"]),
    ("6/8 Fair lending audit (four-fifths, parity, equalized odds)", ["fairness/audit.py"]),
    ("6b/8 Proxy mechanism analysis with bootstrap CIs", ["fairness/proxy_mechanism.py"]),
    ("7/8 Geospatial redlining analysis (H3)", ["geospatial/redlining_analysis.py"]),
    ("8/8 Retrieval evaluation (BM25 vs dense vs hybrid, + chunking experiment)",
     ["rag/evaluate_retrieval.py"]),
]

LLM_STEPS = [
    ("RAG red-team suite", ["rag/red_team.py"]),
    ("Answerability gate experiment", ["rag/answerability_experiment.py"]),
    ("Adverse action notices (LLM vs deterministic template)", ["fairness/adverse_action.py"]),
]


def run(label, args):
    print(f"\n{'=' * 72}\n{label}...\n{'=' * 72}")
    r = subprocess.run([PY] + args, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"\nStep failed (exit {r.returncode}): {' '.join(args)}")
        sys.exit(1)


def run_spark_stage():
    """Spark feature engineering - under WSL when we're on Windows."""
    label = "3/8 PySpark feature engineering (runs under WSL - see module docstring)"
    print(f"\n{'=' * 72}\n{label}...\n{'=' * 72}")
    if platform.system() == "Windows":
        cmd = ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-c",
               f"cd {WSL_PROJECT} && python3 spark_layer/build_features.py"]
    else:
        cmd = [PY, "spark_layer/build_features.py"]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"\nSpark stage failed (exit {r.returncode}).\n"
              f"On Windows this stage needs WSL with Java 17 and pyspark:\n"
              f"  wsl -d Ubuntu -- sudo apt-get install -y openjdk-17-jdk-headless\n"
              f"  bash scripts/wsl_setup.sh\n"
              f"Or re-run with --skip-spark to reuse an existing feature table.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-spark", action="store_true",
                     help="Reuse the existing database/features table.")
    ap.add_argument("--include-llm-steps", action="store_true",
                     help="Also run the ~60 local Ollama calls (slow).")
    args = ap.parse_args()

    run(*CORE_STEPS[0])
    run(*CORE_STEPS[1])

    if args.skip_spark:
        print("\nSkipping the Spark stage (--skip-spark); reusing database/features.")
        if not (ROOT / "database" / "features").exists():
            sys.exit("No existing feature table found - run without --skip-spark first.")
    else:
        run_spark_stage()

    for label, step in CORE_STEPS[2:]:
        run(label, step)

    if args.include_llm_steps:
        for label, step in LLM_STEPS:
            run(label, step)
    else:
        print("\nSkipped LLM steps (red team, answerability, adverse action) - "
              "re-run with --include-llm-steps.")

    print("\n" + "=" * 72)
    print("Pipeline complete. Launch the dashboard:")
    print("    streamlit run dashboard/streamlit_app.py --server.port 8506 "
          "--server.fileWatcherType none")
    print("=" * 72)


if __name__ == "__main__":
    main()
