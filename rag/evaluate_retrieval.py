"""
evaluate_retrieval.py
------------------------
Scores BM25 vs. dense vs. hybrid retrieval against the labeled question
set in eval_questions.py, and runs the chunking-strategy experiment.

Metrics:
  recall@k - did the correct section appear anywhere in the top k
  MRR      - mean reciprocal rank of the correct section (rewards putting
             the right answer FIRST, not merely somewhere in the top 5,
             which is what actually matters when the top chunk is what
             gets fed to the LLM)

Results are broken out BY QUESTION TYPE (lexical / paraphrase / hard) as
well as overall, because a single blended number hides the entire point:
the methods have different strengths, and the reason to pay for hybrid
retrieval is what happens on the paraphrase bucket.

Output: reports/retrieval_evaluation.json/.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.retrieval import load_corpus, build_retrievers, CHUNK_STRATEGIES  # noqa: E402
from rag.eval_questions import EVAL_QUESTIONS  # noqa: E402

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

K_VALUES = (1, 3, 5)


def parent_id(chunk: dict) -> str:
    """Map any chunk (including fixed-size sub-chunks) back to its section."""
    return chunk["chunk_id"]


def score_retriever(retriever, questions) -> dict:
    per_type = defaultdict(lambda: {"n": 0, "rr": 0.0, **{f"hit@{k}": 0 for k in K_VALUES}})
    overall = {"n": 0, "rr": 0.0, **{f"hit@{k}": 0 for k in K_VALUES}}
    details = []

    for question, correct_id, qtype in questions:
        results = retriever.search(question, max(K_VALUES))
        # de-duplicate to section level, preserving rank order
        seen, ranked_ids = set(), []
        for chunk, _ in results:
            pid = parent_id(chunk)
            if pid not in seen:
                seen.add(pid)
                ranked_ids.append(pid)

        rank = ranked_ids.index(correct_id) + 1 if correct_id in ranked_ids else None
        rr = 1.0 / rank if rank else 0.0

        for bucket in (overall, per_type[qtype]):
            bucket["n"] += 1
            bucket["rr"] += rr
            for k in K_VALUES:
                if rank and rank <= k:
                    bucket[f"hit@{k}"] += 1

        details.append({"question": question, "type": qtype, "correct_section": correct_id,
                        "retrieved_rank": rank, "top_result": ranked_ids[0] if ranked_ids else None})

    def finalize(b):
        return {"n": b["n"], "mrr": round(b["rr"] / b["n"], 4),
                **{f"recall@{k}": round(b[f"hit@{k}"] / b["n"], 4) for k in K_VALUES}}

    return {"overall": finalize(overall),
            "by_type": {t: finalize(b) for t, b in sorted(per_type.items())},
            "details": details}


def main():
    corpus = load_corpus()

    # --- main comparison, on the natural section chunking ---
    section_chunks = CHUNK_STRATEGIES["section"](corpus)
    retrievers = build_retrievers(section_chunks)

    results = {}
    for name, retriever in retrievers.items():
        results[name] = score_retriever(retriever, EVAL_QUESTIONS)
        o = results[name]["overall"]
        print(f"{name:<12} MRR={o['mrr']:.3f}  R@1={o['recall@1']:.3f}  "
              f"R@3={o['recall@3']:.3f}  R@5={o['recall@5']:.3f}")

    # --- chunking experiment (hybrid retriever held constant) ---
    print("\nChunking experiment (hybrid retriever):")
    chunking = {}
    for strategy_name, strategy in CHUNK_STRATEGIES.items():
        chunks = strategy(corpus)
        r = build_retrievers(chunks)["hybrid_rrf"]
        scored = score_retriever(r, EVAL_QUESTIONS)
        chunking[strategy_name] = {"n_chunks": len(chunks), **scored["overall"]}
        print(f"  {strategy_name:<22} n_chunks={len(chunks):<4} MRR={scored['overall']['mrr']:.3f}  "
              f"R@1={scored['overall']['recall@1']:.3f}")

    summary = {
        "n_questions": len(EVAL_QUESTIONS),
        "question_type_counts": {t: sum(1 for _, _, tt in EVAL_QUESTIONS if tt == t)
                                  for t in sorted({tt for _, _, tt in EVAL_QUESTIONS})},
        "retrieval_comparison": {k: {"overall": v["overall"], "by_type": v["by_type"]}
                                  for k, v in results.items()},
        "chunking_experiment": chunking,
        "per_question_detail": {k: v["details"] for k, v in results.items()},
    }
    with open(REPORTS_DIR / "retrieval_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "retrieval_evaluation.md", "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation: BM25 vs. Dense vs. Hybrid\n\n")
        f.write(f"{len(EVAL_QUESTIONS)} labeled questions "
                f"({', '.join(f'{v} {k}' for k, v in summary['question_type_counts'].items())}).\n\n")
        f.write("## Overall\n\n| Method | MRR | Recall@1 | Recall@3 | Recall@5 |\n|---|---|---|---|---|\n")
        for name, r in results.items():
            o = r["overall"]
            f.write(f"| {name} | {o['mrr']:.3f} | {o['recall@1']:.1%} | "
                    f"{o['recall@3']:.1%} | {o['recall@5']:.1%} |\n")
        f.write("\n## By question type (MRR)\n\n| Method | "
                + " | ".join(sorted(summary["question_type_counts"])) + " |\n")
        f.write("|---" * (len(summary["question_type_counts"]) + 1) + "|\n")
        for name, r in results.items():
            row = [f"{r['by_type'].get(t, {}).get('mrr', 0):.3f}"
                   for t in sorted(summary["question_type_counts"])]
            f.write(f"| {name} | " + " | ".join(row) + " |\n")
        f.write("\n## Chunking experiment (hybrid retriever)\n\n"
                "| Strategy | Chunks | MRR | Recall@1 |\n|---|---|---|---|\n")
        for name, c in chunking.items():
            f.write(f"| {name} | {c['n_chunks']} | {c['mrr']:.3f} | {c['recall@1']:.1%} |\n")

    print(f"\nSaved to {REPORTS_DIR / 'retrieval_evaluation.json'}")


if __name__ == "__main__":
    main()
