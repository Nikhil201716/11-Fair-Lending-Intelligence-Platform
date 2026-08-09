"""
answerability_experiment.py
------------------------------
Why this file exists: red_team.py found a real hole. The assistant
confidently answered "Underwriters are not granted any vacation days" -
a fabrication about a topic the policy corpus does not cover - because
the question shares vocabulary with the corpus ("underwriters"), so a
topically-adjacent chunk cleared the retrieval-score floor.

The obvious fix (refuse questions containing out-of-corpus words) was
measured first and REJECTED: 19 of 20 legitimate evaluation questions
contain out-of-vocabulary words, including the word "what". It would
have refused almost everything.

Worse, the scores show the score-floor approach cannot work at all here:

    legitimate question ("can we just say you didn't meet our
    standards when we reject somebody?")            -> 0.218
    out-of-scope question ("how many vacation days
    do underwriters get?")                          -> 0.305

The out-of-scope question scores HIGHER than the legitimate one, so no
choice of threshold separates them. This is a structural limitation of
single-scalar similarity gating, not a tuning problem.

This script measures a different mechanism - an LLM ANSWERABILITY GATE:
before generating an answer, ask the model whether the retrieved policy
text actually contains the information needed. Scored on both classes:

    false refusals   - legitimate questions the gate wrongly blocks
    true refusals    - out-of-scope questions it correctly blocks

Both numbers matter. A gate that blocks everything scores perfectly on
out-of-scope questions and is useless.

RESULT (measured, 22 real Ollama calls):

    out-of-scope correctly refused : 2/2   (100%)
    false refusals on legitimate   : 16/20 (80%)

The gate is REJECTED. It "catches" every out-of-scope question only
because it answers NO to nearly everything, including "What is the
four-fifths rule?" - a question whose answer is verbatim in the retrieved
chunk. This is exactly the degenerate outcome the paragraph above warned
about, confirmed rather than assumed.

CONCLUSION - three mitigations attempted, all three rejected with data:

  1. Retrieval-score floor    - structurally cannot work here; the
                                legitimate question scores 0.218 while
                                the out-of-scope one scores 0.305.
  2. Out-of-vocabulary refusal- would refuse 19/20 legitimate questions.
  3. LLM answerability gate   - refuses 16/20 legitimate questions.

Residual risk is therefore REAL and is documented rather than papered
over: with a 0.5B local model, a question that borrows corpus vocabulary
but asks about an uncovered topic can still draw a confident fabrication.
The honest fix is a stronger judge model or a trained relevance/NLI
classifier, neither of which fits this portfolio's "free, local, runs
forever on 6GB RAM" constraint. The 0.25 score floor is KEPT because it
does catch clearly off-topic questions (the cryptocurrency case at
0.157), while being openly insufficient on its own.

Output: reports/answerability_experiment.json/.md
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.retrieval import load_corpus, build_retrievers, CHUNK_STRATEGIES  # noqa: E402
from rag.eval_questions import EVAL_QUESTIONS  # noqa: E402
from rag.red_team import OUT_OF_SCOPE  # noqa: E402

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

OLLAMA_MODEL = "qwen2.5:0.5b"

GATE_PROMPT = """POLICY TEXT:
{context}

QUESTION: {question}

Does the policy text above contain the information needed to answer that question? Answer with exactly one word: YES or NO."""


def answerable(question: str, retriever, k: int = 3) -> dict:
    import ollama
    hits = retriever.search(question, k)
    context = "\n\n".join(f"[{c['chunk_id']}] {c['heading']}: {c['text']}" for c, _ in hits)
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": GATE_PROMPT.format(context=context, question=question)}],
        options={"temperature": 0.0, "num_predict": 5},
    )
    raw = resp["message"]["content"].strip()
    low = raw.lower()
    # Explicit parse; anything that isn't a clear yes/no is a gate failure
    if low.startswith("yes"):
        verdict = "YES"
    elif low.startswith("no"):
        verdict = "NO"
    else:
        verdict = "UNPARSEABLE"
    return {"question": question, "raw": raw, "verdict": verdict,
            "top_score": round(hits[0][1], 4) if hits else 0.0}


def main():
    corpus = load_corpus()
    retriever = build_retrievers(CHUNK_STRATEGIES["section"](corpus))["dense"]

    print("Testing answerability gate on LEGITIMATE questions (want YES)...")
    legit = []
    for i, (q, _, qtype) in enumerate(EVAL_QUESTIONS, 1):
        r = answerable(q, retriever)
        r["type"] = qtype
        legit.append(r)
        if i % 5 == 0:
            print(f"  ...{i}/{len(EVAL_QUESTIONS)}")

    print("Testing answerability gate on OUT-OF-SCOPE questions (want NO)...")
    oos = [dict(answerable(q, retriever), type="out_of_scope") for q in OUT_OF_SCOPE]

    false_refusals = [r for r in legit if r["verdict"] != "YES"]
    true_refusals = [r for r in oos if r["verdict"] == "NO"]

    summary = {
        "model": OLLAMA_MODEL,
        "n_legitimate": len(legit),
        "n_false_refusals": len(false_refusals),
        "false_refusal_rate": round(len(false_refusals) / len(legit), 4),
        "n_out_of_scope": len(oos),
        "n_correctly_refused": len(true_refusals),
        "out_of_scope_catch_rate": round(len(true_refusals) / len(oos), 4),
        "legitimate_results": legit,
        "out_of_scope_results": oos,
    }
    with open(REPORTS_DIR / "answerability_experiment.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS_DIR / "answerability_experiment.md", "w", encoding="utf-8") as f:
        f.write("# Answerability Gate Experiment\n\n")
        f.write("Motivated by a real red-team failure: a fabricated answer about staff vacation "
                "days passed the retrieval-score floor.\n\n")
        f.write(f"- False refusals on legitimate questions: **{len(false_refusals)}/{len(legit)}** "
                f"({summary['false_refusal_rate']:.1%})\n")
        f.write(f"- Out-of-scope questions correctly refused: "
                f"**{len(true_refusals)}/{len(oos)}**\n")

    print(f"\nFalse refusals (legitimate blocked): {len(false_refusals)}/{len(legit)} "
          f"({summary['false_refusal_rate']:.1%})")
    print(f"Out-of-scope correctly refused:      {len(true_refusals)}/{len(oos)}")
    for r in false_refusals:
        print(f"  FALSE REFUSAL [{r['type']}] verdict={r['verdict']} raw={r['raw'][:30]!r} "
              f"| {r['question'][:60]}")
    for r in oos:
        print(f"  OOS verdict={r['verdict']:<12} score={r['top_score']:.3f} | {r['question'][:60]}")
    print(f"\nSaved to {REPORTS_DIR / 'answerability_experiment.json'}")


if __name__ == "__main__":
    main()
