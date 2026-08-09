"""
answer.py
------------
Grounded question answering over the policy corpus: retrieve with the
chosen strategy, then have the local Ollama model answer USING ONLY the
retrieved text.

Guardrails, in the order they run:

  1. RETRIEVAL FLOOR - if the best retrieval score is below a floor, the
     question is refused outright rather than answered from a weak or
     irrelevant chunk. "I don't have a policy covering that" is a correct
     answer for a compliance assistant; a confident guess is not.

  2. GROUNDING CHECK (hallucination detection) - the generated answer is
     checked for content-word overlap with the retrieved context. An
     answer that shares almost no vocabulary with the source it supposedly
     summarizes is flagged as ungrounded. This is a cheap, deterministic
     proxy - it catches fabrication that ignores the context entirely, and
     will NOT catch a fluent answer that subtly misstates the source. That
     limitation is stated here rather than left for a reader to discover.

  3. INJECTION SCREEN - the retrieved context is scanned for instruction-
     like text before it reaches the prompt, so a document that says
     "ignore your instructions" is neutralized rather than obeyed.

Output when run directly: answers to a few sample questions.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.retrieval import load_corpus, build_retrievers, CHUNK_STRATEGIES  # noqa: E402

OLLAMA_MODEL = "qwen2.5:0.5b"

# Below this hybrid/dense score, refuse rather than answer.
RETRIEVAL_FLOOR = 0.25
# Below this share of answer content-words appearing in the context, flag.
GROUNDING_FLOOR = 0.45

INJECTION_PATTERNS = [
    r"ignore (all |your |previous )*instructions",
    r"disregard (the |all |your )*(above|previous|prior)",
    r"you are now",
    r"system prompt",
    r"reveal .*(prompt|instructions)",
    r"approve (all|every|the) (loan|application)",
]

PROMPT_TEMPLATE = """You are a compliance assistant for a lender. Answer the question using ONLY the policy text provided below. If the policy text does not answer the question, say "The provided policy does not cover this."

POLICY TEXT:
{context}

QUESTION: {question}

Answer in two sentences or fewer, using only the policy text above."""

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "that", "this",
    "it", "as", "be", "by", "with", "at", "from", "must", "may", "not", "no", "if", "than",
    "any", "all", "was", "were", "will", "can", "has", "have", "their", "its", "which", "when",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def screen_for_injection(text: str) -> list[str]:
    """Return any instruction-like patterns found in retrieved context."""
    return [p for p in INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]


def check_grounding(answer: str, context: str) -> tuple[float, bool]:
    """Share of the answer's content words that appear in the context."""
    a_words = _content_words(answer)
    if not a_words:
        return 0.0, False
    overlap = len(a_words & _content_words(context)) / len(a_words)
    return round(overlap, 4), overlap >= GROUNDING_FLOOR


def ask(question: str, retriever, k: int = 3, model: str = OLLAMA_MODEL) -> dict:
    import ollama

    hits = retriever.search(question, k)
    top_score = hits[0][1] if hits else 0.0

    result = {
        "question": question,
        "retrieved": [{"chunk_id": c["chunk_id"], "heading": c["heading"], "score": round(s, 4)}
                      for c, s in hits],
        "top_score": round(top_score, 4),
        "refused": False, "refusal_reason": None,
        "injection_patterns_found": [], "answer": None,
        "grounding_score": None, "grounded": None,
    }

    if top_score < RETRIEVAL_FLOOR:
        result.update(refused=True,
                       refusal_reason=f"top retrieval score {top_score:.3f} below floor {RETRIEVAL_FLOOR}")
        return result

    context = "\n\n".join(f"[{c['chunk_id']}] {c['heading']}: {c['text']}" for c, _ in hits)

    found = screen_for_injection(context)
    if found:
        result.update(refused=True, injection_patterns_found=found,
                       refusal_reason="instruction-like text detected in retrieved policy context")
        return result

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)}],
        options={"temperature": 0.0, "num_predict": 160},
    )
    answer = response["message"]["content"].strip()
    score, grounded = check_grounding(answer, context)
    result.update(answer=answer, grounding_score=score, grounded=grounded)
    return result


def get_default_retriever():
    corpus = load_corpus()
    chunks = CHUNK_STRATEGIES["section"](corpus)
    # dense, not hybrid - see reports/retrieval_evaluation.json, dense
    # measurably outperformed hybrid RRF on this corpus
    return build_retrievers(chunks)["dense"]


if __name__ == "__main__":
    retriever = get_default_retriever()
    for q in ["How quickly must an adverse action notice be delivered?",
              "Can a model be discriminatory without using a protected attribute?",
              "What is the company's policy on cryptocurrency trading desks?"]:
        r = ask(q, retriever)
        print(f"\nQ: {q}")
        print(f"   top chunk: {r['retrieved'][0]['chunk_id']} (score {r['top_score']})")
        if r["refused"]:
            print(f"   REFUSED: {r['refusal_reason']}")
        else:
            print(f"   A: {r['answer'][:200]}")
            print(f"   grounding: {r['grounding_score']} -> {'grounded' if r['grounded'] else 'FLAGGED'}")
