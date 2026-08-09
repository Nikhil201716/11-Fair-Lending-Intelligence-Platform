"""
retrieval.py
---------------
Three retrieval strategies over the policy corpus, plus the chunking
strategies used in the chunking experiment:

  * SPARSE  - BM25 (rank_bm25). Strong when the question reuses the
              document's own terminology.
  * DENSE   - sentence-transformers embeddings (all-MiniLM-L6-v2, ~90MB,
              downloaded once from HuggingFace and cached). Strong when
              the question paraphrases the concept without sharing
              vocabulary.
  * HYBRID  - Reciprocal Rank Fusion of the two. RRF is used rather than
              a weighted score blend specifically because BM25 scores and
              cosine similarities are on incomparable scales, and tuning
              a blend weight on the same 20-question set used to evaluate
              would be fitting to the test set.

The point of having all three is that the evaluation harness
(evaluate_retrieval.py) scores them against a labeled question set that
deliberately contains BOTH vocabulary-matching questions (where BM25
should win) and paraphrased questions (where dense should win) - so the
comparison measures something real instead of confirming a preference.

Output (when run directly): a quick smoke query against each strategy.
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def load_corpus() -> list[dict]:
    with open(DATA_DIR / "policy_corpus.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Chunking strategies (for the chunking experiment)
# ---------------------------------------------------------------------
def chunk_by_section(corpus: list[dict]) -> list[dict]:
    """One chunk per policy section - the natural semantic boundary."""
    return [dict(c, chunk_key=c["chunk_id"]) for c in corpus]


def chunk_fixed(corpus: list[dict], size: int, overlap: int = 0) -> list[dict]:
    """Fixed word-count chunks, ignoring section boundaries. Sub-chunks
    keep their parent section_id so retrieval can still be scored against
    the same ground truth."""
    out = []
    for c in corpus:
        words = c["text"].split()
        step = max(1, size - overlap)
        for i, start in enumerate(range(0, len(words), step)):
            piece = words[start:start + size]
            if not piece:
                continue
            out.append(dict(c, text=" ".join(piece), chunk_key=f"{c['chunk_id']}#{i}"))
            if start + size >= len(words):
                break
    return out


CHUNK_STRATEGIES = {
    "section": lambda c: chunk_by_section(c),
    "fixed_60w": lambda c: chunk_fixed(c, 60, 0),
    "fixed_40w_overlap15": lambda c: chunk_fixed(c, 40, 15),
}


# ---------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c["text"] + " " + c["heading"]) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        model = _get_model()
        texts = [f"{c['heading']}. {c['text']}" for c in chunks]
        self.emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        q = _get_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.emb @ q
        order = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order]


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum over retrievers of 1/(rrf_k + rank).

    RRF_K=60 is the value from the original RRF paper, used as-is rather
    than tuned - tuning it on the same 20 questions used for evaluation
    would make the reported hybrid numbers optimistic.
    """
    name = "hybrid_rrf"
    RRF_K = 60

    def __init__(self, chunks: list[dict], sparse=None, dense=None):
        self.chunks = chunks
        self.sparse = sparse or BM25Retriever(chunks)
        self.dense = dense or DenseRetriever(chunks)

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        pool = max(k, 10)
        fused: dict[str, float] = {}
        by_key: dict[str, dict] = {}
        for retriever in (self.sparse, self.dense):
            for rank, (chunk, _) in enumerate(retriever.search(query, pool)):
                key = chunk["chunk_key"]
                by_key[key] = chunk
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(by_key[key], score) for key, score in ranked]


def build_retrievers(chunks: list[dict]) -> dict:
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    return {
        "bm25": sparse,
        "dense": dense,
        "hybrid_rrf": HybridRetriever(chunks, sparse, dense),
    }


if __name__ == "__main__":
    corpus = load_corpus()
    chunks = chunk_by_section(corpus)
    print(f"Corpus: {len(corpus)} sections\n")
    retrievers = build_retrievers(chunks)
    q = "How long does the lender have to notify someone that their application was turned down?"
    print(f"Query: {q}\n")
    for name, r in retrievers.items():
        top = r.search(q, 3)
        print(f"  {name}:")
        for chunk, score in top:
            print(f"    {chunk['chunk_id']:<12} {chunk['heading'][:46]:<48} {score:.4f}")
        print()
