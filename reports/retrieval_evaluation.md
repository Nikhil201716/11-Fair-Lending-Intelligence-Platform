# Retrieval Evaluation: BM25 vs. Dense vs. Hybrid

20 labeled questions (5 hard, 7 lexical, 8 paraphrase).

## Overall

| Method | MRR | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|---|
| bm25 | 0.796 | 75.0% | 85.0% | 90.0% |
| dense | 0.917 | 90.0% | 95.0% | 95.0% |
| hybrid_rrf | 0.875 | 80.0% | 95.0% | 95.0% |

## By question type (MRR)

| Method | hard | lexical | paraphrase |
|---|---|---|---|
| bm25 | 1.000 | 1.000 | 0.490 |
| dense | 1.000 | 1.000 | 0.792 |
| hybrid_rrf | 1.000 | 1.000 | 0.688 |

## Chunking experiment (hybrid retriever)

| Strategy | Chunks | MRR | Recall@1 |
|---|---|---|---|
| section | 15 | 0.875 | 80.0% |
| fixed_60w | 25 | 0.875 | 85.0% |
| fixed_40w_overlap15 | 39 | 0.833 | 75.0% |
