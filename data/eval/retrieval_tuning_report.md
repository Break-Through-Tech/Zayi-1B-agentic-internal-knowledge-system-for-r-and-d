# Milestone 3 — Task 1: Retrieval Accuracy & Reranking

*2026-11-07. Reproduce with `python scripts/run_gt_retrieval_eval.py` (no LLM calls; raw numbers in `gt_retrieval_results.json`).*

## What changed in how we measure

The October evaluation measured retrieval at the **paper** level (was any chunk
from the right paper in the top 5?). That is 95% and hides the real problem:
the right paper is often retrieved without the passage that holds the answer.

We added page labels for all 40 ground-truth questions
(`ground_truth_pages.json`: the page(s) whose text contains the evidence for
each reference answer, with a note per question). A **page hit** means a
chunk from one of those pages was retrieved. That is an upper bound — it does
not check that the chunk itself holds the answer text, which matters slightly
more for small chunks (several per page).

## Results (40 questions)

`p@k` = page-level hit rate in the top k; `paper@5` = October's metric.

| Chunks | Rerank pool | p@1 | p@3 | p@5 | p@10 | paper@5 | Retrieval time / question |
|---|---|---:|---:|---:|---:|---:|---:|
| 512/64 | off (October eval) | 0.28 | 0.63 | 0.78 | 0.83 | 0.95 | ~0.01 s |
| 512/64 | 20 (current app default) | 0.63 | 0.80 | 0.88 | 0.90 | 0.98 | ~1.4 s |
| 512/64 | 40 | 0.58 | 0.83 | 0.90 | 0.95 | 0.98 | ~2.9 s |
| 512/64 | 80 | 0.60 | 0.83 | 0.93 | 0.98 | 1.00 | ~8 s |
| 384/48 | 40 | 0.55 | 0.78 | 0.88 | 0.93 | 0.98 | — |
| 256/32 | 20 | 0.63 | 0.83 | 0.83 | 0.90 | 0.95 | — |

(Times: re-measured separately on a laptop CPU, 10 questions, models already
loaded. The `seconds_per_query` values in the JSON are higher (2.2 / 4.1 /
9.3 s) because that sweep ran alongside other jobs — use them only to compare
pool sizes with each other. Full table incl. all chunk sizes is in the JSON.)

## Findings

1. **Reranking is the biggest win.** On the same 512-token chunks, the
   cross-encoder lifts p@1 from 0.28 to 0.63 and p@5 from 0.78 to 0.88. The app
   (`get_retriever()`, `scripts/ask.py`) already reranks by default, but the
   October answer-quality run (`scripts/evaluate_rag.py`) used `rerank=False`,
   so October's 27/40 correct was measured with weaker retrieval than the app
   actually uses. The final evaluation (Task 5) should use the app's settings.
2. **A larger rerank pool helps a little, at a latency cost.** Pool 20 → 40
   fixes Q8 and Q17 but pushes Q29 out of the top 5 (net p@5 0.88 → 0.90) for
   ~1.5 s more per question; pool 80 also recovers Q13 (the CRAG formula) for
   ~5 s more. These are 1–2 question
   differences on a 40-question set, so treat them as directional. The pool is
   now configurable: `get_retriever(rerank_pool=40)` / `ask.py --rerank-pool 40`.
3. **Smaller chunks did not help.** 256- and 384-token chunks were no better
   than 512 at p@5 with reranking, so 512/64 stays the default.
4. **k:** with pool 40, p@5 = 0.90 and p@10 = 0.95. Going to k=10 doubles the
   context sent to Gemini for two more questions; k=5 stays the default.

## Remaining misses (512/64, pool 40, top 5)

- **Q28 (Ragas "Context Relevance").** The definition chunk (p.3) is in the
  candidate pool (15th by embedding similarity) but the cross-encoder moves it
  only to 13th, below the Ragas title page and survey pages. Not a
  truncation issue: the definition starts ~170 tokens into the chunk, inside
  the reranker's 512-token window.
- **Q39 (DPR vs BM25/ORQA).** Broad question; retrieval returns DPR's results
  pages (pp. 5–9), which discuss the same comparisons, instead of the
  abstract/intro numbers in the reference answer.
- **Q13 (CRAG formula)** — found only with pool 80.
- **Q29 (LoRA rank r vs d=12,288)** — 4th with pool 20, 6th with pools 40/80.

## Suggested next steps

- Pick the default pool (40 is the middle ground) and use the app's retriever
  settings in the Task 5 final evaluation.
- Try a hybrid keyword + embedding retriever (BM25 + dense) for exact-term
  questions like Q13/Q29.
