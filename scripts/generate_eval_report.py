import json
from collections import Counter
from pathlib import Path

eval_path = Path("data/eval/rag_evaluation.json")
scores_path = Path("data/eval/manual_scores.json")
report_path = Path("data/eval/evaluation_report.md")

results = json.loads(eval_path.read_text(encoding="utf-8"))
manual = json.loads(scores_path.read_text(encoding="utf-8"))

labels = manual["scores"]
rubric = manual["rubric"]
total = len(results)

assert total == 40, "Expected 40 evaluation questions"
assert set(labels) == {str(i) for i in range(1, total + 1)}

counts = Counter(labels.values())
successful = [r for r in results if r["status"] == "success"]

retrieval_hits = sum(
    r["retrieval_hit_at_5"] for r in successful
)

valid_citations = sum(
    r["citation_valid"] for r in successful
)

keyword_scores = [
    r["keyword_recall"]
    for r in successful
    if r["keyword_recall"] is not None
]

weighted_score = sum(
    rubric[label] for label in labels.values()
) / total

report = f"""# Milestone 2 — Task 5: End-to-End RAG Evaluation

## 1. Evaluation Objective

Evaluate the baseline Retrieval-Augmented Generation (RAG)
pipeline on retrieval performance, generated-answer quality,
citation-label validity, and system reliability.

## 2. Evaluation Methodology

- Dataset: 40 ground-truth questions covering 10 research papers.
- Retrieval: ChromaDB vector retrieval, top 5 passages.
- Reranking: Disabled for the baseline.
- Generation: Gemini 3.5 Flash-Lite.
- Citation validation: Checks whether cited source labels
  correspond to retrieved source identifiers.
- Keyword recall: Case-insensitive exact substring matching
  against ground-truth keywords.
- Answer quality: Manual comparison against ground-truth answers.

Manual answer-quality rubric:
- Correct: 1.0 point
- Partially correct: 0.5 points
- Incorrect: 0.0 points

## 3. Evaluation Results

| Metric | Result |
|---|---:|
| Total questions | {total} |
| Successful requests | {len(successful)}/{total} |
| Retrieval Hit@5 | {retrieval_hits / len(successful) * 100:.2f}% |
| Citation label validity | {valid_citations / len(successful) * 100:.2f}% |
| Average keyword recall | {sum(keyword_scores) / len(keyword_scores) * 100:.2f}% |
| Fully correct answers | {counts["correct"]}/{total} |
| Partially correct answers | {counts["partial"]}/{total} |
| Incorrect answers | {counts["incorrect"]}/{total} |
| Fully correct accuracy | {counts["correct"] / total * 100:.2f}% |
| Weighted answer quality | {weighted_score * 100:.2f}% |

## 4. Identified Limitations

### Retrieval granularity

Retrieving the expected paper does not guarantee that the
specific passage needed to answer the question is retrieved.

For example, Question 2 retrieved the correct RAG paper,
but the generated answer omitted important training details.

### Missing information in retrieved passages

Some answers explicitly reported that the retrieved context
did not contain the requested information.

Examples include Questions 8, 13, 16, and 25.

### Keyword-matching limitations

Exact keyword recall can underestimate answer quality when
the generated response uses semantically equivalent wording.

Question 4 was manually classified as correct despite
having relatively low keyword recall.

### Citation validation limitations

Citation-label validity checks source identifier references,
not whether the cited passages substantively support each
claim. It is not equivalent to citation faithfulness.

### Evaluation limitations

Manual answer-quality scores are based on comparison with
the supplied ground-truth answers. They are not independent
fact-checking against every original research paper.

The evaluation uses 40 questions from a fixed research
corpus, so results may not generalize to other documents,
question types, or production workloads.

## 5. Recommended Improvements

1. Evaluate larger retrieval depths, such as top 10.
2. Compare retrieval with and without reranking.
3. Investigate whether relevant passages are indexed
   but ranked below the retrieval cutoff.
4. Improve retrieval for precise formulas and technical details.
5. Add claim-level citation support verification.
6. Re-evaluate answer quality after retrieval improvements.

## 6. Conclusion

The baseline RAG system successfully processed all
40 evaluation questions and achieved strong expected-paper
retrieval performance.

However, the manual evaluation identified incomplete and
incorrect answers, demonstrating that paper-level retrieval
and citation-label validity alone are insufficient measures
of end-to-end RAG answer quality.

The baseline provides a measurable starting point for
retrieval and generation improvements.
"""

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")

print("Report generated successfully!")
print("Saved to:", report_path)
print("Weighted answer quality:", f"{weighted_score * 100:.2f}%")
