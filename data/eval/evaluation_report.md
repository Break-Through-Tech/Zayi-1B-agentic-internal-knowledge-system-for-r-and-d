# Milestone 2 — Task 5: End-to-End RAG Evaluation

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
| Total questions | 40 |
| Successful requests | 40/40 |
| Retrieval Hit@5 | 95.00% |
| Citation label validity | 100.00% |
| Average keyword recall | 68.25% |
| Fully correct answers | 27/40 |
| Partially correct answers | 9/40 |
| Incorrect answers | 4/40 |
| Fully correct accuracy | 67.50% |
| Weighted answer quality | 78.75% |

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
