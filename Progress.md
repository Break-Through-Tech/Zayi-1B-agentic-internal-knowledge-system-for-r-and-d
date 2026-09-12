# Progress Tracker

## Current Status

Tasks 1 and 2 completed.

## Completed

### Task 1: Set Up Development Environment & Review Dataset

- Reviewed the project repository structure and confirmed the provided dataset resources.
- Confirmed that `data/curated_papers_text.json` contains 10 research papers and 218 extracted pages.
- Confirmed that all papers contain the same six top-level metadata fields.
- Confirmed that every page record contains `page_number` and `text`, with sequential page numbering.
- Found no missing or null metadata values, empty pages, duplicate paper IDs, or duplicate PDF URLs.
- Documented known PDF text-extraction artifacts, including control characters, ligatures, broken line wraps, and mixed tables, equations, and figures.
- Observed that section structure is present implicitly in extracted text but is not represented as structured metadata.

### Task 2: Conduct Exploratory Data Analysis (EDA)

- Recorded the analysis in the project notebook.
- Analyzed paper metadata, page-level structure, document lengths, missing values, and formatting quality.
- Confirmed the following dataset measurements:

| Measure | Result |
| --- | ---: |
| Papers analyzed | 10 |
| Total pages | 218 |
| Extracted words | 121,577 |
| Extracted characters | 799,909 |
| Average pages per paper | 21.8 |
| Minimum pages per paper | 8 |
| Maximum pages per paper | 34 |
| Average words per page | 557.69 |

- Found no empty pages.
- Identified one page with fewer than 100 words and four pages with more than 1,000 words.
- Confirmed that all five word-count outliers occur in the RAG survey paper and are primarily reference-heavy pages rather than corrupted or missing data.
- Documented the implication that page boundaries should not automatically be treated as retrieval chunks.
- Recommended preserving paper ID, title, category, and page number during preprocessing and chunking to support accurate retrieval and citations.

## In Progress

- 

## Blockers

- 

## Next Steps

- 

## Notes

- Reference-heavy pages may introduce noise into semantic retrieval.


