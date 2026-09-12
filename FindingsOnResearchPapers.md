# Findings from the Research Papers

## Dataset Overview

| Measure | Result |
| --- | ---: |
| Papers analyzed | 10 |
| Total pages | 218 |
| Extracted words | 121,577 |
| Extracted characters | 799,909 |

## Data Quality Findings

### Consistent Structure

- All papers contain the same six top-level metadata fields.
- Every page record contains `page_number` and `text`.
- Page numbering is sequential for all papers.

### Completeness and Uniqueness

- No missing or null metadata values were detected.
- No pages are empty.
- Paper IDs are unique.
- PDF URLs are unique.

## Known Limitations

The primary issues are PDF text-extraction and formatting artifacts, including:

- Control characters
- Ligatures
- Broken line wraps
- Tables, equations, and figures mixed into the page text

## Structural Observation

Section structure is present implicitly in the extracted text, but it is not currently represented as structured metadata.
