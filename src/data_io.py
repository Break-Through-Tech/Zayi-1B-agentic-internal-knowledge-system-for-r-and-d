"""Loading and normalizing the curated papers dataset.

Produces flat page records so downstream steps (cleaning, chunking) don't
depend on the raw JSON layout.
"""

import json
from pathlib import Path

# Path works from repo root and from notebooks/
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "curated_papers_text.json"


def load_papers(path=DEFAULT_DATA_PATH):
    """Load the raw dataset: a list of 10 papers, each with id, title,
    category, pdf_url, total_pages, and pages[{page_number, text}]."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_pages(papers):
    """Flatten papers into one page record per page, carrying the paper
    metadata needed later for citations (paper id, title, page number)."""
    for paper in papers:
        for page in paper["pages"]:
            yield {
                "paper_id": paper["id"],
                "paper_title": paper["title"],
                "category": paper["category"],
                "pdf_url": paper["pdf_url"],
                "page_number": page["page_number"],
                "text": page["text"],
            }


def load_pages(path=DEFAULT_DATA_PATH):
    """Convenience: load and flatten in one call."""
    return list(iter_pages(load_papers(path)))
