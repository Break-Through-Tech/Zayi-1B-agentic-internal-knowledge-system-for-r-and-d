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


# Task 3's cleaned output (flat records with `cleaned_text`); adapter so the
# chunking pipeline can consume it instead of the interim cleaner once the
# team settles on a canonical cleaned file.
TEAM_CLEANED_PATH = DEFAULT_DATA_PATH.parent / "processed" / "cleaned_papers.json"


def load_team_cleaned_pages(path=TEAM_CLEANED_PATH):
    """Team-cleaned records mapped to the same page-record shape the chunker
    expects. These records are already cleaned — pass them straight to
    `build_chunks`, not through `clean_pages`."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [
        {
            "paper_id": r["paper_id"],
            "paper_title": r["title"],
            "category": r["category"],
            "pdf_url": r["pdf_url"],
            "page_number": r["page_number"],
            "text": r["cleaned_text"],
            "cleaner": "team-task3-v1",
        }
        for r in records
    ]
