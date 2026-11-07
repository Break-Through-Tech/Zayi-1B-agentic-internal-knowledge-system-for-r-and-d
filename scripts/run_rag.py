"""Minimal example of calling the RAG pipeline from Python.

For the command-line tool (options, retrieve-only mode) use scripts/ask.py.
Run from the repo root: python scripts/run_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import answer_question, page_label, source_label

result = answer_question("How does LoRA reduce trainable parameters?")

print("ANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(f"- [{source_label(source['n'])}] {source['paper_title']}, {page_label(source)}")
