"""Ask a question of the 10 papers from the command line (Task 13).

Prints the grounded answer followed by the numbered sources its bracket
citations refer to. `--retrieve-only` skips the LLM (no API key needed) and
shows what the retriever would feed it — handy for debugging retrieval.

Run from the repo root:
    python scripts/ask.py "What does LoRA freeze during fine-tuning?"
    python scripts/ask.py "..." --k 8 --no-rerank --provider openrouter --model <slug>
    python scripts/ask.py "..." --retrieve-only
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from src.rag import (
    PROVIDERS,
    MissingConfigError,
    build_rag_chain,
    get_llm,
    page_label,
    source_label,
    sources_from,
)

PREVIEW_CHARS = 240


def _print_sources(sources):
    for s in sources:
        print(f"[{source_label(s['n'])}] {s['paper_title']} (arXiv:{s['paper_id']}), {page_label(s)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5, help="excerpts to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="skip cross-encoder reranking")
    parser.add_argument("--provider", choices=PROVIDERS, help="default: LLM_PROVIDER or gemini")
    parser.add_argument("--model", help="model name (default from .env / provider default)")
    parser.add_argument(
        "--retrieve-only", action="store_true", help="show retrieved excerpts; no LLM call"
    )
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question must not be empty")
    if args.k < 1:
        parser.error("--k must be at least 1")

    llm = None
    if not args.retrieve_only:
        try:  # before loading the retriever, so a missing key fails fast
            llm = get_llm(args.provider, args.model)
        except (MissingConfigError, ValueError) as err:  # missing key, bad provider
            sys.exit(f"error: {err}")

    from src.retriever import get_retriever

    retriever = get_retriever(k=args.k, rerank=not args.no_rerank)

    if args.retrieve_only:
        docs = retriever.invoke(args.question)
        if not docs:
            print("No excerpts retrieved.")
        for source, doc in zip(sources_from(docs), docs):
            scores = ", ".join(
                f"{key}={doc.metadata[key]}"
                for key in ("similarity", "rerank_score")
                if key in doc.metadata
            )
            _print_sources([source])
            print(f"    {scores}  chunk={source['chunk_id']}")
            preview = " ".join(doc.page_content.split())
            if len(preview) > PREVIEW_CHARS:
                preview = preview[:PREVIEW_CHARS] + "..."
            print(f"    {preview}\n")
        return

    try:
        result = build_rag_chain(retriever, llm).invoke(args.question)
    except Exception as err:  # auth, unknown model, quota, network: one line, not a traceback
        sys.exit(f"error: LLM call failed ({type(err).__name__}): {err}")
    print(result["answer"].strip())
    if result["sources"]:
        print("\nSources:")
        _print_sources(result["sources"])


if __name__ == "__main__":
    main()
