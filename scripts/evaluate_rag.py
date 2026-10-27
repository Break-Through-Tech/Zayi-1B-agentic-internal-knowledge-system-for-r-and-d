import argparse
import json
import re
import time
from pathlib import Path

from src.rag import build_rag_chain, get_llm, validate_citations
from src.retriever import get_retriever


def evaluate(limit):
    dataset_path = Path("data/ground_truth_qa.json")
    output_path = Path("data/eval/rag_evaluation.json")

    questions = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = questions[:limit]

    print("Loading retriever...")
    retriever = get_retriever(k=5, rerank=False)

    print("Loading Gemini...")
    llm = get_llm()

    chain = build_rag_chain(retriever, llm)
    results = []

    for i, item in enumerate(questions, start=1):
        print(f"\nEvaluating question {i}/{len(questions)}")

        record = {
            "question": item["question"],
            "expected_answer": item["ground_truth_answer"],
            "expected_paper_id": item["paper_id"],
        }

        try:
            response = chain.invoke(item["question"])

            answer = response["answer"]
            sources = response["sources"]

            expected_paper_found = any(
                source["paper_id"] == item["paper_id"]
                for source in sources
            )

            citation_check = validate_citations(answer, sources)

            keywords = item.get("key_keywords", [])
            matched_keywords = [
                keyword for keyword in keywords
                if keyword.lower() in answer.lower()
            ]

            record.update({
                "generated_answer": answer,
                "sources": sources,
                "retrieval_hit_at_5": expected_paper_found,
                "citation_valid": citation_check["valid"],
                "citation_count": len(citation_check["cited_sources"]),
                "keyword_recall": (
                    round(len(matched_keywords) / len(keywords), 3)
                    if keywords else None
                ),
                "matched_keywords": matched_keywords,
                "status": "success",
            })

            print("Answer:", answer)
            print("Expected paper retrieved:", expected_paper_found)
            print("Citation valid:", citation_check["valid"])

        except Exception as error:
            record["status"] = "error"
            record["error"] = str(error)
            print("Error:", error)

        results.append(record)

        # Save after each question so progress survives interruptions.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Reduce the risk of hitting API rate limits.
        if i < len(questions):
            time.sleep(5)

    successful = [r for r in results if r["status"] == "success"]

    print("\nEVALUATION SUMMARY")
    print("Total:", len(results))
    print("Successful:", len(successful))
    print("Errors:", len(results) - len(successful))

    if successful:
        print(
            "Retrieval Hit@5:",
            round(
                sum(r["retrieval_hit_at_5"] for r in successful)
                / len(successful), 3
            ),
        )
        print(
            "Citation validity:",
            round(
                sum(r["citation_valid"] for r in successful)
                / len(successful), 3
            ),
        )

    print("Results saved to:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.limit <= 40:
        parser.error("--limit must be between 1 and 40")

    evaluate(args.limit)
