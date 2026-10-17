from src.rag import create_rag_pipeline


pipeline = create_rag_pipeline(
    retriever_kwargs={"k": 5, "rerank": True}
)

result = pipeline.answer("How does LoRA reduce trainable parameters?")

print("ANSWER:")
print(result.answer)

print("\nSOURCES:")
for document in result.source_documents:
    metadata = document.metadata
    print(
        f"- {metadata['paper_title']}, "
        f"pages {metadata.get('page_start')}-{metadata.get('page_end')}"
    )
