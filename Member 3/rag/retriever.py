"""Retrieve relevant PDF chunks from the saved FAISS index."""

from pathlib import Path
from typing import TypedDict

from rag.config import DEFAULT_TOP_K
from rag.vector_store import load_vector_store


class RetrievalResult(TypedDict):
    retrieval_score: float
    source_pdf: str
    document_id: str | None
    page_number: int | None
    text: str


def retrieve_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query: User query or RCA summary.
        top_k: Number of chunks to retrieve.
        verbose: If True, print retrieved chunks for debugging.

    Returns:
        A list of retrieval results with metadata.
    """

    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    vector_store = load_vector_store()

    matches = vector_store.similarity_search_with_score(
        query.strip(),
        k=top_k,
    )

    results: list[RetrievalResult] = []

    for rank, (document, score) in enumerate(matches, start=1):

        source = document.metadata.get("source_filename")

        if not source:
            source = Path(
                str(document.metadata.get("source", "Unknown"))
            ).name

        zero_based_page = document.metadata.get("page")

        page_number = (
            int(zero_based_page) + 1
            if zero_based_page is not None
            else None
        )

        result: RetrievalResult = {
            "retrieval_score": float(score),
            "source_pdf": source,
            "document_id": document.metadata.get("document_id"),
            "page_number": page_number,
            "text": document.page_content,
        }

        results.append(result)

        if verbose:
            print("\n" + "=" * 60)
            print(f"Result {rank}")
            print("=" * 60)
            print(f"Retrieval Score : {result['retrieval_score']:.4f}")
            print(f"Document        : {result['document_id']}")
            print(f"Source PDF      : {result['source_pdf']}")
            print(f"Page            : {result['page_number'] or 'Unknown'}")
            print("\nRetrieved Text\n")
            print(result["text"][:500])

    return results