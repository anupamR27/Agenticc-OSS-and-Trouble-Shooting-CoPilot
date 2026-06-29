"""Build the FAISS index from every PDF in the knowledge base."""

from rag.chunker import chunk_documents
from rag.config import VECTOR_DB_DIR
from rag.loader import load_pdf_documents
from rag.vector_store import build_vector_store, save_vector_store


def build_index() -> None:
    """Build a searchable FAISS vector database from all PDFs."""

    print("\n========== Building Knowledge Base ==========\n")

    # Step 1: Load PDF pages
    documents = load_pdf_documents()
    print(f"✓ Loaded {len(documents)} PDF pages")

    # Step 2: Split into chunks
    chunks = chunk_documents(documents)
    print(f"✓ Created {len(chunks)} text chunks")

    # Step 3: Generate embeddings and build FAISS index
    print("✓ Generating embeddings...")
    vector_store = build_vector_store(chunks)

    # Step 4: Save locally
    save_vector_store(vector_store)

    print("\n========== Index Build Complete ==========")
    print(f"PDF Pages Indexed : {len(documents)}")
    print(f"Chunks Created    : {len(chunks)}")
    print(f"Saved To          : {VECTOR_DB_DIR}")
    print("==========================================\n")


if __name__ == "__main__":
    build_index()