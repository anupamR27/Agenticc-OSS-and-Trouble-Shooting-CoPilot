"""Create, save, and load the local FAISS vector database."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL_NAME, VECTOR_DB_DIR

# Cache the embedding model so it is only loaded once
_embeddings: HuggingFaceEmbeddings | None = None


def create_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached sentence-transformers embedding model."""
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def build_vector_store(documents: list[Document]) -> FAISS:
    """Embed document chunks and construct an in-memory FAISS index."""
    if not documents:
        raise ValueError("Cannot build a vector store without document chunks.")

    return FAISS.from_documents(
        documents=documents,
        embedding=create_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def save_vector_store(
    vector_store: FAISS,
    directory: Path = VECTOR_DB_DIR,
) -> None:
    """Persist the FAISS index and metadata locally."""
    directory.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(directory))

    print(f"\n✓ Vector database saved to:\n{directory}")


def load_vector_store(directory: Path = VECTOR_DB_DIR) -> FAISS:
    """Load a previously generated FAISS index."""
    required_files = (directory / "index.faiss", directory / "index.pkl")

    missing_files = [path.name for path in required_files if not path.exists()]

    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"FAISS index is missing {missing} in {directory}.\n"
            "Run `python -m rag.build_index` first."
        )

    return FAISS.load_local(
        str(directory),
        embeddings=create_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,

        # Safe because the index is generated locally by this project.
        # Never load FAISS indexes from untrusted sources.
        allow_dangerous_deserialization=True,
    )