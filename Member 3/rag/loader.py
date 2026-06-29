"""Load every PDF document found in the knowledge base."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag.config import KNOWLEDGE_BASE_DIR


def find_pdf_files(directory: Path = KNOWLEDGE_BASE_DIR) -> list[Path]:
    """Return all PDF files below *directory* in a deterministic order."""
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {directory}")

    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )


def load_pdf_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    """Load all pages from all PDFs in the knowledge base."""
    pdf_files = find_pdf_files(directory)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {directory}")

    documents: list[Document] = []

    for pdf_path in pdf_files:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()

            for page in pages:
                page.metadata["source"] = str(pdf_path.resolve())
                page.metadata["source_filename"] = pdf_path.name
                page.metadata["page_number"] = page.metadata.get("page", 0) + 1
                page.metadata["document_id"] = pdf_path.stem

            documents.extend(pages)

        except Exception as e:
            print(f"Skipping {pdf_path.name}: {e}")

    return documents
