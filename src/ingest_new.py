import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from unstructured.partition.auto import partition

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Element categories we consider real, usable content.
# Everything else (Image, Figure, FigureCaption, PageBreak, Header, Footer)
# gets dropped before chunking — this is what fixes the diagram/token-soup noise.
KEEP_CATEGORIES = {
    "NarrativeText",
    "Title",
    "ListItem",
    "Table",
    "UncategorizedText",
}


def load_documents():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if not (filename.endswith(".txt") or filename.endswith(".pdf")):
            continue

        print(f"Partitioning {filename}...")
        elements = partition(filename=path)

        dropped_by_category = 0
        for el in elements:
            category = el.category if hasattr(el, "category") else "Unknown"
            if category not in KEEP_CATEGORIES:
                dropped_by_category += 1
                continue

            text = str(el).strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "category": category},
                )
            )

        print(f"  Kept elements: {sum(1 for el in elements if getattr(el, 'category', None) in KEEP_CATEGORIES)}, "
              f"Dropped (non-content, e.g. figures/images): {dropped_by_category}")

    return documents


def is_reference_chunk(text: str) -> bool:
    """Heuristic filter to drop bibliography/citation-style chunks."""
    stripped = text.strip()
    if re.match(r'^\[?\d+\]?\.?\s', stripped):
        return True
    markers = [
        "arXiv preprint", "In Empirical Methods", "Computational linguistics",
        "Advances in Neural Information Processing", "ISSN", "doi.org",
    ]
    if any(m in text for m in markers):
        return True
    link_count = text.count("http") + text.count("ISSN") + text.count("doi")
    return link_count > 3


def main():
    print("Loading and partitioning documents...")
    raw_docs = load_documents()
    print(f"\nTotal content elements collected: {len(raw_docs)}")

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} raw chunks")

    chunks = [c for c in chunks if not is_reference_chunk(c.page_content)]
    print(f"Kept {len(chunks)} content chunks after reference filtering")

    print("Embedding and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"Done. Vector store saved to '{PERSIST_DIR}'")


if __name__ == "__main__":
    main()