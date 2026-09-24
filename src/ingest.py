import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from unstructured.partition.auto import partition

# Load .env from the project root (one level up from src/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

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

        kept, dropped = 0, 0
        for el in elements:
            category = getattr(el, "category", "Unknown")
            if category not in KEEP_CATEGORIES:
                dropped += 1
                continue
            text = str(el).strip()
            if not text:
                continue
            documents.append(Document(page_content=text, metadata={"source": filename, "category": category}))
            kept += 1

        print(f"  Kept: {kept}, Dropped (figures/images/etc): {dropped}")

    return documents


def is_reference_chunk(text: str) -> bool:
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

    print("Connecting to Supabase...")
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Embedding and storing in Supabase (pgvector)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents",
    )

    print("Done. Chunks stored in Supabase 'documents' table.")


if __name__ == "__main__":
    main()