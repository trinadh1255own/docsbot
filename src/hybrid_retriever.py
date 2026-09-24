import os
import pickle
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from sentence_transformers import CrossEncoder

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BM25_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "bm25_index.pkl")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base")

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)


def build_bm25_index():
    """Run once (or after re-ingesting) to build a BM25 index from all chunks in Supabase.

    Supabase's REST API caps a single select() at 1000 rows, so we page through
    the table in batches until a batch comes back empty.
    """
    all_rows = []
    batch_size = 1000
    offset = 0

    while True:
        response = (
            supabase_client.table("documents")
            .select("content, metadata")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        offset += batch_size

    documents = [row["content"] for row in all_rows]
    metadatas = [row["metadata"] for row in all_rows]

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents, "metadatas": metadatas}, f)

    print(f"Built BM25 index over {len(documents)} chunks")


def bm25_search(question: str, k: int = 6):
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    documents = data["documents"]

    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Rank document indices by BM25 score, descending (higher = more relevant)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [(documents[i], scores[i]) for i in ranked_indices]


def vector_search(question: str, k: int = 6):
    # SupabaseVectorStore uses similarity_search_with_relevance_scores, not
    # similarity_search_with_score (that's Chroma's method name). This returns
    # a relevance score where higher = more similar, same direction as BM25.
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [(doc.page_content, score) for doc, score in results]


def reciprocal_rank_fusion(bm25_results, vector_results, k: int = 60):
    """
    RRF formula: score(doc) = sum over each ranking list of 1 / (k + rank)
    Lower rank number (1st place) contributes more than a lower rank (10th place).
    k=60 is the standard constant from the original RRF paper — dampens the effect
    of any single list's top result dominating.
    """
    scores = {}

    for rank, (text, _) in enumerate(bm25_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)

    for rank, (text, _) in enumerate(vector_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)

    # Sort by fused score, descending (higher fused score = more relevant)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(question: str, k: int = 4):
    bm25_results = bm25_search(question, k=6)
    vector_results = vector_search(question, k=6)

    fused = reciprocal_rank_fusion(bm25_results, vector_results)

    return fused[:k]


def rerank(question: str, candidates: list, top_k: int = 4):
    """
    candidates: list of (text, fused_score) tuples from hybrid_search / RRF
    Returns the top_k candidates re-ordered by actual cross-encoder relevance.
    """
    texts = [text for text, _ in candidates]

    # CrossEncoder scores each (question, chunk) pair directly — higher score = more relevant
    pairs = [[question, text] for text in texts]
    rerank_scores = reranker.predict(pairs)

    scored = list(zip(texts, rerank_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


if __name__ == "__main__":
    build_bm25_index()

    question = "How does self-attention work?"

    print(f"\n--- Hybrid (RRF) results for: '{question}' ---")
    fused_results = hybrid_search(question, k=8)  # get more candidates for reranker to work with
    for text, score in fused_results:
        print(f"  fused_score={score:.4f} | {text[:80]!r}")

    print(f"\n--- Reranked results ---")
    reranked = rerank(question, fused_results, top_k=4)
    for text, score in reranked:
        print(f"  rerank_score={score:.4f} | {text[:80]!r}")