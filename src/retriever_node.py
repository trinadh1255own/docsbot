import os
from langchain_core.documents import Document
from graph_state import RAGState
from hybrid_retriever import hybrid_search, rerank, build_bm25_index, BM25_INDEX_PATH

# Build the BM25 index once if it doesn't exist yet.
# (Re-run build_bm25_index() manually after re-ingesting new documents.)
if not os.path.exists(BM25_INDEX_PATH):
    print("No BM25 index found — building one now...")
    build_bm25_index()


def retriever_node(state: RAGState, k: int = 4, candidate_pool: int = 8) -> RAGState:
    question = state["question"]

    fused_candidates = hybrid_search(question, k=candidate_pool)
    reranked = rerank(question, fused_candidates, top_k=k)

    print(f"Retrieved {len(reranked)} chunks (hybrid + reranked) for question: '{question}'")
    for text, score in reranked:
        print(f"  rerank_score={score:.4f} | {text[:80]!r}")

    # Wrap plain text back into Document objects so downstream nodes
    # (writer_node) can keep using chunk.page_content uniformly.
    documents = [Document(page_content=text) for text, score in reranked]

    return {**state, "retrieved_chunks": documents}


if __name__ == "__main__":
    test_state = {"question": "How does self-attention work?", "retrieved_chunks": [], "final_answer": ""}
    result = retriever_node(test_state)