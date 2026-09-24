from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    original_question: str
    retrieved_chunks: List[Document]
    final_answer: str
    retry_count: int
    needs_retry: bool