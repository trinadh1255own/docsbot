import os
import secrets
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from graph import build_graph
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from guardrail import is_suspicious



load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

print(f"LangSmith tracing enabled: {os.environ.get('LANGSMITH_TRACING')}")
print(f"LangSmith project: {os.environ.get('LANGSMITH_PROJECT')}")

API_KEY = os.environ["DOCSBOT_API_KEY"]
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Security(api_key_header)):
    if provided_key is None or not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return provided_key


app = FastAPI(
    title="DocsBot API",
    description="Production RAG assistant with hybrid search, reranking, and grounded answers.",
    version="0.1.0",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

rag_graph = build_graph()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask DocsBot")


class SourceChunk(BaseModel):
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
def query(request: Request, body: QueryRequest, api_key: str = Depends(verify_api_key)):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if is_suspicious(body.question):
        raise HTTPException(status_code=400, detail="Question contains disallowed instruction-like content")

    initial_state = {
        "question": body.question,
        "retrieved_chunks": [],
        "final_answer": "",
    }

    result = rag_graph.invoke(initial_state)
    sources = [SourceChunk(content=chunk.page_content) for chunk in result["retrieved_chunks"]]

    return QueryResponse(
        question=body.question,
        answer=result["final_answer"],
        sources=sources,
    )