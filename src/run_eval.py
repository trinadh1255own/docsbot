import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from graph import build_graph

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Client()
rag_graph = build_graph()


def target(inputs: dict) -> dict:
    """Wraps your graph so LangSmith can call it with each dataset example's input."""
    question = inputs["question"]
    state = {
        "question": question,
        "original_question": question,
        "retrieved_chunks": [],
        "final_answer": "",
        "retry_count": 0,
        "needs_retry": False,
    }
    result = rag_graph.invoke(state)
    return {"final_answer": result["final_answer"], "retrieved_chunks": result["retrieved_chunks"]}


def perform_eval(run, example) -> dict:
    answer = run.outputs.get("final_answer", "").lower()
    chunks = run.outputs.get("retrieved_chunks", [])
    sources_text = " ".join(
        chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        for chunk in chunks
    ).lower()

    answer_words = set(answer.split())
    source_words = set(sources_text.split())

    if not answer_words:
        return {"key": "groundedness", "score": 0.0}

    overlap = len(answer_words & source_words) / len(answer_words)
    return {"key": "groundedness", "score": overlap}


if __name__ == "__main__":
    results = evaluate(
        target,
        data="docsbot-baseline",
        evaluators=[perform_eval],
        experiment_prefix="docsbot-eval",
    )
    print("Evaluation complete. Check the LangSmith UI under Datasets & Experiments > docsbot-baseline > Experiments.")