from langgraph.graph import StateGraph, END
from graph_state import RAGState
from retriever_node import retriever_node
from writer_node import writer_node
from critic_node import critic_node


def route_after_critic(state: RAGState) -> str:
    return "retry" if state.get("needs_retry", False) else "done"


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retriever", retriever_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "writer")
    workflow.add_edge("writer", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retry": "retriever",
            "done": END,
        },
    )

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    question = "What are the exact hyperparameter values used in the third experiment's ablation study?"
    initial_state = {
        "question": question,
        "original_question": question,
        "retrieved_chunks": [],
        "final_answer": "",
        "retry_count": 0,
        "needs_retry": False,
    }

    result = graph.invoke(initial_state)

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
    print(f"\nRetries used: {result['retry_count']}")