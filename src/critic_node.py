#from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from graph_state import RAGState
from llm import get_llm

#llm = ChatOllama(model="llama3.1")
llm = get_llm()

MAX_RETRIES = 2


def critic_node(state: RAGState) -> RAGState:
    question = state["original_question"]
    answer = state["final_answer"]
    retry_count = state.get("retry_count", 0)

    system_prompt = """You are a strict quality checker for a documentation assistant.
Given a question and an answer, decide if the answer is well-grounded, specific, and actually answers the question.
Respond with exactly one word: GOOD or WEAK.
Respond WEAK if the answer hedges, says the context is insufficient, or is vague."""

    human_prompt = f"Question: {question}\n\nAnswer: {answer}\n\nVerdict (GOOD or WEAK):"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    verdict = response.content.strip().upper()
    is_weak = "WEAK" in verdict

    print(f"Critic verdict: {verdict} (retry_count={retry_count})")

    if is_weak and retry_count < MAX_RETRIES:
        # Reformulate the question to try to get better retrieval next time
        reformulate_prompt = f"""The following question did not get a good answer from a document search system.
Rewrite it as a clearer, more specific search query to help find better matching content.
Return only the rewritten question, nothing else.

Original question: {question}"""

        reformulated = llm.invoke([HumanMessage(content=reformulate_prompt)]).content.strip()
        print(f"Reformulated question: {reformulated!r}")

        return {
            **state,
            "question": reformulated,
            "needs_retry": True,
            "retry_count": retry_count + 1,
        }

    return {**state, "needs_retry": False}