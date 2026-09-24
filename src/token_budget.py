import tiktoken

# cl100k_base is a good general-purpose tokenizer approximation across most modern LLMs
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def fit_chunks_to_budget(chunks: list, max_tokens: int = 1500) -> list:
    """
    chunks: list of strings, already ordered by relevance (best first — e.g. post-reranking)
    Returns the largest prefix of chunks that fits within max_tokens.
    """
    selected = []
    running_total = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if running_total + chunk_tokens > max_tokens:
            break
        selected.append(chunk)
        running_total += chunk_tokens

    print(f"Selected {len(selected)}/{len(chunks)} chunks, using {running_total}/{max_tokens} tokens")
    return selected


if __name__ == "__main__":
    sample_chunks = [
        "Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence.",
        "As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations.",
        "What it does",
    ]
    fitted = fit_chunks_to_budget(sample_chunks, max_tokens=30)
    for c in fitted:
        print(f"  kept: {c[:60]!r}")