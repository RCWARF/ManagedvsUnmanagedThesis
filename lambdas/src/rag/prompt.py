QA_V1 = (
    "Answer the following question using ONLY the context provided below. "
    "If the answer cannot be found in the context, say "
    "\"I don't have enough information to answer that question.\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_prompt(question: str, contexts: list[str]) -> str:
    return QA_V1.format(context="\n\n---\n\n".join(contexts), question=question)
