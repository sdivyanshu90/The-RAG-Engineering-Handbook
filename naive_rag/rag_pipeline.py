from config import COLLECTION_NAME, TOP_K, MAX_TOKENS
from embeddings import embed_text

PROMPT_TEMPLATE = """
Use the following pieces of information enclosed in <context> tags 
to answer the question enclosed in <question> tags.

If the answer is not in the context, say you don't know.

<context>
{context}
</context>

<question>
{question}
</question>
"""


def ask_question(question, milvus_client, llm_client):

    question_embedding = embed_text(question)

    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[question_embedding],
        limit=TOP_K,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    retrieved_chunks = [
        res["entity"]["text"] for res in search_res[0]
    ]

    context = "\n\n".join(retrieved_chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )

    return response.choices[0].message.content.strip()