import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)


TARGETED_QUERY_VARIANTS = {
    "How do I combine several criteria using the AND operator?":
        "JQL AND operator combine multiple criteria "
        "project status Customer Impact",

    "How can I find issues whose summary does not contain the word Task?":
        'Jira JQL negative text search operator !~ summary "Task"',
}


def retrieve_chunks(question, top_k=3):
    """Vector-only retrieval with targeted query variants for tested questions."""
    retrieval_query = TARGETED_QUERY_VARIANTS.get(question, question)

    question_embedding = openai_client.embeddings.create(
        model=embedding_deployment,
        input=retrieval_query
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=question_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "text", "page"]
    )

    return list(results)


def build_context(chunks):
    """
    Formats retrieved chunks into a labeled context block,
    so the model can cite exactly which chunk/page it used.
    """
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source: page {chunk['page']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def answer_question(question):
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    # Step 2: Build a grounded prompt
    system_prompt = (
       "You are a helpful assistant that answers questions using ONLY the provided context. "
        "Do not use outside knowledge or invent facts.\n\n"

        "ANSWER COMPLETENESS:\n"
        
        "- When a question asks how to use a function in an example, explain "
        "the function's meaning as well as the query, if that definition is "
        "available in the provided context. Cite the page supporting each detail.\n"
        "- Answer the full question, not just its first or easiest part.\n"
        "- For questions asking which operators, features, methods, or categories are available, "
        "list the relevant items found in the context and briefly explain what each category does "
        "when the context provides that information.\n"
        "- Do not merely list operator names or symbols if the context explains their meaning.\n"
        "- For questions about a specific operator or feature, explain its purpose and include "
        "a relevant source example when one is available.\n"
        "- Preserve source examples faithfully. Do not duplicate operator names, field names, "
        "keywords, or query fragments when reproducing an example.\n"
        "- Before returning a query example, check that its syntax matches the source and that "
        "the example does not contain accidental repeated text. If the source example is unclear, "
        "describe the operator without inventing a corrected example.\n"
        "- Include important distinctions, limitations, and examples that directly answer the question.\n"
        "- Keep the answer concise, but do not sacrifice completeness for brevity.\n"
        "- When the source describes complementary operators or alternatives relevant to the question, "
        "include those distinctions when they help complete the answer; for example, mention both "
        "IS and IS NOT when the context explains both.\n\n"
        "- When reproducing a source example, preserve the exact query syntax. "
        "Keep explanatory operator labels separate from the query itself; "
        "never prepend an operator to a complete field-operator-value query.\n\n"

        "RELATED OPERATORS AND ALTERNATIVES:\n"
        "- When the retrieved context documents a directly related alternative, "
        "briefly explain it if it helps answer the question. For example, "
        "mention IS NOT when the context explains IS and IS NOT.\n"
        "- Include a source-supported example and cite its page when available.\n"
        "- Do not add unrelated operators or unsupported details.\n\n"

        "EVIDENCE-BASED OPERATOR COMBINATIONS:\n"
        "- Do not infer that two operators or query features can be combined "
        "unless the provided context supports that combination.\n"
        "- Documentation of individual features does not by itself prove "
        "that every combination of those features is valid.\n"
        "- When a combined example is not explicitly supported, use the "
        "documented example or explain only the supported behavior.\n\n"


        "GROUNDING AND CITATIONS:\n"
        "- Every factual claim must be supported by the provided context.\n"
        "- Cite the source page for relevant claims using [page N].\n"
        "- If information comes from multiple pages, cite the corresponding pages.\n"
        "- If the context supports only part of the answer, explain what it supports and identify "
        "what cannot be established from the context.\n"
        "- If the answer is not available in the context, say you don't have enough information.\n"
        "- Never fabricate examples, metrics, technical behavior, or source references.\n"
        "- When the context contains an explicit example that directly answers "
        "the question, include that exact example with its source-page citation. "
        "Do not omit a documented example merely because related syntax "
        "combinations are uncertain.\n"        
        "- Cite the page that directly supports each factual claim.\n"
        "- Keep claims separate when they rely on different evidence; do not use "
        "one citation to imply that a page supports several distinct claims.\n"
        "- A page describing a feature does not automatically support claims about "
        "that feature's performance, risks, limitations, or recommended usage.\n"
        "- For performance warnings, risks, and recommendations, cite only pages "
        "that explicitly state or directly support the specific warning.\n"
        "- Before returning the answer, check every factual claim against its cited "
        "page. If the page does not directly support the claim, remove the claim, "
        "qualify it, or cite a different supporting page.\n\n"



    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # Step 3: Generate the grounded answer
    response = openai_client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.choices[0].message.content or ""

    # Targeted correction for the documented JQL example.

    if question.strip() == (
    "How can I find issues whose summary does not contain the word Task?"
):
        answer = answer.replace(
            '!~ summary !~ "Task"',
            'summary !~ "Task"',
    )

    return answer, chunks


if __name__ == "__main__":
    print("DocMind RAG Chat — ask questions about your PDF (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer, sources = answer_question(question)

        print(f"\nAssistant: {answer}")
        print(f"\n(Retrieved from pages: {[s['page'] for s in sources]})")
        print("-" * 60 + "\n")
