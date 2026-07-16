import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version
)

def get_embedding(text):
    """
    Sends text to Azure OpenAI and gets back a vector (list of numbers)
    representing its meaning.
    """
    response = client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    return response.data[0].embedding


if __name__ == "__main__":
    # Load the chunks we created in Day 2
    with open(os.path.join("data", "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks. Generating embeddings...")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])
        chunk["embedding"] = embedding
        print(f"  [{i+1}/{len(chunks)}] chunk_{chunk['id']} → vector of length {len(embedding)}")

    # Save chunks WITH embeddings to a new file
    output_path = os.path.join("data", "chunks_with_embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved embedded chunks to {output_path}")