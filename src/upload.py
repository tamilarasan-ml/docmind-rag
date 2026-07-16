import os
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

credential = AzureKeyCredential(search_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

# Load chunks with embeddings from Day 3 step 1
with open(os.path.join("data", "chunks_with_embeddings.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Reshape each chunk to match our index's field names exactly
documents = []
for chunk in chunks:
    documents.append({
        "id": chunk["id"],
        "text": chunk["text"],
        "page": chunk["page"],
        "embedding": chunk["embedding"]
    })

# Upload in one batch
result = search_client.upload_documents(documents=documents)

success_count = sum(1 for r in result if r.succeeded)
print(f"Uploaded {success_count} / {len(documents)} documents successfully.")   