import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField,
)

load_dotenv()

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

credential = AzureKeyCredential(search_key)
index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

# Define the vector search configuration (HNSW = the algorithm used for fast similarity search)
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
    profiles=[
        VectorSearchProfile(
            name="vector-profile",
            algorithm_configuration_name="hnsw-config"
        )
    ]
)

# Define the fields our index will store
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="text", type=SearchFieldDataType.String),
    SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,  # matches text-embedding-3-small output size
        vector_search_profile_name="vector-profile"
    ),
]

index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

result = index_client.create_or_update_index(index)
print(f"Index '{result.name}' created successfully.")