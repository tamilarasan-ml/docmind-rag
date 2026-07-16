## Tech Stack

- **Azure AI Foundry** — `gpt-5-mini` (chat), `text-embedding-3-small` (embeddings)
- **Azure AI Search** — vector index with HNSW similarity search
- **PyMuPDF** — PDF text extraction
- **Streamlit** — chat interface
- **Python** — orchestration

## Key Features

- **Automatic boilerplate removal** — detects and strips headers/footers that repeat across 50%+ of pages, rather than manually excluding pages
- **Grounded generation** — system prompt enforces answers come only from retrieved context; verified the model correctly refuses out-of-scope questions instead of hallucinating
- **Automated evaluation pipeline** — LLM-as-judge scoring for groundedness, relevance, and appropriate refusal, run against a test set that includes an intentional "trap" question outside the document's scope
- **Source citations** — every answer links back to the specific page(s) it was drawn from

## Project Structure

Docmind-rag/
├── data/
│   ├── eval_questions.json      # test set with a deliberate out-of-scope trap question
│   └── eval_report.json         # evaluation results
├── src/
│   ├── ingest.py                # PDF extraction, boilerplate removal, chunking
│   ├── embed.py                 # embedding generation
│   ├── setup_index.py           # Azure AI Search index creation
│   ├── upload.py                # push chunks + embeddings to the index
│   ├── search_test.py           # standalone retrieval test
│   ├── chat.py                  # CLI RAG chat
│   ├── evaluate.py              # automated evaluation pipeline
│   └── app.py                   # Streamlit UI
├── requirements.txt
└── .env.example                 # template for required environment variables

## Setup

1. Clone the repo and create a virtual environment:
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

2. Create a `.env` file (see `.env.example`) with your Azure credentials:
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=

3. Run the pipeline in order:
python src/ingest.py
python src/embed.py
python src/setup_index.py
python src/upload.py

4. Launch the app:
streamlit run src/app.py

## Evaluation Results

5/5 test questions scored as grounded and relevant, including correct refusal on an intentionally out-of-scope question ("What is the capital of France?") — confirming the system doesn't hallucinate when the answer isn't in the source document.

## What I Learned

Building this surfaced real engineering issues that don't show up in tutorials: PDF boilerplate polluting chunk quality, the difference between a 404 (wrong endpoint) and a DNS failure (typo in hostname), why LLM judges need their own scrutiny (caught a false-positive citation flag in my own eval pipeline), and why virtual environments break when a project folder is moved.

## Possible Next Steps

- Hybrid search (vector + keyword) instead of pure vector search
- Support multi-document ingestion
- Deploy to Azure App Service / Container Apps