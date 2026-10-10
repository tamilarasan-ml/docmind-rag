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

```text
docmind-rag/
├── data/
│   ├── eval_gold.json
│   ├── eval_questions.json
│   ├── eval_report.json
│   ├── chunks.json
│   └── chunks_with_embeddings.json
├── src/
│   ├── ingest.py
│   ├── embed.py
│   ├── setup_index.py
│   ├── upload.py
│   ├── chat.py
│   ├── app.py
│   ├── citation_utils.py
│   ├── evaluate.py
│   ├── retrieval_eval.py
│   ├── query_rewrite_eval.py
│   └── additional evaluation and validation scripts
├── tests/
│   └── test_citation_utils.py
├── requirements.txt
├── README.md
└── .env.example
```
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


The current saved evaluation report covers 26 questions: 22 answerable questions and 4 intentionally unanswerable questions.

* **Groundedness:** 26/26 judge results passed
* **Relevance:** 26/26 judge results passed
* **Retrieval Hit@1:** 22/22 answerable questions (100%)
* **Retrieval Hit@3:** 22/22 answerable questions (100%)
* **Retrieval Hit@5:** 22/22 answerable questions (100%)
* **Fact coverage:** 52/52 expected facts covered
* **Out-of-scope handling:** 4/4 trap questions correctly refused
* **Citation-page retrieval:** cited pages were present in the retrieved context for all 26 questions
* **Citation-support judgment:** 25/26 passed; one citation-attribution issue remains under review
* **Evaluation errors:** 0

These results describe the saved evaluation run, not a guarantee of production performance. Citation presence and citation correctness are evaluated separately.
Building this surfaced real engineering issues that don't show up in tutorials: PDF boilerplate polluting chunk quality, the difference between a 404 (wrong endpoint) and a DNS failure (typo in hostname), why LLM judges need their own scrutiny (caught a false-positive citation flag in my own eval pipeline), and why virtual environments break when a project folder is moved.


## Possible Next Steps

* Expand the evaluation set with more answerable, ambiguous, and out-of-scope questions.
* Investigate the remaining citation-attribution issue and verify the correction with a targeted evaluation.
* Extend ingestion to support multiple documents.
* Add deployment and operational monitoring for Azure hosting.
