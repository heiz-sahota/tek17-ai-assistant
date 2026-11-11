# TEK17 RAG assistant

A personal project: a retrieval-augmented assistant over the Norwegian building
regulations (TEK17) and a fictional pricing knowledge base, built on Azure with
FastAPI, Microsoft Foundry and Azure AI Search, with a focus on evaluation.

## Architecture
TODO: add the architecture diagram and the eight-step flow.

## Setup
1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill it in
4. `az login`
5. `python -m scripts.check_setup`

## Ingestion
`python -m ingest.parse`, then `python -m ingest.chunk`, then `python -m ingest.embed_upload`

## Run
`uvicorn app.main:app --reload`, then open http://127.0.0.1:8000/docs

## Evaluation results
TODO: retrieval table, judge agreement, known failures.

## Data
- TEK17: public regulation text from DiBK. Add the PDF to `data/tek17/`.
- Lyngpakke AS: a fictional company. All pricing documents are sample data.
