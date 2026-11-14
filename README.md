# Semantic Product Search API

A small Flask application for searching an e-commerce product catalogue with sentence embeddings and Pinecone. It supports direct text queries and OCR-assisted queries from uploaded images.

## How it works

1. `integrate_pinecone.py` reads a cleaned product CSV and encodes product descriptions with `all-MiniLM-L6-v2`.
2. The ingestion script stores each vector and its product metadata in a Pinecone index.
3. `/api/search` embeds a text query and returns the closest products.
4. `/api/ocr-search` extracts text from an uploaded image with EasyOCR and searches using the extracted text.

The Gemini-generated waiting message from the previous version was removed. It did not contribute to retrieval or product ranking, so it was not part of the core search workflow.

## Requirements

- Python 3.10 or newer
- A Pinecone account and API key
- A cleaned e-commerce CSV containing `Description`, `StockCode` and `UnitPrice` columns
- The dataset should be placed at `data/Cleaned_Ecommerce_dataset.csv` by default

The dataset is not included in this repository. Before publishing this project as a portfolio submission, document the dataset's original source, licence and any preprocessing performed.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and configure credentials:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and provide a newly rotated Pinecone API key. Do not commit `.env`.

## Build the vector index

Place the cleaned CSV at `data/Cleaned_Ecommerce_dataset.csv`, then run:

```bash
python integrate_pinecone.py
```

Optional arguments:

```bash
python integrate_pinecone.py --input data/Cleaned_Ecommerce_dataset.csv --max-per-description 200 --batch-size 100
```

Set `--max-per-description 0` to include every valid row. The script creates the configured index if it does not exist and upserts vectors with product metadata. Re-running it uses deterministic IDs, so existing records are overwritten rather than duplicated.

## Start the API

```bash
python app.py
```

The API listens on `http://127.0.0.1:5000`.

## API endpoints

### Health check

`GET /health`

### Text search

`POST /api/search`

```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"pink bags"}'
```

Optional `top_k` query parameter accepts values from 1 to 20.

### OCR-assisted search

`POST /api/ocr-search`

```bash
curl -X POST http://127.0.0.1:5000/api/ocr-search \
  -F "file=@shopping-list.jpg"
```

The image upload limit is 8 MB. The first OCR request may take longer while EasyOCR loads its model.

## Tests

Install the project dependencies and run:

```bash
pip install pytest
pytest
```

## Current limitations

- The source dataset is not included, and its provenance still needs to be documented.
- Pinecone and OCR integration require external setup and have not yet been verified against live services in this repository.
- There is no frontend in the repository. The application exposes a JSON API.
- Search quality has not been evaluated with a labelled query/product relevance set. Do not claim retrieval accuracy or production readiness without that evaluation.

## Security note

A Pinecone credential was committed in the repository's earlier history. Treat it as compromised: revoke or rotate it in Pinecone before using this project. Removing a key from the current source does not remove it from Git history. The old Gemini integration also contained a credential value; rotate that credential too if it was a real key.
