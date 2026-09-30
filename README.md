# Telco Incident Copilot 📡

A two-stage retrieval pipeline for technical support teams: paste an incoming incident, perform hybrid filtering, and retrieve the most relevant past ticket resolutions in real time using **Voyage AI** and **MongoDB Atlas Vector Search**.

Built to demonstrate how adding cross-encoder reranking (`rerank-2`) on top of vector similarity (`voyage-3`) drastically improves top-k retrieval quality without relying on expensive LLM generation passes.

---

## Architecture

```
Incoming Incident
       │
       ▼
src/query.py (embed with voyage-3, input_type="query")
       │
       ▼
MongoDB Atlas Vector Search ($vectorSearch ANN + pre-filtering)
       │ (returns top 25 candidate docs)
       ▼
Voyage AI rerank-2 (cross-encoder re-scoring)
       │ (boosts joint query-candidate relevance)
       ▼
app.py (Streamlit UI comparison & resolution grounding)
```

### Key Technical Decisions
* **Asymmetric Embeddings:** Corpus documents are embedded with `input_type="document"`, while incoming queries use `input_type="query"`. This aligns with Voyage's bi-encoder training for asymmetric retrieval tasks.
* **No Answer Contamination:** The `text_blob` used for indexing contains only the ticket subject and body. Historical `resolution` data is strictly reserved for display and grounding—preventing vector search from cheating on answer phrasing that an incoming query wouldn't have.
* **Two-Stage Retrieval:** Vector search acts as a fast candidate generator (ANN over cosine similarity). `rerank-2` acts as a heavy joint-attention cross-encoder, re-scoring candidates based on specific hardware, symptom, and temporal nuances.
* **Idempotent Ingestion:** Corpus docs use SHA-1 hashed `ticket_id` keys with MongoDB `upsert=True` bulk operations to prevent duplicate entries on re-runs.

---

## Quickstart

### 1. Prerequisites & Environment Setup
Clone the repo and set up your virtual environment:

```bash
git clone https://github.com/your-username/telco-incident-copilot.git
cd telco-incident-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`):

```ini
VOYAGE_API_KEY=your_voyage_key
MONGODB_URI=your_mongodb_atlas_uri
MONGODB_DB=telco_copilot
MONGODB_COLLECTION=tickets
```

### 2. Atlas Vector Search Index Configuration
In your MongoDB Atlas UI, navigate to **Search & Vector Search** -> **Create Search Index** (JSON Editor) under the `telco_copilot.tickets` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "language" },
    { "type": "filter", "path": "telco_category" }
  ]
}
```

### 3. Pipeline Execution

```bash
# 1. Prepare data (sample dataset to ~1,500 stratified telco tickets)
python data/prepare_data.py

# 2. Batch-embed corpus with voyage-3 and upsert to MongoDB
python src/ingest.py

# 3. Test CLI query execution & compare Vector-Only vs Reranked results
python src/query.py "Router keeps dropping connection during peak hours"

# 4. Launch Streamlit UI
streamlit run app.py
```

---

## Project Structure

```
telco-incident-copilot/
├── app.py                # Streamlit UI (Side-by-side comparison & resolution grounding)
├── data/
│   ├── prepare_data.py   # Corpus reframing, language sampling & SHA-1 hashing
│   └── processed/        # Stratified dataset output
├── src/
│   ├── ingest.py         # Batch voyage-3 embedding & MongoDB bulk upserts
│   ├── query.py          # $vectorSearch + rerank-2 execution logic
│   └── eval.py           # Multi-category / cross-lingual pipeline evaluation
├── .env.example
├── .gitignore
└── requirements.txt
```
