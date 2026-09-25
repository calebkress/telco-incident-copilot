# Telco Incident Copilot

A "similar incident retrieval" tool for telco support: paste in a new incident and
get back the most similar past tickets — plus, eventually, a drafted resolution
grounded in them.

Built as a hands-on Voyage AI + MongoDB Atlas Vector Search project: the point
isn't just wiring the API calls together, it's understanding the mechanics
(`input_type` modes, why reranking sits on top of vector search instead of
replacing it, embedding dimensionality, batching/rate limits) well enough to
explain them in a Solutions Engineering conversation.

## Pipeline

```
raw ticket CSV (Kaggle, multilingual customer support tickets)
    │
    ▼
data/prepare_data.py           reframe as telco tickets, build text_blob
    │
    ▼
src/ingest.py                  voyage-3 embed (input_type="document") → Atlas
    │
    ▼
Atlas Vector Search index      (created manually in the Atlas UI — see below)
    │
    ▼
src/query.py                   voyage-3 embed (input_type="query")
                                → $vectorSearch (candidates)
                                → rerank-2 (re-score)
    │
    ▼
app.py (Streamlit)             paste an incident, compare vector-only vs
                                reranked results, [stretch] drafted resolution
```

## Data

Source: Tobias Bueck's multilingual customer support tickets dataset (Kaggle).
`data/prepare_data.py` turns it into a telco-flavored corpus:

- Drops the "Human Resources" queue (no telco-support analogue).
- Relabels queues onto telco categories (Technical Support, Network Outage &
  Maintenance, Billing & Payments, Device Returns & Exchanges, etc.) — the
  ticket *content* isn't rewritten, since the underlying issues (platform
  crashes, outages, billing disputes) already read like real enterprise
  support tickets, and rewriting embedding corpus text for brand flavor
  doesn't change what the model matches on.
- Adds a heuristic `product_line` tag (Mobile / Broadband / TV & Streaming /
  IoT & M2M / Enterprise Platform) via keyword matching — flavor for a
  segmented-book-of-business demo, not a real classifier. Currently coarse
  (most tickets land in "General Services" / "Enterprise Platform"); worth
  tightening the keyword lists once you see it in the UI.
- Samples down to ~1500 tickets (stratified by language) so embedding and
  iterating on the pipeline stays fast. The full 20k rows work fine too —
  pass `--sample-size 0`.
- Builds `text_blob` = subject + body **only**. The ticket's `answer` is kept
  separately as `resolution`, never folded into the embedded text — a real
  incoming incident has no resolution yet, so embedding the answer into the
  corpus would let vector search partly match on resolution phrasing the
  query-time embedding could never have. See the docstring in
  `prepare_data.py` for the full reasoning.

```bash
python data/prepare_data.py \
    --input data/raw/dataset-tickets-multi-lang-4-20k.csv \
    --output data/processed/tickets.csv \
    --sample-size 1500
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in VOYAGE_API_KEY and MONGODB_URI
```

## Atlas Vector Search index

Created manually in the Atlas UI (Search & Vector Search → Create Search Index
→ JSON Editor), against the `voyage-practice` cluster, `telco_copilot.tickets`
collection:

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

`numDimensions: 1024` matches voyage-3's default output dimension — if you
switch models or truncate dimensions, this has to match exactly or
`$vectorSearch` will reject the query. The two filter fields let `$vectorSearch`
pre-filter by language or category before the ANN search runs, which matters
once the corpus is big enough that "similar English billing tickets" and
"similar German outage tickets" are meaningfully different searches.

## Status

- [x] Data prep (`data/prepare_data.py`)
- [ ] Embedding + ingestion (`src/ingest.py`)
- [ ] Atlas Vector Search index (manual, see above)
- [ ] Query pipeline: vector search + rerank (`src/query.py`)
- [ ] Streamlit frontend (`app.py`)
- [ ] Stretch: LLM-drafted resolution
