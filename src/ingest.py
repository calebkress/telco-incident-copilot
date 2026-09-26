"""
ingest.py

Batch-embeds the processed ticket corpus with Voyage's voyage-3 model and
upserts each ticket + its embedding vector into MongoDB Atlas.

MECHANICS THAT MATTER HERE:

1. input_type="document"
   Voyage's embedding models support an `input_type` parameter set to either
   "document" or "query". This isn't a cosmetic label — Voyage prepends a
   different internal instruction/prefix to the text depending on which one
   you pass, so a "document" embedding and a "query" embedding of the exact
   same string are NOT the same vector. The models are trained so that a
   "query" embedding of an incident and a "document" embedding of a similar
   past ticket land close together in vector space — asymmetric retrieval,
   the same shape as a classic bi-encoder. Every ticket we embed here goes in
   as input_type="document" because this is the corpus being searched, not
   the thing doing the searching. Get this backwards (embed the corpus as
   "query") and vector search still runs and still returns *something* — it
   just quietly returns worse matches, which is a nastier failure mode than
   an error.

2. Batching
   Voyage's embedding endpoint accepts a batch of texts per request, not just
   one. Sending 1500 individual one-ticket requests would mean 1500 network
   round trips and would burn through your rate limit far faster than 1500
   tickets actually requires — the API is built to accept many texts per call
   specifically so you don't do that. There are two limits to respect at
   once, per request: a max number of texts, and a max total token count
   across all of them. Voyage's current limits are documented at
   https://docs.voyageai.com/docs/rate-limits and do change between models
   and pricing tiers, so BATCH_SIZE below is a conservative default, not a
   number to trust blindly — if you hit a 400 for exceeding a token limit,
   lower it.

3. Rate limits + retries
   Free-tier rate limits are generous but real. A 429 here doesn't mean
   anything is broken — it means we're asking faster than the tier allows.
   The retry loop below backs off and tries again rather than failing the
   whole ingest run over a transient limit.

4. Idempotent upserts
   We key each Mongo document on `ticket_id` (a stable hash from
   prepare_data.py) and use `update_one(..., upsert=True)` in bulk. Re-running
   ingest.py after fixing a bug, or after appending new tickets to the CSV,
   updates/creates rather than duplicating.

Usage:
    python src/ingest.py --input data/processed/tickets.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

load_dotenv()

# Conservative defaults — see point 2 above. voyage-3's embeddings are 1024
# dimensions; this MUST match the `numDimensions` in your Atlas Vector Search
# index definition (see README) or $vectorSearch queries will fail outright.
BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2


def embed_batches(client, texts: list[str]) -> list[list[float]]:
    """Embed `texts` in batches of BATCH_SIZE, input_type='document', with
    exponential-backoff retry on rate limits / transient errors."""
    all_embeddings: list[list[float]] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[start:start + BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                result = client.embed(batch, model="voyage-3", input_type="document")
                all_embeddings.extend(result.embeddings)
                break
            except Exception as e:  # voyageai raises various error types on 429/5xx
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                print(f"  batch at {start} failed ({e}); retrying in {delay}s "
                      f"[attempt {attempt + 1}/{MAX_RETRIES}]")
                time.sleep(delay)

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/tickets.csv")
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    mongo_uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB", "telco_copilot")
    collection_name = os.environ.get("MONGODB_COLLECTION", "tickets")

    if not voyage_key:
        sys.exit("VOYAGE_API_KEY is not set — copy .env.example to .env and fill it in.")
    if not mongo_uri:
        sys.exit("MONGODB_URI is not set — copy .env.example to .env and fill it in.")

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Input file not found: {src}. Run data/prepare_data.py first.")

    df = pd.read_csv(src)
    print(f"Loaded {len(df)} processed tickets from {src}")

    # Imported here so a missing/invalid VOYAGE_API_KEY fails with our own
    # message above rather than a less obvious import-time error.
    import voyageai
    voyage_client = voyageai.Client(api_key=voyage_key)

    texts = df["text_blob"].tolist()
    embeddings = embed_batches(voyage_client, texts)

    if len(embeddings) != len(df):
        sys.exit(f"Embedded {len(embeddings)} vectors for {len(df)} tickets — mismatch, aborting before writing to Atlas.")

    print(f"Embedded {len(embeddings)} tickets, dimension {len(embeddings[0])}")

    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[db_name][collection_name]

    ops = []
    for (_, row), embedding in zip(df.iterrows(), embeddings):
        doc = row.to_dict()
        doc["embedding"] = embedding
        ops.append(UpdateOne({"ticket_id": doc["ticket_id"]}, {"$set": doc}, upsert=True))

    result = collection.bulk_write(ops)
    print(f"Upserted into {db_name}.{collection_name}: "
          f"{result.upserted_count} inserted, {result.modified_count} updated")
    print("\nNext: create the Atlas Vector Search index (see README), then run src/query.py")


if __name__ == "__main__":
    main()