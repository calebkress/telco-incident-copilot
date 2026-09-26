"""
query.py

Given a new incident description, finds the most similar past tickets:
embed the incident -> $vectorSearch for candidates -> rerank-2 to re-score
-> print vector-search-only results next to reranked results so you can see
what reranking actually changed.

MECHANICS THAT MATTER HERE:

1. input_type="query" (mirror of ingest.py's input_type="document")
   The incident text gets embedded with input_type="query" because it's
   playing the opposite role from every ticket in the corpus: it's the thing
   searching, not the thing being searched. Same reasoning as ingest.py, just
   flipped. If you embedded the incident as "document" instead, vector search
   would still run and still return results — just quietly worse ones, since
   you'd be comparing two "document"-mode vectors instead of a "query"-mode
   vector against "document"-mode vectors, which isn't what the model was
   trained to make close together.

2. $vectorSearch: numCandidates vs limit
   Atlas Vector Search is an *approximate* nearest-neighbor search (ANN), not
   an exhaustive brute-force comparison against all 1500 vectors. numCandidates
   controls how many candidates the ANN algorithm actually gathers internally
   before returning the top `limit` of them to you. A bigger numCandidates
   means a wider, more thorough internal search (better recall, i.e. less
   chance of missing something that should've been near the top) at the cost
   of latency. Atlas's own guidance is roughly 10-20x your `limit` for
   small-to-medium collections; NUM_CANDIDATES=150 against LIMIT=25 here is on
   the lower end of that range, which is fine for a ~1500-document demo corpus
   and would need raising as the corpus grows.

3. Why rerank sits ON TOP of vector search instead of replacing it
   This is the one people usually gloss over. Vector search compares the query
   embedding against each ticket's *precomputed, independent* embedding — every
   ticket was embedded once, in isolation, with no knowledge of what the query
   would eventually be (that's what let ingest.py embed 1500 tickets in a
   single afternoon instead of re-embedding on every search). That's a
   "bi-encoder" setup: fast and scalable, because comparing precomputed vectors
   is just arithmetic, but it can miss relevance signals that only show up when
   you actually look at the query and a specific candidate *together*.

   The reranker (rerank-2) is a "cross-encoder": it takes the actual query text
   and a candidate's text as a pair, and runs them jointly through a model that
   attends to both at once, producing a relevance score specific to that exact
   pairing. That's meaningfully more accurate for judging "is this actually a
   good match" — but it's also far more expensive, because you can't precompute
   it. Every (query, candidate) pair needs its own forward pass at query time.
   Running a cross-encoder over all 1500 tickets on every search would be slow
   and expensive. Running it over the ~25 candidates vector search already
   narrowed things down to is cheap and fast. That division of labor — cheap
   approximate search over everything, expensive precise re-scoring over a
   short list — is the whole reason the two steps exist as separate stages
   instead of picking one or the other.

Usage:
    python src/query.py "customer's router keeps dropping connection at night"
    python src/query.py --language en --category "Network Outage & Maintenance" "..."
    python src/query.py   # no argument -> uses a built-in example incident
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

VECTOR_INDEX_NAME = "vector_index"       # must match the name you gave the index in Atlas
NUM_CANDIDATES = 150                     # ANN candidate pool size (see point 2 above)
VECTOR_SEARCH_LIMIT = 25                 # how many candidates vector search returns
RERANK_TOP_K = 5                         # how many survive reranking, shown as final results

EXAMPLE_INCIDENT = (
    "Customer's router keeps dropping the connection every evening during "
    "peak hours. Tried restarting the modem but the issue persists."
)


def embed_query(voyage_client, text: str) -> list[float]:
    result = voyage_client.embed([text], model="voyage-3", input_type="query")
    return result.embeddings[0]


def vector_search(collection, query_vector, language=None, telco_category=None):
    vs_stage = {
        "index": VECTOR_INDEX_NAME,
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": NUM_CANDIDATES,
        "limit": VECTOR_SEARCH_LIMIT,
    }

    filters = {}
    if language:
        filters["language"] = {"$eq": language}
    if telco_category:
        filters["telco_category"] = {"$eq": telco_category}
    if filters:
        vs_stage["filter"] = filters

    pipeline = [
        {"$vectorSearch": vs_stage},
        {
            "$project": {
                "_id": 0,
                "ticket_id": 1,
                "subject": 1,
                "text_blob": 1,
                "resolution": 1,
                "telco_category": 1,
                "language": 1,
                "priority": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def rerank(voyage_client, query_text: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    docs = [c["text_blob"] for c in candidates]
    result = voyage_client.rerank(query_text, docs, model="rerank-2", top_k=top_k)

    reranked = []
    for r in result.results:
        candidate = dict(candidates[r.index])
        candidate["rerank_score"] = r.relevance_score
        reranked.append(candidate)
    return reranked


def print_results(title: str, results: list[dict], score_key: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if not results:
        print("(no results)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['ticket_id']}]  {score_key}={r[score_key]:.4f}  "
              f"| {r['telco_category']} | {r['language']} | priority={r.get('priority')}")
        print(f"   {r['subject']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("incident", nargs="?", default=None,
                         help="Incident text to search for. Omit to use a built-in example.")
    parser.add_argument("--language", default=None, help="Optional filter, e.g. 'en' or 'de'")
    parser.add_argument("--category", default=None, help="Optional filter on telco_category")
    args = parser.parse_args()

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    mongo_uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB", "telco_copilot")
    collection_name = os.environ.get("MONGODB_COLLECTION", "tickets")

    if not voyage_key:
        sys.exit("VOYAGE_API_KEY is not set — copy .env.example to .env and fill it in.")
    if not mongo_uri:
        sys.exit("MONGODB_URI is not set — copy .env.example to .env and fill it in.")

    incident_text = args.incident or EXAMPLE_INCIDENT
    print(f"Incident:\n  {incident_text}\n")
    if not args.incident:
        print("(no incident text given — using the built-in example above)")

    import voyageai
    voyage_client = voyageai.Client(api_key=voyage_key)

    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[db_name][collection_name]

    query_vector = embed_query(voyage_client, incident_text)

    candidates = vector_search(collection, query_vector, language=args.language, telco_category=args.category)
    if not candidates:
        sys.exit("No candidates returned. Check that the Atlas Vector Search index "
                  f"'{VECTOR_INDEX_NAME}' is Active and that ingest.py has run successfully.")

    print_results(f"VECTOR SEARCH ONLY  (top 5 of {len(candidates)} candidates retrieved)",
                  candidates[:5], "score")

    reranked = rerank(voyage_client, incident_text, candidates, top_k=RERANK_TOP_K)
    print_results(f"AFTER RERANK (rerank-2)  (top {len(reranked)})", reranked, "rerank_score")

    print(f"\n{'=' * 72}")
    print("Note: vector search and rerank scores are on different scales "
          "(cosine similarity vs. rerank relevance) — don't compare the numbers "
          "directly across the two sections. Compare the *ordering*.")


if __name__ == "__main__":
    main()