"""
eval.py

Runs a small, hand-picked set of test incidents — spanning different telco
categories and both languages in the corpus — through the full pipeline
(embed -> vector search -> rerank) and prints the top reranked results for
each one.

WHAT THIS IS AND ISN'T:

This is a breadth check, not a rigorous evaluation. There's no labeled ground
truth here (no "ticket X's correct matches are Y and Z" answer key), so this
script can't produce a real accuracy number. What it CAN tell you: does the
pipeline return sensible, on-topic results across a *range* of incident
types, or does it only look good on the one query you've already tested to
death?

If this looks solid across the board, the next real step (not included here)
is a proper eval set: hand-label a couple dozen incidents with which ticket
ids you'd consider a correct match, then measure precision@k, recall@k, and
mean reciprocal rank — actual numbers you can point to and track over time
as you tune filtering/thresholds. This script is the fast, cheap gut-check
before investing in that.

Usage:
    python src/eval.py
"""

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

from query import embed_query, vector_search, rerank, print_results

load_dotenv()

TEST_INCIDENTS = [
    {
        "label": "Network outage",
        "text": "Customers in the downtown area have reported completely "
                "losing mobile signal since this morning. Possible cell "
                "tower outage.",
    },
    {
        "label": "Billing dispute",
        "text": "I was charged twice for last month's bill and need a "
                "refund on the duplicate charge.",
    },
    {
        "label": "Device return",
        "text": "I want to return the router I ordered two weeks ago "
                "because it doesn't work with my ISP setup.",
    },
    {
        "label": "Account security",
        "text": "A customer can't log into their account portal and thinks "
                "their password may have been compromised.",
    },
    {
        "label": "Sales / upgrade",
        "text": "A customer is asking about upgrading their current mobile "
                "plan to unlimited data with international roaming.",
    },
    {
        "label": "Network outage (German)",
        "text": "Seit heute Morgen haben Kunden im Innenstadtbereich "
                "komplett kein Mobilfunksignal mehr, vermutlich ein Ausfall "
                "eines Mobilfunkmasts.",
    },
]


def main():
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    mongo_uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB", "telco_copilot")
    collection_name = os.environ.get("MONGODB_COLLECTION", "tickets")

    if not voyage_key or not mongo_uri:
        sys.exit("VOYAGE_API_KEY and MONGODB_URI must be set in .env")

    import voyageai
    voyage_client = voyageai.Client(api_key=voyage_key)

    mongo_client = MongoClient(mongo_uri)
    collection = mongo_client[db_name][collection_name]

    for case in TEST_INCIDENTS:
        print(f"\n\n########## {case['label']} ##########")
        print(f"Incident: {case['text']}")

        query_vector = embed_query(voyage_client, case["text"])
        candidates = vector_search(collection, query_vector)
        if not candidates:
            print("  (no candidates returned)")
            continue

        reranked = rerank(voyage_client, case["text"], candidates, top_k=3)
        print_results(f"Top 3 reranked results — {case['label']}", reranked, "rerank_score")

    print("\n\nDone. Eyeball each section: do the top results actually match "
          "the incident category? Flag any that look wrong the way the "
          "medical-data-breach case did earlier.")


if __name__ == "__main__":
    main()