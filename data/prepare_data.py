"""
prepare_data.py

Turns Tobias Bueck's multilingual customer-support-tickets dataset (Kaggle) into a
telco "similar incident retrieval" corpus.

WHAT THIS DOES (and why):

1. Loads the raw CSV (subject, body, answer, type, queue, priority, language, tag_1..8).

2. Drops the "Human Resources" queue. It's the one category in the source data that
   has no telco-support analogue, and keeping it would poison a "similar incident"
   demo with unrelated internal-HR tickets.

3. Relabels the generic support queues onto telco-flavored categories
   (QUEUE_MAP below). We do NOT rewrite the ticket bodies themselves — the
   underlying issues (platform crashes, outages, billing disputes, hardware
   failures) already read like real enterprise/telco support tickets. Rewriting
   136k characters of German and English prose to insert telco vocabulary would
   burn a weekend for approximately zero improvement in embedding quality: Voyage
   embeds meaning, not brand flavor. The relabeling is what a reader (or a demo
   audience) actually notices.

4. Adds a lightweight, heuristic `product_line` tag (Mobile, Broadband/Fixed Line,
   TV & Streaming, IoT & M2M, Enterprise Platform) via keyword matching. This is
   NOT a rigorous classifier — it's demo flavor so results look like a segmented
   telco book of business instead of an undifferentiated ticket pile. Say so if
   anyone asks how it was built.

5. Builds `text_blob` = subject + body ONLY. The `answer` column is kept as a
   separate `resolution` field, never folded into text_blob. This is a deliberate
   modeling decision, not an oversight:

   At query time, a real incoming incident has no resolution yet — that's the
   thing you're trying to find. If we embedded (subject + body + answer) for the
   corpus, the vector index would partly be matching on resolution phrasing,
   which is information the query-time embedding could never have. The retrieval
   would look great in a demo and be quietly cheating. Keeping `text_blob` limited
   to what a support agent sees when a ticket comes in — and only using
   `resolution` afterward, for display and for the stretch-goal LLM drafting step
   — keeps the input_type="document" (corpus) / input_type="query" (incoming
   incident) contract honest on both sides.

6. Samples down to a manageable corpus size (default 1500 tickets, stratified by
   language so the en/de mix is preserved). Voyage's free tier and rate limits
   can absolutely handle the full 20k rows, but for a weekend build there's no
   reason to wait on batch-embedding calls (and re-embedding while you iterate on
   the pipeline) for more tickets than you need to demo "similar incident
   retrieval" convincingly. Bump --sample-size or pass 0 for "keep everything" once
   the pipeline is proven out.

Usage:
    python data/prepare_data.py \\
        --input data/raw/dataset-tickets-multi-lang-4-20k.csv \\
        --output data/processed/tickets.csv \\
        --sample-size 1500
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import pandas as pd

# Source queue -> telco-flavored category. "Human Resources" is intentionally
# absent: rows in that queue are dropped in main(), not mapped.
QUEUE_MAP = {
    "General Inquiry": "General Inquiry",
    "Customer Service": "General Customer Service",
    "Technical Support": "Technical Support",
    "IT Support": "Technical Support",
    "Product Support": "Device & Service Support",
    "Billing and Payments": "Billing & Payments",
    "Service Outages and Maintenance": "Network Outage & Maintenance",
    "Returns and Exchanges": "Device Returns & Exchanges",
    "Sales and Pre-Sales": "Sales & Upgrades",
}

# Heuristic keyword -> product line. Checked in order; first match wins.
# Deliberately coarse — this is demo segmentation, not a trained classifier.
PRODUCT_LINE_KEYWORDS = [
    ("Network Outage & Maintenance", ["outage", "down", "ausfall", "störung", "netzwerk", "network"]),
    ("Mobile", ["phone", "smartphone", "mobile", "handy", "sim", "roaming", "app crash"]),
    ("Broadband / Fixed Line", ["router", "wifi", "wlan", "broadband", "dsl", "internet connection", "modem"]),
    ("TV & Streaming", ["streaming", "tv", "set-top", "settop", "channel", "sender"]),
    ("IoT & M2M", ["sensor", "iot", "m2m", "device fleet", "connected device"]),
    ("Enterprise Platform", ["platform", "plattform", "server", "database", "datenbank", "api", "cloud"]),
]
DEFAULT_PRODUCT_LINE = "General Services"


def classify_product_line(subject: str, body: str) -> str:
    text = f"{subject} {body}".lower()
    for label, keywords in PRODUCT_LINE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return label
    return DEFAULT_PRODUCT_LINE


def make_ticket_id(row_index: int, subject: str, body: str) -> str:
    # Stable, content-derived id so re-running prepare_data.py on the same source
    # data reproduces the same ids (matters once you're upserting into Atlas).
    h = hashlib.sha1(f"{row_index}:{subject}:{body}".encode("utf-8")).hexdigest()[:12]
    return f"tkt_{h}"


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df
    frac_per_lang = sample_size / len(df)
    parts = []
    for _, group in df.groupby("language", group_keys=False):
        n = max(1, round(len(group) * frac_per_lang))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    out = pd.concat(parts)
    # groupby rounding can overshoot/undershoot by a couple rows; trim/pad to exact size
    if len(out) > sample_size:
        out = out.sample(n=sample_size, random_state=seed)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/raw/dataset-tickets-multi-lang-4-20k.csv")
    parser.add_argument("--output", default="data/processed/tickets.csv")
    parser.add_argument("--sample-size", type=int, default=1500, help="0 = keep all rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Input file not found: {src}\n"
                  f"Drop the Kaggle CSV at that path (or pass --input) before running this.")

    df = pd.read_csv(src)
    required = {"subject", "body", "answer", "type", "queue", "priority", "language"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing expected columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["subject", "body"]).copy()
    df = df[df["queue"] != "Human Resources"].copy()
    print(f"Loaded {before} rows -> {len(df)} after dropping empty/HR rows")

    df["telco_category"] = df["queue"].map(QUEUE_MAP)
    unmapped = df[df["telco_category"].isna()]["queue"].unique()
    if len(unmapped):
        sys.exit(f"QUEUE_MAP is missing entries for: {list(unmapped)}. Update QUEUE_MAP and rerun.")

    df["product_line"] = [
        classify_product_line(s, b) for s, b in zip(df["subject"], df["body"])
    ]

    df = stratified_sample(df, args.sample_size, seed=args.seed)

    df["ticket_id"] = [
        make_ticket_id(i, s, b) for i, (s, b) in enumerate(zip(df["subject"], df["body"]))
    ]
    # The corpus text: what a support agent (and the embedding model) sees when
    # the ticket comes in. No `answer` in here — see module docstring, point 5.
    df["text_blob"] = df["subject"].str.strip() + "\n\n" + df["body"].str.strip()

    out_cols = [
        "ticket_id", "language", "telco_category", "product_line", "priority",
        "type", "subject", "body", "text_blob", "answer",
    ]
    df = df.rename(columns={"answer": "resolution"})
    out_cols = [c if c != "answer" else "resolution" for c in out_cols]
    df = df[out_cols]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} tickets to {out_path}")
    print("\nLanguage split:")
    print(df["language"].value_counts().to_string())
    print("\ntelco_category split:")
    print(df["telco_category"].value_counts().to_string())
    print("\nproduct_line split:")
    print(df["product_line"].value_counts().to_string())


if __name__ == "__main__":
    main()
