import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import voyageai

# Import functions from your existing pipeline logic
from src.query import (
    embed_query,
    vector_search,
    rerank,
    VECTOR_INDEX_NAME,
    EXAMPLE_INCIDENT,
)

load_dotenv()

st.set_page_config(
    page_title="Telco Incident Copilot",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Telco Incident Copilot")
st.caption("Similar Incident Retrieval & Resolution Grounding with Voyage AI + MongoDB Atlas Vector Search")

# Setup clients
@st.cache_resource
def init_clients():
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    mongo_uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB", "telco_copilot")
    collection_name = os.environ.get("MONGODB_COLLECTION", "tickets")

    if not voyage_key or not mongo_uri:
        st.error("Missing VOYAGE_API_KEY or MONGODB_URI in environment variables.")
        st.stop()

    v_client = voyageai.Client(api_key=voyage_key)
    m_client = MongoClient(mongo_uri)
    coll = m_client[db_name][collection_name]
    return v_client, coll

voyage_client, collection = init_clients()

# Sidebar options & filtering
with st.sidebar:
    st.header("Search Filters")
    language = st.selectbox("Language Filter", options=["All", "en", "de"], index=0)
    category = st.selectbox(
        "Telco Category Filter",
        options=[
            "All",
            "Technical Support",
            "Network Outage & Maintenance",
            "Billing & Payments",
            "Device & Service Support",
            "Device Returns & Exchanges",
            "Sales & Upgrades",
            "General Customer Service",
            "General Inquiry",
        ],
        index=0,
    )
    
    st.divider()
    top_k = st.slider("Top Reranked Results", min_value=1, max_value=10, value=5)

lang_filter = None if language == "All" else language
cat_filter = None if category == "All" else category

# Input Form
st.subheader("1. Incoming Incident")
incident_text = st.text_area(
    "Describe the issue or paste a new support ticket:",
    value=EXAMPLE_INCIDENT,
    height=100,
)

search_button = st.button("Find Similar Incidents", type="primary")

if search_button and incident_text.strip():
    with st.spinner("Embedding query, querying Atlas Vector Search, and reranking..."):
        # 1. Vector Search
        query_vector = embed_query(voyage_client, incident_text)
        candidates = vector_search(
            collection,
            query_vector,
            language=lang_filter,
            telco_category=cat_filter,
        )

        if not candidates:
            st.warning("No candidate tickets found. Check your search filters or index status.")
            st.stop()

        # 2. Rerank
        reranked = rerank(
            voyage_client,
            incident_text,
            candidates,
            top_k=top_k,
        )

    st.divider()
    st.subheader("2. Retrieval Results Comparison")

    col1, col2 = st.columns(2)

    # Column 1: Vector Search Only
    with col1:
        st.markdown("### 🔍 Vector Search Only (`voyage-3`)")
        st.caption("Approximate Nearest Neighbor (ANN) cosine similarity scores")

        for idx, item in enumerate(candidates[:top_k], 1):
            with st.expander(
                f"**#{idx} [{item['ticket_id']}]** Score: `{item['score']:.4f}` — {item['subject']}"
            ):
                st.markdown(f"**Category:** `{item['telco_category']}` | **Lang:** `{item['language']}` | **Priority:** `{item.get('priority')}`")
                st.markdown("**Ticket Body:**")
                st.text(item["text_blob"])
                st.markdown("**Resolution:**")
                st.info(item.get("resolution", "No resolution recorded."))

    # Column 2: Vector Search + Reranker
    with col2:
        st.markdown("### 🎯 After Reranking (`rerank-2`)")
        st.caption("Cross-encoder joint attention relevance scores")

        for idx, item in enumerate(reranked, 1):
            with st.expander(
                f"**#{idx} [{item['ticket_id']}]** Score: `{item['rerank_score']:.4f}` — {item['subject']}",
                expanded=(idx == 1),
            ):
                st.markdown(f"**Category:** `{item['telco_category']}` | **Lang:** `{item['language']}` | **Priority:** `{item.get('priority')}`")
                st.markdown("**Ticket Body:**")
                st.text(item["text_blob"])
                st.markdown("**Resolution:**")
                st.success(item.get("resolution", "No resolution recorded."))

    # Stretch Goal: Drafted Resolution Grounding
    st.divider()
    st.subheader("3. Drafted Resolution Grounding")
    st.caption("Grounding incoming resolution based on top reranked historical tickets:")
    
    top_match = reranked[0]
    top_score = top_match["rerank_score"]

    # Visual confidence badge based on cross-encoder rerank score
    if top_score >= 0.65:
        st.success(f"🎯 High Match Confidence ({top_score:.2f}) — Direct resolution match found.")
    elif top_score >= 0.50:
        st.warning(f"⚠️ Moderate Match Confidence ({top_score:.2f}) — Review resolution carefully before sending.")
    else:
        st.error(f"🚨 Low Match Confidence ({top_score:.2f}) — No exact historical ticket found. Recommend Tier-2 escalation.")

    st.markdown(f"**Top Recommended Fix (from ticket `{top_match['ticket_id']}`):**")
    st.info(top_match.get("resolution", "No historical resolution found."))