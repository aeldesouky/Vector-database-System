"""
Streamlit UI for the RAG system.
FINAL, SESSION-SAFE, PERFORMANCE-SAFE VERSION
"""

import sys
from pathlib import Path
import json
from typing import List, Dict

import streamlit as st
import numpy as np

# --- Path setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import TextEmbedder
from src.generation.generator import Generator
from src.retrieval.retriever import Retriever
from src.vector_store.numpy_store import NumpyVectorStore
from src.vector_store.faiss_store import FaissVectorStore


DATA_PATH = PROJECT_ROOT / "data" / "documents.json"


@st.cache_resource
def get_embedder() -> TextEmbedder:
    return TextEmbedder()


@st.cache_resource
def get_generator(model_name: str) -> Generator:
    return Generator(model_name)


@st.cache_data
def load_documents_cached() -> List[Dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_system(backend: str) -> None:
    documents = load_documents_cached()
    texts = [doc["content"] for doc in documents]
    metadata = [
        {"id": doc["id"], "title": doc["title"], "text": doc["content"]}
        for doc in documents
    ]

    embedder = get_embedder()
    vectors = embedder.embed_texts(texts)

    if backend == "FAISS":
        store = FaissVectorStore(dim=vectors.shape[1])
    else:
        store = NumpyVectorStore(
            vector_path=PROJECT_ROOT / "data" / "vectors.npy",
            metadata_path=PROJECT_ROOT / "data" / "metadata.json",
        )

    store.add(vectors, metadata)

    st.session_state.store = store
    st.session_state.retriever = Retriever(embedder, store, top_k=3)
    st.session_state.results = None
    st.session_state.answer = None


def ensure_session_state() -> None:
    defaults = {
        "results": None,
        "answer": None,
        "store": None,
        "retriever": None,
        "backend": None,
        "show_all_weights": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="RAG Database System", layout="wide")
    ensure_session_state()

    st.title("Retrieval-Augmented Generation System")

    documents = load_documents_cached()
    st.caption(f"Dataset size: {len(documents)} documents")

    backend = st.sidebar.selectbox("Vector Store Backend", ["NumPy", "FAISS"])

    model_name = st.sidebar.selectbox(
        "Generation Model",
        [
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=1,
    )

    if (
        st.session_state.retriever is None
        or st.session_state.backend != backend
    ):
        initialize_system(backend)
        st.session_state.backend = backend

    if st.sidebar.button("Rebuild Index"):
        initialize_system(backend)
        st.sidebar.success("Index rebuilt.")

    # --- Query + Controls ---
    col_q, col_tok, col_score = st.columns([4, 1, 1])

    with col_q:
        query = st.text_input("Enter your query")

    with col_tok:
        max_tokens = st.number_input(
            "Max tokens",
            min_value=50,
            max_value=500,
            value=220,
            step=10,
        )

    with col_score:
        min_score_threshold = st.number_input(
            "Min score",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05,
            format="%.2f",
        )

    # --- Search ---
    if st.button("Search"):
        if query.strip():
            results = st.session_state.retriever.retrieve(query)
            st.session_state.results = sorted(
                results,
                key=lambda r: r["scores"].get("final", 0.0),
                reverse=True,
            )

            generator = get_generator(model_name)
            st.session_state.answer = generator.generate(
                query,
                st.session_state.results,
                max_tokens=max_tokens,
                min_score_threshold=min_score_threshold,
            )

    # --- Display ---
    if st.session_state.results:
        st.subheader("Generated Answer")
        st.write(st.session_state.answer)

        st.subheader("Retrieved Documents")

        for rank, result in enumerate(
            st.session_state.results, start=1
        ):
            idx = result["index"]
            scores = result["scores"]
            metadata = scores["metadata"]

            with st.expander(
                f"#{rank} · {metadata['title']} (doc index {idx})"
            ):
                metric_items = [
                    (k, v)
                    for k, v in scores.items()
                    if k != "metadata"
                ]

                cols = st.columns(len(metric_items))
                for col, (name, value) in zip(cols, metric_items):
                    with col:
                        st.caption(name)
                        st.markdown(f"**{value:.4f}**")

                st.markdown("---")
                st.write(metadata["text"])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Relevant", key=f"pos_{idx}"):
                        st.session_state.store.apply_feedback(idx, True)

                with col2:
                    if st.button("Not Relevant", key=f"neg_{idx}"):
                        st.session_state.store.apply_feedback(idx, False)

    # --- Learning Visualization (Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Learning Weights")

    if st.session_state.store:
        weights = st.session_state.store.learning_weights

        min_w = float(np.min(weights))
        max_w = float(np.max(weights))
        avg_w = float(np.mean(weights))

        st.sidebar.write(f"Min weight: {min_w:.3f}")
        st.sidebar.write(f"Max weight: {max_w:.3f}")
        st.sidebar.write(f"Avg weight: {avg_w:.3f}")

        if st.sidebar.button("View all weights"):
            st.session_state.show_all_weights = (
                not st.session_state.show_all_weights
            )

        if st.session_state.show_all_weights:
            st.sidebar.markdown("**All document weights:**")
            for i, w in enumerate(weights):
                st.sidebar.write(f"Document {i + 1}: {w:.3f}")

        if st.sidebar.button("Reset Learning"):
            st.session_state.store.learning_weights[:] = 1.0
            st.sidebar.success("Learning reset")


if __name__ == "__main__":
    main()
