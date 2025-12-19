"""
Demo script for the RAG system.

This script demonstrates:
- Retrieval behavior
- Ranking
- Generated answers
- Deterministic behavior without the UI
"""

import sys
from pathlib import Path
import json

# --- Path setup (same pattern as app.py) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import TextEmbedder
from src.retrieval.retriever import Retriever
from src.vector_store.numpy_store import NumpyVectorStore
from src.generation.generator import Generator


DATA_PATH = PROJECT_ROOT / "data" / "documents.json"


def load_documents():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_demo(query: str) -> None:
    print("=" * 80)
    print(f"QUERY: {query}\n")

    documents = load_documents()
    texts = [d["content"] for d in documents]
    metadata = [
        {"id": d["id"], "title": d["title"], "text": d["content"]}
        for d in documents
    ]

    embedder = TextEmbedder()
    vectors = embedder.embed_texts(texts)

    store = NumpyVectorStore()
    store.add(vectors, metadata)

    retriever = Retriever(embedder, store, top_k=3)
    results = retriever.retrieve(query)

    print("RETRIEVED DOCUMENTS (ranked):\n")

    for rank, result in enumerate(results, start=1):
        scores = result["scores"]
        meta = scores["metadata"]

        print(f"Rank {rank}: {meta['title']}")
        print(f"  Final score: {scores['final']:.4f}")
        print(f"  Cosine:      {scores['cosine']:.4f}")
        print(f"  Dot:         {scores['dot']:.4f}")
        print(f"  Euclidean:   {scores['euclidean']:.4f}")
        print()

    generator = Generator("google/flan-t5-base")
    answer = generator.generate(query, results)

    print("GENERATED ANSWER:\n")
    print(answer)
    print()


def main() -> None:
    run_demo("What are database indexes?")
    run_demo("What is the university attendance policy?")


if __name__ == "__main__":
    main()
