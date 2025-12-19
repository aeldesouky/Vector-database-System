# Retrieval-Augmented Generation System  
Advanced Database Project

## 1. Overview

This project implements a Retrieval-Augmented Generation (RAG) system with a strong emphasis on **advanced database system principles**, including vector storage, similarity search, retrieval pipelines, and adaptive ranking through feedback-driven learning.

The system supports multiple vector store backends, configurable generation behavior, and provides a user interface for interactive querying, evaluation, and live testing.

---

## 2. System Architecture

### High-Level Architecture

```mermaid
flowchart LR
    Q[User Query]
    E[Embedding Model]
    V[Vector Store]
    R[Retriever]
    G[Generation Module]
    A[Answer]

    Q --> E
    E --> V
    V --> R
    R --> G
    G --> A
````

### Detailed Retrieval Pipeline

```mermaid
flowchart TD
    D[Documents]
    EM[Sentence Embeddings]
    NS[NumPy Store]
    FS[FAISS Store]
    S[Similarity Search]
    L[Learning Weights]
    K[Top-K Results]

    D --> EM
    EM --> NS
    EM --> FS
    NS --> S
    FS --> S
    L --> S
    S --> K
```

---

## 3. Embedding Method Selection

We use Sentence-Transformers to generate dense vector embeddings for documents and queries.

**Reasons for selection:**

* Produces semantically meaningful embeddings
* Efficient for medium-sized datasets
* Widely used and well-documented
* Suitable for both transparent and high-performance vector indexing

Embeddings are stored without normalization to allow meaningful comparison across multiple similarity metrics.

---

## 4. Vector Store Implementations

### 4.1 NumPy Vector Store

The NumPy vector store is a fully transparent, in-memory implementation designed to emphasize database fundamentals.

**Features:**

* Explicit storage of vectors as NumPy arrays
* Multiple similarity metrics:

  * Cosine similarity
  * Dot product
  * Euclidean distance
* Weighted score aggregation
* Persistent storage using `.npy` files
* Learning weights applied directly to scoring

**Strengths:**

* High transparency
* Easy to inspect and debug
* Ideal for educational and experimental settings

**Limitations:**

* Linear scan complexity
* Not suitable for very large datasets

---

### 4.2 FAISS Vector Store

The FAISS vector store uses Facebook AI Similarity Search for efficient inner-product-based vector retrieval.

**Features:**

* Optimized similarity search implementation
* In-memory index
* Efficient nearest-neighbor retrieval
* Learning-aware score adjustment performed outside the FAISS index
* Minimal metadata overhead

**Strengths:**

* High performance
* Scalable to larger datasets
* Industry-standard vector search library

**Limitations:**

* Less transparent than NumPy
* Fewer exposed similarity metrics
* Learning logic must be handled externally

For further details kindly check [FAISS_Details.md](FAISS_Details.md)

---

## 5. Retrieval and Generation Integration

The retriever embeds the user query and performs similarity search using the selected backend.

Retrieved documents are passed to the generation module, which constructs a context-aware prompt and produces a grounded answer using an instruction-tuned language model.

Key integration characteristics:

* Retrieval is performed independently of generation
* Generation is strictly constrained to retrieved content
* A configurable minimum similarity threshold determines whether generation proceeds
* Deterministic fallback responses are returned when retrieval confidence is low

This separation ensures:

* Retrieval correctness
* Generation grounded in retrieved data
* Clear modular design
* Predictable system behavior

---

## 6. Self-Learning Mechanism

The system implements a feedback-driven self-learning layer.

### Learning Flow

```mermaid
flowchart LR
    U[User Feedback]
    W[Learning Weights]
    S[Similarity Scoring]

    U --> W
    W --> S
```

**Behavior:**

* Positive feedback increases document weight
* Negative feedback decreases document weight
* Weights directly affect similarity scoring
* Updated rankings are visible on subsequent searches

The learning state is summarized using aggregate statistics (minimum, maximum, and average weights), with optional inspection of per-document weights.

This satisfies the requirement for an automatic self-learning layer.

---

## 7. Execution Examples

### Query 1

**Query:** “What are database indexes?”
**Retrieved Documents:**

* Database Indexing Strategies
* Vector Embeddings Explained

**Explanation:**
The query shares semantic overlap with indexing and retrieval concepts, resulting in high similarity scores and a grounded generated answer.

---

### Query 2

**Query:** “What is the university attendance policy?”
**Retrieved Documents:**

* University Attendance Policy

**Explanation:**
The mixed-domain dataset demonstrates that the retriever correctly isolates non-technical content, and the generator produces an answer strictly grounded in the retrieved policy document.

---

## 8. Strengths, Limitations, and Improvements

### Strengths

* Clear database-centric design
* Multiple vector backends with different trade-offs
* Transparent similarity scoring
* Configurable retrieval and generation behavior
* Interactive feedback-driven learning
* Strong unit test coverage

### Limitations

* In-memory storage only
* Linear scan for NumPy backend
* Lightweight generation model
* No persistence of learning weights across sessions

### Potential Improvements

* Persistent learning state
* Hybrid retrieval (BM25 + dense vectors)
* Larger and more diverse datasets
* Advanced relevance feedback strategies
* Approximate indexing for NumPy backend

---

## 9. Conclusion

This project demonstrates a complete, testable, and extensible RAG system with strong alignment to advanced database system principles.
It emphasizes retrieval correctness, transparency, and controlled generation, making it suitable for live evaluation and academic discussion.