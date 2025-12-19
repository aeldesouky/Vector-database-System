## FAISS Vector Store (Detailed Discussion)

### What FAISS Stands For

FAISS stands for **Facebook AI Similarity Search**. It is an open-source library developed by Facebook AI Research (now Meta AI) specifically for **efficient similarity search and clustering of dense vectors**.

The library was introduced to address a fundamental problem in modern data systems:
how to perform fast nearest-neighbor search over **high-dimensional vector data** at scale.

---

### Historical Context and Motivation

As machine learning systems began producing dense embeddings for text, images, and audio, traditional database indexing structures such as B-trees and hash indexes became insufficient. These classical structures are optimized for exact matching or ordered keys, not for geometric similarity in high-dimensional spaces.

FAISS was designed to:

* Support **vector similarity search** efficiently
* Scale to **millions or billions of vectors**
* Provide both **exact** and **approximate nearest-neighbor (ANN)** search
* Leverage **CPU and GPU acceleration**

This made FAISS a foundational technology for systems such as:

* Recommendation engines
* Semantic search
* Large-scale retrieval-augmented generation systems

---

### FAISS as a Database Component

From a database systems perspective, FAISS can be viewed as a **specialized vector index** rather than a full database. It focuses exclusively on indexing and searching vectors, while leaving document storage, metadata management, and persistence to the surrounding system.

In this project, FAISS plays the role of a **pluggable vector index backend**, similar in spirit to how a relational database might allow different index types for different workloads.

---

### Indexing and Similarity in FAISS

FAISS supports multiple index types. In this project, we use:

* **IndexFlatIP** (Inner Product)

This index performs:

* Exact similarity search
* Linear scan over all vectors
* Optimized low-level vector operations

The inner product score is commonly used as a similarity measure for embeddings, especially when embeddings are normalized, as it approximates cosine similarity.

---

### Learning Integration with FAISS

Unlike the NumPy-based vector store, FAISS itself does not manage learning or feedback mechanisms. Instead, learning is implemented **outside the FAISS index** by adjusting scores after retrieval.

In this project:

* FAISS retrieves candidate vectors efficiently
* Learning weights are applied post-retrieval
* Final ranking reflects both similarity and user feedback

This separation preserves FAISS performance while allowing adaptive behavior.

---

### Strengths of FAISS

* Extremely fast similarity search
* Industry-standard library
* Supports large-scale vector datasets
* Well-documented and actively maintained
* Widely used in production systems

---

### Limitations of FAISS

* Limited transparency compared to raw NumPy
* Learning logic must be external
* Index rebuilding is required when vectors change
* Not a full database system on its own

---

### Why FAISS Was Chosen for This Project

FAISS was selected to:

* Demonstrate a realistic, industry-grade vector search backend
* Contrast with a transparent, educational NumPy implementation
* Highlight trade-offs between performance and interpretability
* Strengthen the database-centric nature of the project

Using both FAISS and NumPy allows the system to compare **low-level control** with **high-performance indexing**, which is directly relevant to advanced database systems.
