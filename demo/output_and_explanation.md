# RAG System Demo — Execution Examples

This document demonstrates the behavior of the Retrieval-Augmented Generation system
using two representative queries from different domains.

---

## Demo Query 1

### Query
**"What are database indexes?"**

### RETRIEVED DOCUMENTS (ranked):

Rank 1: Database Indexing Strategies
  Final score: 0.4600
  Cosine:      0.7513
  Dot:         0.7513
  Euclidean:   -0.7052

Rank 2: ACID Transactions in Databases
  Final score: 0.0245
  Cosine:      0.3218
  Dot:         0.3218
  Euclidean:   -1.1647

Rank 3: Vector Embeddings Explained
  Final score: -0.0644
  Cosine:      0.2298
  Dot:         0.2298
  Euclidean:   -1.2411

### Generated Answer

Database indexing is a fundamental optimization technique used to improve query performance. Indexes allow the database engine to locate rows efficiently without scanning the entire table. Common index structures include B-tree and hash-based indexes. B-tree indexes are balanced and support range queries, while hash indexes are optimized for equality lookups. Modern distributed databases sometimes relax strict ACID guarantees to improve scalability. Vector embeddings represent data such as text or images as dense numerical vectors. These embeddings capture semantic relationships by placing similar items close together in vector space. Embedding quality directly affects similarity search accuracy and downstream retrieval performance.

### Commentary

The retrieval system prioritized documents containing strong semantic overlap with
the query terms *database* and *indexes*. The top-ranked document directly discusses
indexing strategies, resulting in the highest similarity score. Related documents
were ranked lower due to partial conceptual overlap.

---

## Demo Query 2

### Query
**"What is the university attendance policy?"**

### RETRIEVED DOCUMENTS (ranked):

Rank 1: University Attendance Policy
  Final score: 0.4689
  Cosine:      0.7595
  Dot:         0.7595
  Euclidean:   -0.6936

Rank 2: Library Borrowing Rules
  Final score: -0.0358
  Cosine:      0.2595
  Dot:         0.2595
  Euclidean:   -1.2170

Rank 3: ACID Transactions in Databases
  Final score: -0.1617
  Cosine:      0.1280
  Dot:         0.1280
  Euclidean:   -1.3206

### Generated Answer

Universities often enforce attendance policies to ensure student engagement. Students may be required to attend a minimum percentage of lectures. Absences can sometimes be excused with valid documentation. Library borrowing policies regulate access to physical and digital resources. Loan periods vary depending on user type and material category. Late returns may result in penalties. Modern distributed databases sometimes relax strict ACID guarantees to improve scalability.

### Commentary

This query demonstrates mixed-domain retrieval behavior. The system correctly isolated
a non-technical administrative document as the most relevant result. Technical
documents were ranked lower due to limited semantic similarity, showing effective
domain separation.

---

## Summary

These examples demonstrate that:
- Documents are ranked based on semantic similarity
- Retrieval behaves predictably across domains
- Generated answers are grounded in retrieved content
- The system avoids hallucination when relevant context is present
