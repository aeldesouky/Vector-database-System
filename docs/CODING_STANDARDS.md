@"
# RAG System — Advanced Database Project

This project implements a Retrieval-Augmented Generation (RAG) system with a strong focus on database concepts:
- Vector storage
- Similarity search
- Retrieval evaluation
- Lightweight generation
- Self-learning feedback loop

## Requirements
- Python 3.11+
## Setup

1. Create virtual environment
   python -m venv .venv
   .\.venv\Scripts\activate

2. Install dependencies
   pip install -r requirements.txt

3. Run unit tests
   pytest

4. Start the UI
   streamlit run src/ui/app.py

## Features
- In-memory vector database
- Multiple similarity metrics:
  - Cosine similarity
  - Dot product
  - Euclidean distance
- Retrieval visualization
- Local or free lightweight generation model
- Automatic self-learning feedback
- Full unit test coverage

## Notes
- Dataset is synthetic and mixed-domain
- System is intentionally simple and transparent
- Designed for live evaluation and testing
- Not a production solution

"@ | Set-Content README.md
