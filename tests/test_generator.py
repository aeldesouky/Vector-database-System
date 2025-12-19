"""
Unit tests for the generation module.
"""

from src.generation.generator import Generator


MODEL_NAME = "google/flan-t5-small"


def test_prompt_construction() -> None:
    generator = Generator(MODEL_NAME)

    query = "What is a database index?"
    retrieved_docs = [
        {
            "scores": {
                "metadata": {
                    "text": "Database indexes improve query performance."
                }
            }
        }
    ]

    prompt = generator.build_prompt(query, retrieved_docs)

    assert "Database indexes improve query performance" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "3 to 5 sentences" in prompt


def test_generation_returns_longer_text() -> None:
    generator = Generator(MODEL_NAME)

    query = "What is a database index?"
    retrieved_docs = [
        {
            "scores": {
                "metadata": {
                    "text": (
                        "Database indexes improve query performance by allowing "
                        "faster data access without scanning the entire table."
                    )
                }
            }
        }
    ]

    answer = generator.generate(query, retrieved_docs, max_tokens=200)

    assert isinstance(answer, str)
    assert len(answer) > 50


def test_generation_fallback_when_answer_missing() -> None:
    generator = Generator(MODEL_NAME)

    query = "What is quantum teleportation?"
    retrieved_docs = [
        {
            "scores": {
                "metadata": {
                    "text": "This document discusses database indexing only."
                }
            }
        }
    ]

    answer = generator.generate(query, retrieved_docs, max_tokens=200)

    assert "not contained in the context" in answer.lower()
def test_generation_returns_longer_text() -> None:
    generator = Generator(MODEL_NAME)

    query = "What is a database index?"
    retrieved_docs = [
        {
            "scores": {
                "final": 0.85,
                "metadata": {
                    "text": (
                        "Database indexes improve query performance by allowing "
                        "faster data access without scanning the entire table. "
                        "They are commonly implemented using B-tree or hash-based "
                        "data structures."
                    )
                }
            }
        }
    ]

    answer = generator.generate(query, retrieved_docs, max_tokens=200)

    assert isinstance(answer, str)
    assert len(answer) > 50
