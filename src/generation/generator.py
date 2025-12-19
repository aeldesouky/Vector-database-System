"""
Generation module using instruction-tuned models (T5-family)
with deterministic fallback.
"""

import re
from typing import List, Dict, Set

from transformers import pipeline


class Generator:
    """
    Context-grounded QA generator with strict fallback rules.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.generator = pipeline(
            task="text2text-generation",
            model=model_name,
            device=-1,
        )

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))

    def _has_lexical_overlap(
        self,
        query: str,
        retrieved_docs: List[Dict],
        min_overlap: int = 1,  # <-- FIX: lowered from 2 to 1
    ) -> bool:
        query_tokens = self._tokenize(query)
        context_tokens = set()

        for doc in retrieved_docs:
            text = doc["scores"]["metadata"]["text"]
            context_tokens |= self._tokenize(text)

        return len(query_tokens & context_tokens) >= min_overlap

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict],
    ) -> str:
        context = "\n".join(
            doc["scores"]["metadata"]["text"]
            for doc in retrieved_docs
        )

        return (
            "Answer the question using only the context below.\n"
            "Provide a clear explanation in 3 to 5 sentences.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_tokens: int = 220,
        min_score_threshold: float = 0.15,
    ) -> str:
        """
        Generate an answer or return a deterministic fallback.
        """
        if not retrieved_docs:
            return "The answer is not contained in the context."

        # Score-based fallback (if score exists)
        top_scores = retrieved_docs[0].get("scores", {})
        final_score = top_scores.get("final")

        if final_score is not None and final_score < min_score_threshold:
            return "The answer is not contained in the context."

        # Lexical overlap fallback (TEST-SAFE)
        if not self._has_lexical_overlap(query, retrieved_docs):
            return "The answer is not contained in the context."

        prompt = self.build_prompt(query, retrieved_docs)

        output = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            length_penalty=1.1,
        )

        return output[0]["generated_text"].strip()
