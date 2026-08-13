from __future__ import annotations

from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from fact_verification.retrieval.baseRetriever import BaseRetriever

@dataclass(frozen=True)
class BM25Config:
    """Configuration for Okapi BM25."""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    lowercase: bool = False


class BM25Retriever(BaseRetriever):
    """
    Okapi BM25 lexical retriever.

    Candidates are ranked according to lexical overlap with the query.
    The retriever assumes that each row in `candidates` represents one
    retrievable unit.
    """

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()

    @property
    def name(self) -> str:
        return "bm25"

    def _tokenise(self, text: str) -> list[str]:
        """
        Tokenise text consistently for both queries and candidates.
        """

        text = str(text)

        if self.config.lowercase:
            text = text.lower()

        return nltk.word_tokenize(text, preserve_line=True)

    def _retrieve(self, query: str, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Rank candidates against a single query using Okapi BM25.
        """

        tokenised_corpus = [self._tokenise(text) for text in candidates["text"].fillna("")]
        index = BM25Okapi(tokenised_corpus, k1=self.config.k1, b=self.config.b, epsilon=self.config.epsilon)

        tokenised_query = self._tokenise(query)
        scores = np.asarray(index.get_scores(tokenised_query))

        # Stable sorting means ties retain the deterministic
        # ordering supplied by the candidate collection.
        ranked_indices = np.argsort(-scores, kind="stable")[:k]

        results = candidates.iloc[ranked_indices].copy()
        results["score"] = scores[ranked_indices]
        results["rank"] = np.arange(1, len(results) + 1)
        results["method"] = self.name

        return results