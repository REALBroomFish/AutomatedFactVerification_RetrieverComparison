from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseReranker(ABC):
    """
    Common interface for second-stage reranking systems.

    A reranker receives a query and an already retrieved candidate set,
    then returns those candidates in a new relevance order.
    """

    REQUIRED_COLUMNS = {"candidate_id", "text"}

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for the reranking method"""
        ...

    def rerank(self, query: str, candidates: pd.DataFrame, k: int | None = None) -> pd.DataFrame:

        self._validate_query(query)
        self._validate_candidates(candidates)

        if k is not None and k <= 0:
            raise ValueError("k must be greater than zero")

        if candidates.empty:
            return candidates.copy()

        if k is None:
            k = len(candidates)

        k = min(k, len(candidates))

        results = self._rerank(query=query, candidates=candidates.copy(), k=k)
        self._validate_results(results)

        return results.reset_index(drop=True)

    @abstractmethod
    def _rerank(self, query: str, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        ...

    def _validate_candidates(self, candidates: pd.DataFrame) -> None:

        missing = (self.REQUIRED_COLUMNS - set(candidates.columns))

        if missing:
            raise ValueError(f"Reranker input is missing required columns: {sorted(missing)}")

        if candidates["candidate_id"].duplicated().any():
            raise ValueError("candidate_id must be unique within a candidate set")

    @staticmethod
    def _validate_query(query: str) -> None:

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

    @staticmethod
    def _validate_results(results: pd.DataFrame) -> None:

        required = {"candidate_id", "text", "score", "rank", "method"}

        missing = required - set(results.columns)

        if missing:
            raise ValueError(f"Reranker output is missing required columns: {sorted(missing)}")

        expected_ranks = list(range(1, len(results) + 1))

        if results["rank"].tolist() != expected_ranks:
            raise ValueError("Reranked results must have contiguous ranks, starting from 1")