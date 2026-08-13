from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

class BaseRetriever(ABC):
    REQUIRED_CANDIDATE_COLUMNS = {"candidate_id", "source_url", "text"}
    REQUIRED_RESULT_COLUMNS = {"candidate_id", "source_url", "text", "score", "rank", "method"}

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def retrieve(self, query:str, candidates:pd.DataFrame, k:int = 100) -> pd.DataFrame:
        """
        retrieve top k candidates for a single query
        
        validation and output formatting are handled here. 
        subclasses implement only _retrieve
        """
        self._validate_query(query)
        self._validate_candidates(candidates)

        if k <= 0:
            raise ValueError("k must be greater than zero")

        if candidates.empty:
            return self._empty_results(candidates)

        results = self._retrieve(query=query, candidates=candidates.copy(), k=min(k, len(candidates)))

        self._validate_results(results)

        return results.sort_values("rank").reset_index(drop=True)


    @abstractmethod
    def _retrieve(self, query:str, candidates:pd.DataFrame, k:int) -> pd.DataFrame:
        """
        Implement retrieval for a single query
        
        Must return the standard retrieval result schema
        """

        ...

    def retrieve_batch(self, claims:pd.DataFrame, candidates:pd.DataFrame, k:int=100) -> pd.DataFrame:

        required_claim_columns = {"claim_id", "claim"}

        missing = required_claim_columns - set(claims.columns)

        if missing:
            raise ValueError(f"claims are missing req. columns: {sorted(missing)}")

        if "claim_id" not in candidates.columns:
            raise ValueError("Batch retrieval required candidates to contain 'claim_id'")

        all_results = []

        for claim in claims.itertuples(index=False):
            claim_candidates = candidates[candidates["claim_id"] == claim.claim_id]

            results = self.retrieve(query=claim.claim, candidates=claim_candidates, k=k)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def _validate_candidates(self, candidates: pd.DataFrame) -> None:

        missing = self.REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)

        if missing:
            raise ValueError(f"Candidate data is missing required columns: {sorted(missing)} ")

    def _validate_results(self, results:pd.DataFrame) -> None:
        missing = self.REQUIRED_CANDIDATE_COLUMNS - set(results.columns)

        if missing:
            raise ValueError(f"Retriever output is missing required columns: {sorted(missing)} ")

        if results["rank"].duplicated().any():
            raise ValueError("Retrieval ranks must be unique within a query.")

        expected_ranks = list(range(1, len(results) + 1))
        if results["rank"].tolist() != expected_ranks:
            raise ValueError("Retrieval ranks must start at 1 and be contiguous.")

    @staticmethod
    def _validate_query(query:str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be string")
        if not query.strip():
            raise ValueError("query cannot be empty")

    def _empty_results(self, candidates:pd.DataFrame) -> pd.DataFrame:
        results = candidates.iloc[0:0].copy()

        results["score"] = pd.Series(dtype=float)
        results["rank"] = pd.Series(dtype=int)
        results["method"] = pd.Series(dtype=str)

        return results
            