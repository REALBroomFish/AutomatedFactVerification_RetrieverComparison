from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fact_verification.retrieval.baseRetriever import BaseRetriever


@dataclass(frozen=True)
class HybridConfig:
    """
    Configuration for hybrid sparse+dense retrieval.
    """

    method: Literal["rrf", "weighted"] = "rrf"

    # number of results requested from each component retriever
    # before fusion
    fusion_depth: int = 100

    # RRF rank offset constant
    rrf_constant: int = 60

    # weight assigned to the dense score for weighted fusion
    # BM25 therefore receives weight (1 - alpha)
    alpha: float = 0.5


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining lexical and dense rankings.

    The component retrievers must follow the BaseRetriever interface.
    """

    def __init__(self, lexical_retriever: BaseRetriever, dense_retriever: BaseRetriever, config: HybridConfig | None = None) -> None:
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.config = config or HybridConfig()

        if self.config.method not in {"rrf", "weighted"}:
            raise ValueError("Hybrid method must be 'rrf' or 'weighted'")

        if not 0.0 <= self.config.alpha <= 1.0:
            raise ValueError("Hybrid alpha must be in range of [0, 1]")

        if self.config.fusion_depth <= 0:
            raise ValueError("Hybrid fusion depth must be greater than 0")

        if self.config.rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than 0")


    @property
    def name(self) -> str:
        return f"hybrid-{self.config.method}"

    def _retrieve(self, query: str, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Run both component retrievers and fuse their rankings.
        """

        depth = min(max(k, self.config.fusion_depth), len(candidates))

        # runs both retrievers separately to get their top-k results
        lexical_results = self.lexical_retriever.retrieve(query=query, candidates=candidates, k=depth)
        dense_results = self.dense_retriever.retrieve(query=query, candidates=candidates, k=depth)

        # fuse the two rankings into a single ranking of the top-k results
        return self.fuse(lexical_results=lexical_results, dense_results=dense_results, k=k)


    def fuse(self, lexical_results:pd.DataFrame, dense_results:pd.DataFrame, k:int) -> pd.DataFrame:
        """
        Fuse already-computed lexical and dense rankings.

        This allows experiment notebooks to reuse cached BM25/DPR
        results rather than rerunning both retrievers.
        """

        self._validate_component_results(lexical_results, "lexical")
        self._validate_component_results(dense_results, "dense")

        if self.config.method == "rrf":
            return self._rrf_fusion(lexical_results, dense_results, k)

        return self._weighted_fusion(lexical_results, dense_results, k)

    def _rrf_fusion(self, lexical_results:pd.DataFrame, dense_results:pd.DataFrame, k:int) -> pd.DataFrame:
        """
        Reciprocal Rank Fusion.

        Each candidate receives:

            1 / (C + lexical_rank)
            +
            1 / (C + dense_rank)

        when present in both rankings.
        """

        constant = self.config.rrf_constant
        lexical = lexical_results.copy()
        dense = dense_results.copy()

        lexical["fusion_score"] = 1.0 / (constant + lexical["rank"])
        dense["fusion_score"] = 1.0 / (constant + dense["rank"])

        contributions = pd.concat([lexical[["candidate_id", "fusion_score"]], dense[["candidate_id", "fusion_score"]]], ignore_index=True)

        scores = contributions.groupby("candidate_id", as_index=False)["fusion_score"].sum()

        metadata = pd.concat([lexical_results, dense_results], ignore_index=True).drop_duplicates("candidate_id").drop(columns=["score","rank","method"], errors="ignore")

        results = metadata.merge(scores, on="candidate_id", how="inner").rename(columns={"fusion_score": "score"}).sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(k).reset_index(drop=True)
        results["rank"] = np.arange(1, len(results) + 1)
        results["method"] = self.name

        return results


    def _weighted_fusion(self, lexical_results:pd.DataFrame, dense_results:pd.DataFrame, k:int) -> pd.DataFrame:
        """
        Weighted interpolation of normalised BM25 and dense scores.
        """

        alpha = self.config.alpha

        lexical = lexical_results.copy()
        dense = dense_results.copy()

        lexical["lexical_score"] = self._minmax(lexical["score"])
        dense["dense_score"] = self._minmax(dense["score"])

        lexical_scores = lexical[["candidate_id", "lexical_score"]]
        dense_scores = dense[["candidate_id", "dense_score"]]

        combined = lexical_scores.merge(dense_scores, on="candidate_id", how="outer").fillna(0.0)
        combined["score"] = (1 - alpha) * combined["lexical_score"] + alpha * combined["dense_score"]

        metadata = pd.concat([lexical_results, dense_results], ignore_index=True).drop_duplicates("candidate_id").drop(columns=["score", "rank", "method"], errors="ignore")

        results = metadata.merge(combined[["candidate_id", "score"]], on="candidate_id", how="inner").sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(k).reset_index(drop=True)
        results["rank"] = np.arange(1, len(results) + 1)
        results["method"] = self.name

        return results

    @staticmethod
    def _minmax(scores: pd.Series) -> pd.Series:
        """
        Min-max normalise scores to [0, 1].

        If every score is identical, the ranking contains no
        discriminative score information, so all values become zero.
        """

        scores = scores.astype(float)
        minimum = scores.min()
        maximum = scores.max()

        if np.isclose(minimum, maximum):
            return pd.Series(np.zeros(len(scores)), index=scores.index)

        return (scores - minimum) / (maximum - minimum)

    @staticmethod
    def _validate_component_results(results: pd.DataFrame, name: str) -> None:

        required = {"candidate_id", "score", "rank"}

        missing = required - set(results.columns)

        if missing:
            raise ValueError(f"{name} results are missing columns: {sorted(missing)}")