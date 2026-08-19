from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
from fact_verification.reranking.base import BaseReranker


@dataclass(frozen=True)
class CrossEncoderConfig:
    """
    Configuration for cross-encoder reranking.
    """
    model_name: str = ("cross-encoder/ms-marco-MiniLM-L6-v2")

    revision: str | None = None

    batch_size: int = 32

    # None uses models default maximum length
    max_length: int | None = None

    # None lets Sentence Transformers choose an available device
    device: str | None = None


class CrossEncoderReranker(BaseReranker):
    """
    Rerank retrieved passages using joint query-passage scoring.
    """

    def __init__(self, config: CrossEncoderConfig | None = None) -> None:
        self.config = config or CrossEncoderConfig()

        self.model = CrossEncoder(self.config.model_name, revision=self.config.revision, device=self.config.device, max_length=self.config.max_length)


    @property
    def name(self) -> str:
        return "cross_encoder"


    def _rerank(self, query: str, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        
        # Ensure deterministic tie behaviour by starting from
        # the existing retrieval order where one is available.
        if "rank" in candidates.columns:
            candidates = candidates.sort_values("rank", kind="stable").reset_index(drop=True)
        else:
            candidates = candidates.reset_index(drop=True)

        # Preserve information from the first-stage retriever.
        if "score" in candidates.columns:
            candidates["first_stage_score"] = candidates["score"]
        if "rank" in candidates.columns:
            candidates["first_stage_rank"] = candidates["rank"]

        if "method" in candidates.columns:
            candidates["first_stage_method"] = candidates["method"]

        pairs = [(query, text) for text in candidates["text"].fillna("").astype(str)]

        scores = np.asarray(self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False, convert_to_numpy=True)).reshape(-1)

        # stable sort means equal scores retain the existing
        # first-stage ranking
        ranked_indices = np.argsort(-scores, kind="stable")
        ranked_indices = ranked_indices[:k]

        results = candidates.iloc[ranked_indices].copy()
        results["score"] = scores[ranked_indices]
        results["rank"] = np.arange(1, len(results) + 1)
        results["method"] = self.name

        return results.reset_index(drop=True)

    def rerank_batch(self, claims: pd.DataFrame, candidates: pd.DataFrame, candidate_k: int | None = None, output_k: int | None = None) -> pd.DataFrame:

        if "claim_id" not in candidates.columns:
            raise ValueError("Batch reranking requires candidates to contain 'claim_id'.")

        all_results = []

        for _, claim in tqdm(claims.iterrows(), total=len(claims), desc="CrossEncoderReranker", unit="claim"):
            claim_candidates = candidates[candidates["claim_id"] == claim.claim_id].copy()

            if claim_candidates.empty:
                continue

            claim_candidates = claim_candidates.sort_values("rank")

            if candidate_k is not None:
                claim_candidates = claim_candidates.head(candidate_k)

            claim_candidates = claim_candidates.drop(columns="claim_id")

            results = self.rerank(query=claim.claim, candidates=claim_candidates, k=output_k)
            results.insert(0, "claim_id", claim.claim_id)
            all_results.append(results)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)