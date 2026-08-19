from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from fact_verification.retrieval.baseRetriever import BaseRetriever


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    revision: str | None = None
    batch_size: int = 64
    device: str | None = None


class DenseRetriever(BaseRetriever):
    """
    lightweight dense bi-encoder retriever using SentenceTransformers

    candidate documents and queries are independently encoded into
    dense vectors. candidates are ranked using dot-product similarity

    the default model produces normalised embeddings, so dot product
    is equivalent to cosine similarity
    """

    def __init__(self, config: DenseConfig | None = None) -> None:
        self.config = config or DenseConfig()
        self.model = SentenceTransformer(self.config.model_name, revision=self.config.revision, device=self.config.device)
        self.model.eval()

    def _encode_query(self, query: str) -> torch.Tensor:
        embedding = self.model.encode_query(query, batch_size=self.config.batch_size, convert_to_tensor=True, show_progress_bar=False)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        return embedding

    def _encode_candidates(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode_document(texts, batch_size=self.config.batch_size, convert_to_tensor=True, show_progress_bar=False)

    def _retrieve(self, query: str, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        query_embedding = self._encode_query(query)

        candidate_embeddings = self._encode_candidates(candidates["text"].tolist())

        scores = torch.matmul(candidate_embeddings, query_embedding.squeeze(0))
        scores = scores.detach().cpu().numpy()

        results = candidates.copy()
        results["score"] = scores
        results = results.sort_values("score", ascending=False, kind="stable").head(k).reset_index(drop=True)
        results["rank"] = range(1, len(results) + 1)
        results["method"] = "dense"

        return results

    def name(self) -> str:
        return "dense"