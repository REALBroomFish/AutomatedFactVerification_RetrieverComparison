from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

from fact_verification.retrieval.baseRetriever import BaseRetriever

@dataclass(frozen=True)
class DPRConfig:
    """
    Configuration for Dense Passage Retrieval.
    """
    question_model: str = "facebook/dpr-question_encoder-multiset-base"
    context_model: str = "facebook/dpr-ctx_encoder-multiset-base"

    question_revision: str | None = None
    context_revision: str | None = None

    batch_size: int = 32
    max_length: int = 512

    use_title: bool = True

    device: str | None = None


class DPRRetriever(BaseRetriever):
    """
    Dense Passage Retrieval using separate query and context encoders.

    Each candidate is independently embedded with the context encoder.
    The query is embedded with the question encoder. Candidates are ranked
    by dot-product similarity with the query embedding.
    """
    def __init__(self, config: DPRConfig | None = None) -> None:
        self.config = config or DPRConfig()

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.question_tokenizer = AutoTokenizer.from_pretrained(self.config.question_model, revision=self.config.question_revision)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(self.config.question_model, revision=self.config.question_revision).to(self.device)

        self.context_tokenizer = AutoTokenizer.from_pretrained(self.config.context_model, revision=self.config.context_revision)
        self.context_encoder = DPRContextEncoder.from_pretrained(self.config.context_model, revision=self.config.context_revision).to(self.device)

        # disable dropout etc for inference
        self.question_encoder.eval()
        self.context_encoder.eval()


    @property
    def name(self) -> str:
        return "dpr"


    def _encode_query(self, query:str) -> torch.Tensor:
        """
        Encode a single query into a dense DPR query vector.
        """
        inputs = self.question_tokenizer(query, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            embedding = self.question_encoder(**inputs).pooler_output

        return embedding.cpu()


    def _encode_contexts(self, candidates: pd.DataFrame) -> torch.Tensor:
        """
        Encode a list of candidate contexts into dense DPR context vectors.
        """
        embeddings = []

        texts = candidates["text"].fillna("").astype(str).tolist()

        use_titles = self.config.use_title and "title" in candidates.columns
        if use_titles:
            titles = candidates["title"].fillna("").astype(str).tolist()

        batch_size = self.config.batch_size

        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]

            if use_titles:
                batch_titles = titles[start:end]
                inputs = self.context_tokenizer(batch_titles, text_pair=batch_texts, padding=True, truncation=True, max_length=self.config.max_length, return_tensors="pt")
            else:
                inputs = self.context_tokenizer(batch_texts, padding=True, truncation=True, max_length=self.config.max_length, return_tensors="pt")

            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                batch_embeddings = self.context_encoder(**inputs).pooler_output

            # Keep accumulated embeddings out of GPU memory.
            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings,dim=0)


    def _retrieve(self, query:str, candidates:pd.DataFrame, k:int) -> pd.DataFrame:
        """
        Rank candidates against a single query using DPR.
        """

        query_embedding = self._encode_query(query)
        context_embeddings = self._encode_contexts(candidates)

        scores = torch.matmul(context_embeddings, query_embedding.squeeze(0)).numpy()

        ranked_indices = np.argsort(-scores, kind="stable")[:k]

        results = candidates.iloc[ranked_indices].copy()

        results["score"] = scores[ranked_indices]
        results["rank"] = np.arange(1, len(results) + 1)
        results["method"] = self.name

        return results