#!/usr/bin/env python
"""Run a small inference-only smoke test of the retrieval pipeline.

The demo exercises the repository's BM25, dense, hybrid and cross-encoder
implementations over a bundled illustrative candidate collection. It does not
require AVeriTeC and does not train or fine-tune any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import pandas as pd
import torch

from fact_verification.reranking.cross_encoder import CrossEncoderConfig, CrossEncoderReranker
from fact_verification.retrieval.bm25 import BM25Config, BM25Retriever
from fact_verification.retrieval.dense import DenseConfig, DenseRetriever
from fact_verification.retrieval.hybrid import HybridConfig, HybridRetriever


DENSE_MODEL: Final = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DENSE_REVISION: Final = "b207367332321f8e44f96e224ef15bc607f4dbf0"
CROSS_ENCODER_MODEL: Final = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_REVISION: Final = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
DEFAULT_EXAMPLE: Final = Path(__file__).resolve().parent / "examples" / "demo_candidates.json"
REQUIRED_RESULT_COLUMNS: Final = {"rank", "candidate_id", "score", "text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repository's BM25, dense, RRF and cross-encoder retrieval stages over a small candidate collection.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_EXAMPLE, help="JSON file containing a claim and candidate list.")
    parser.add_argument("--claim", help="Optional claim overriding the claim stored in the JSON file.")
    parser.add_argument("--candidate-k", type=int, default=8, help="Number of first-stage candidates retained (default: 8).")
    parser.add_argument("--output-k", type=int, default=5, help="Number of reranked results retained/displayed (default: 5).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device (default: CUDA when available, otherwise CPU).")
    parser.add_argument("--download-models", action="store_true", help="Download/cache the pinned neural-model snapshots, then exit.")
    return parser.parse_args()


def load_example(path: Path) -> tuple[str, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate example not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("The candidate JSON must contain an object.")

    claim = payload.get("claim")
    records = payload.get("candidates")

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("The candidate JSON must contain a non-empty 'claim'.")

    if not isinstance(records, list) or not records:
        raise ValueError("The candidate JSON must contain a non-empty 'candidates' list.")

    candidates = pd.DataFrame(records)
    required = {"candidate_id", "source_url", "text"}
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(f"Candidates are missing fields: {sorted(missing)}")

    if candidates["candidate_id"].isna().any():
        raise ValueError("candidate_id values must not be null.")

    if candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique.")

    if candidates["text"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Candidate text must not be empty.")

    return claim.strip(), candidates


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return requested


def download_models() -> None:
    """Download the exact public snapshots used by this demonstration."""
    from huggingface_hub import snapshot_download

    models = (("Dense retriever", DENSE_MODEL, DENSE_REVISION), ("Cross-encoder", CROSS_ENCODER_MODEL, CROSS_ENCODER_REVISION))

    for label, model_name, revision in models:
        print(f"\n{label}: {model_name}")
        print(f"Revision: {revision}")
        path = snapshot_download(repo_id=model_name, revision=revision)
        print(f"Cached at: {path}")

    print("\nPinned model snapshots are available locally.")


def validate_ranking(name: str, results: pd.DataFrame, expected_size: int, allowed_candidate_ids: set[object]) -> None:
    """Fail loudly if a retrieval stage produces a malformed ranking."""
    missing = REQUIRED_RESULT_COLUMNS - set(results.columns)

    if missing:
        raise RuntimeError(f"{name} output is missing required columns: {sorted(missing)}")

    if len(results) != expected_size:
        raise RuntimeError(f"{name} returned {len(results)} rows; expected {expected_size}.")

    if results["candidate_id"].isna().any():
        raise RuntimeError(f"{name} returned a null candidate_id.")

    if results["candidate_id"].duplicated().any():
        raise RuntimeError(f"{name} returned duplicate candidate_id values.")

    returned_ids = set(results["candidate_id"].tolist())
    unexpected_ids = returned_ids - allowed_candidate_ids

    if unexpected_ids:
        raise RuntimeError(f"{name} returned unknown candidate IDs: {sorted(unexpected_ids)}")

    expected_ranks = list(range(1, expected_size + 1))
    actual_ranks = results["rank"].astype(int).tolist()

    if actual_ranks != expected_ranks:
        raise RuntimeError(f"{name} ranks are not contiguous 1..{expected_size}: {actual_ranks}")

    if results["score"].isna().any():
        raise RuntimeError(f"{name} returned a missing score.")


def print_results(title: str, results: pd.DataFrame) -> None:
    display = results[["rank", "candidate_id", "score", "text"]].copy()
    display["score"] = display["score"].map(lambda score: f"{float(score):.4f}")
    display["text"] = display["text"].astype(str).map(lambda text: text if len(text) <= 92 else text[:89] + "...")

    print(f"\n{title}")
    print("-" * len(title))
    print(display.to_string(index=False))


def main() -> int:
    args = parse_args()

    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be greater than zero.")

    if args.output_k <= 0:
        raise ValueError("--output-k must be greater than zero.")

    if args.download_models:
        download_models()
        return 0

    claim, candidates = load_example(args.candidates.resolve())

    if args.claim is not None:
        if not args.claim.strip():
            raise ValueError("--claim cannot be empty.")
        claim = args.claim.strip()

    device = resolve_device(args.device)
    candidate_k = min(args.candidate_k, len(candidates))
    output_k = min(args.output_k, candidate_k)

    print("Evidence Retrieval Demonstration")
    print("================================")
    print("Inference only: no training or fine-tuning is performed.")
    print(f"Device: {device}")
    print(f"Claim: {claim}")
    print(f"Candidates: {len(candidates)} (first-stage depth={candidate_k}, reranked depth={output_k})")

    bm25 = BM25Retriever(BM25Config(k1=1.5, b=0.75, epsilon=0.25, lowercase=False))
    dense = DenseRetriever(DenseConfig(model_name=DENSE_MODEL, revision=DENSE_REVISION, batch_size=64, device=device))
    cross_encoder = CrossEncoderReranker(CrossEncoderConfig(model_name=CROSS_ENCODER_MODEL, revision=CROSS_ENCODER_REVISION, batch_size=32, device=device))
    hybrid = HybridRetriever(lexical_retriever=bm25, dense_retriever=dense, config=HybridConfig(method="rrf", fusion_depth=candidate_k, rrf_constant=60))

    original_ids = set(candidates["candidate_id"].tolist())

    with torch.inference_mode():
        bm25_results = bm25.retrieve(query=claim, candidates=candidates, k=candidate_k)
        validate_ranking("BM25", bm25_results, candidate_k, original_ids)

        dense_results = dense.retrieve(query=claim, candidates=candidates, k=candidate_k)
        validate_ranking("Dense", dense_results, candidate_k, original_ids)

        hybrid_results = hybrid.fuse(lexical_results=bm25_results, dense_results=dense_results, k=candidate_k)
        validate_ranking("Hybrid", hybrid_results, candidate_k, original_ids)

        reranked_results = cross_encoder.rerank(query=claim, candidates=hybrid_results, k=output_k)
        validate_ranking("Cross-encoder", reranked_results, output_k, set(hybrid_results["candidate_id"].tolist()))

    print_results("BM25 lexical retrieval", bm25_results)
    print_results("Dense bi-encoder retrieval", dense_results)
    print_results("Hybrid retrieval (RRF)", hybrid_results)
    print_results("Hybrid candidates after cross-encoder reranking", reranked_results)

    print("\nSmoke test passed: all four stages produced valid rankings.")
    print("Scores are method-specific and should not be compared directly across tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
