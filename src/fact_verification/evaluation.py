from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

def normalise_url(url: str | None) -> str | None:
    """
    Apply conservative URL normalisation for evidence matching.

    The scheme is ignored so that http/https variants match.
    Fragments are removed, host names are lower-cased and a
    trailing slash is removed.

    Query strings are retained because they may identify genuinely
    different resources.
    """

    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url:
        return None

    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    host = parts.netloc.lower()

    if host.startswith("www."):
        host = host[4:]

    path = parts.path.rstrip("/")

    # Deliberately omit the scheme.
    return urlunsplit(("", host, path, parts.query, ""))


def _ensure_list(value) -> list:
    """
    Convert a potentially scalar/list-like value to a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    return [value]

def extract_gold_source_urls(claim: pd.Series) -> set[str]:
    """
    extract unique annotated evidence URLs for one AVeriTeC claim

    Supports either:

    1. a preprocessed 'gold_source_urls' column or
    2. the original AVeriTeC 'questions' structure
    """

    urls: set[str] = set()

    # preferred preprocessed representation
    if "gold_source_urls" in claim.index:
        for url in _ensure_list(claim["gold_source_urls"]):
            normalised = normalise_url(url)

            if normalised:
                urls.add(normalised)

        return urls

    # original AVeriTeC annotation structure
    if "questions" not in claim.index:
        raise ValueError("claims must contain either 'gold_source_urls' or the original AVeriTeC 'questions' field")

    questions = claim["questions"]

    if not isinstance(questions, list):
        return urls

    for question in questions:
        if not isinstance(question, dict):
            continue

        answers = question.get("answers", [])

        if isinstance(answers, dict):
            answers = [answers]

        for answer in answers:
            if not isinstance(answer, dict):
                continue

            normalised = normalise_url(answer.get("source_url"))

            if normalised:
                urls.add(normalised)

    return urls

def _validate_inputs(claims: pd.DataFrame, results: pd.DataFrame, cutoffs: Iterable[int]) -> list[int]:
    """
    Validate retrieval-evaluation inputs.
    """

    required_claim_columns = {"claim_id"}

    missing_claims = required_claim_columns - set(claims.columns)

    if missing_claims:
        raise ValueError(f"claims are missing required columns: {sorted(missing_claims)}")

    required_result_columns = {"claim_id", "rank", "source_url"}

    missing_results = required_result_columns - set(results.columns)

    if missing_results:
        raise ValueError(f"results are missing required columns: {sorted(missing_results)}")

    cutoffs = sorted(set(cutoffs))

    if not cutoffs:
        raise ValueError("at least one evaluation cutoff is required")

    if any(not isinstance(k, int) or k <= 0 for k in cutoffs):
        raise ValueError("all evaluation cutoffs must be positive integers")

    return cutoffs


def evaluate_retrieval_per_claim(claims: pd.DataFrame, results: pd.DataFrame, cutoffs: Iterable[int]) -> pd.DataFrame:
    """
    calculate source-level retrieval metrics for each claim

    Metrics
    -------
    Recall@k:
        fraction of unique gold evidence sources retrieved in
        the first k results

    Hit@k:
        1 when at least one gold evidence source occurs within
        the first k results, otherwise 0

    Complete@k:
        1 when all annotated gold evidence sources occur within
        the first k results, otherwise 0

    ReciprocalRank:
        reciprocal rank of the first retrieved candidate whose
        source matches an annotated gold evidence source

    claims without any annotated source URL are marked as
    non-evaluable and are excluded from aggregate retrieval
    metrics
    """

    cutoffs = _validate_inputs(claims, results, cutoffs)

    records = []

    for _, claim in claims.iterrows():
        claim_id = claim["claim_id"]
        gold_urls = extract_gold_source_urls(claim)

        claim_results = results[results["claim_id"] == claim_id].sort_values("rank", kind="stable").copy()
        claim_results["_normalised_url"] = claim_results["source_url"].map(normalise_url)

        evaluable = len(gold_urls) > 0
        record = {"claim_id": claim_id, "evaluable": evaluable, "num_gold_sources": len(gold_urls), "num_retrieved": len(claim_results)}

        if not evaluable:
            for k in cutoffs:
                record[f"Recall@{k}"] = np.nan
                record[f"Hit@{k}"] = np.nan
                record[f"Complete@{k}"] = np.nan

            record["ReciprocalRank"] = np.nan
            records.append(record)
            continue

        # cutoff metrics
        for k in cutoffs:
            top_k = claim_results.head(k)
            retrieved_urls = {url for url in top_k["_normalised_url"] if url is not None}

            matched = gold_urls & retrieved_urls
            recall = len(matched) / len(gold_urls)

            record[f"Recall@{k}"] = recall
            record[f"Hit@{k}"] = float(len(matched) > 0)
            record[f"Complete@{k}"] = float(gold_urls.issubset(retrieved_urls))

        # reciprocal rank
        reciprocal_rank = 0.0
        for _, result in claim_results.iterrows():
            if result["_normalised_url"] in gold_urls:
                reciprocal_rank = 1.0 / result["rank"]
                break

        record["ReciprocalRank"] = reciprocal_rank
        records.append(record)

    return pd.DataFrame(records)


def evaluate_retrieval(claims: pd.DataFrame, results: pd.DataFrame, cutoffs: Iterable[int]) -> dict[str, float]:
    """
    calculate aggregate source-level retrieval metrics

    returns a flat dictionary suitable for direct use in the
    experiment comparison table
    """

    per_claim = evaluate_retrieval_per_claim(claims=claims, results=results, cutoffs=cutoffs)
    evaluable = per_claim[per_claim["evaluable"]]
    metrics: dict[str, float] = {"NumClaims": float(len(per_claim)), "NumEvaluableClaims": float(len(evaluable))}

    if evaluable.empty:
        return metrics

    for k in sorted(set(cutoffs)):
        metrics[f"Recall@{k}"] = float(evaluable[f"Recall@{k}"].mean())
        metrics[f"Hit@{k}"] = float(evaluable[f"Hit@{k}"].mean())
        metrics[f"Complete@{k}"] = float(evaluable[f"Complete@{k}"].mean())

    metrics["MRR"] = float(evaluable["ReciprocalRank"].mean())

    return metrics