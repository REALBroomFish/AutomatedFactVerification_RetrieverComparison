from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# claims
def load_claims(root: Path, split: str) -> pd.DataFrame:
    """
    load an AVeriTeC claim split

    the official AVeriTeC files contain a JSON list where each item is
    one claim
    the list index is used as the claim ID in the supplied
    retrieval pipeline, so a claim_id column is added when one is not
    already present
    """

    root = Path(root)

    possible_paths = [root / "data" / f"{split}.json", root / f"{split}.json"]

    claim_path = next((path for path in possible_paths if path.exists()), None)

    if claim_path is None:
        checked = "\n".join(f"  - {path}" for path in possible_paths)

        raise FileNotFoundError(f"could not find AVeriTeC '{split}' claims\n Checked:\n{checked}")

    with claim_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"expected {claim_path} to contain a JSON list")

    claims = pd.DataFrame(data)

    if "claim" not in claims.columns:
        raise ValueError("claim data does not contain a 'claim' field")

    if "claim_id" not in claims.columns:
        claims.insert(0, "claim_id", range(len(claims)))

    if claims["claim_id"].duplicated().any():
        raise ValueError("claim_id values must be unique")

    return claims


class CandidateStore:
    """
    Lazy interface to the AVeriTeC knowledge store.

    Candidate data is loaded independently for each claim rather than
    loading the complete knowledge store into memory.

    Parameters
    ----------
    root:
        Root AVeriTeC directory.

    split:
        Dataset split, e.g. "dev" or "train".

    retrieval_unit:
        "sentence", "passage", or "document".

    chunk_size:
        For passage retrieval, number of neighbouring sentences in
        each passage.

    chunk_overlap:
        For passage retrieval, number of sentences shared between
        adjacent passages.

    knowledge_store_dir:
        Optional explicit path to the extracted knowledge-store
        directory. When omitted, common AVeriTeC locations are tried.
    """

    VALID_UNITS = {"sentence", "passage", "document"}


    def __init__(self, root: Path, split: str, retrieval_unit: str = "sentence", chunk_size: int | None = None, chunk_overlap: int = 0, knowledge_store_dir: Path | None = None) -> None:

        self.root = Path(root)
        self.split = split
        self.retrieval_unit = retrieval_unit
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if retrieval_unit not in self.VALID_UNITS:
            raise ValueError(f"retrieval_unit must be one of {sorted(self.VALID_UNITS)}")

        if retrieval_unit == "passage":

            if chunk_size is None:
                raise ValueError("chunk_size must be specified for passage retrieval")

            if chunk_size <= 0:
                raise ValueError("chunk_size must be greater than zero")

            if chunk_overlap < 0:
                raise ValueError("chunk_overlap cannot be negative")

            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")

        if knowledge_store_dir is not None:
            self.directory = Path(knowledge_store_dir)

        else:
            self.directory = self._find_knowledge_store()

        if not self.directory.exists():
            raise FileNotFoundError(f"knowledge store directory does not exist: {self.directory}")


    def _find_knowledge_store(self) -> Path:
        path = (self.root / "knowledge_store" / self.split / f"output_{self.split}")

        if path.exists() and path.is_dir():
            return path

        raise FileNotFoundError(f"Knowledge store not found: {path}")


    def load_claim(self, claim_id: int | str) -> pd.DataFrame:
        """
        Load and construct retrieval candidates for one claim.
        """

        path = self.directory / f"{claim_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"No knowledge-store file found for claim {claim_id}: {path}")

        candidates = []

        with path.open("r", encoding="utf-8") as file:
            for source_index, line in enumerate(file):
                line = line.strip()

                if not line:
                    continue

                try:
                    source = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {source_index + 1}") from exc

                url = source.get("url")
                sentences = source.get("url2text", [])

                if not isinstance(sentences, list):
                    continue

                sentences = [str(sentence).strip() for sentence in sentences if sentence is not None and str(sentence).strip()]

                if not sentences:
                    continue

                candidates.extend(self._create_units(claim_id=claim_id, source_index=source_index, source_url=url, sentences=sentences, source_type=source.get("type"), source_query=source.get("query")))

        return pd.DataFrame(candidates, columns=["claim_id", "candidate_id", "source_url", "text", "source_index", "sentence_start", "sentence_end", "source_type", "source_query"])


    def _create_units(self, claim_id: int | str, source_index: int, source_url: str | None, sentences: list[str], source_type: str | None, source_query: str | None) -> list[dict]:

        if self.retrieval_unit == "sentence":
            return self._sentence_units(claim_id, source_index, source_url, sentences, source_type, source_query)

        if self.retrieval_unit == "passage":
            return self._passage_units(claim_id, source_index, source_url, sentences, source_type, source_query)

        return self._document_unit(claim_id, source_index, source_url, sentences, source_type, source_query)


    @staticmethod
    def _sentence_units(claim_id, source_index, source_url, sentences, source_type, source_query) -> list[dict]:
        units = []

        for sentence_index, sentence in enumerate(sentences):
            units.append({"claim_id": claim_id,  "candidate_id": (f"{claim_id}:" f"{source_index}:" f"{sentence_index}"), 
                "source_url": source_url, "text": sentence, "source_index": source_index, "sentence_start":sentence_index, 
                "sentence_end":sentence_index, "source_type":source_type, "source_query":source_query
            })

        return units


    def _passage_units(self, claim_id, source_index, source_url, sentences, source_type, source_query) -> list[dict]:

        units = []

        step = self.chunk_size - self.chunk_overlap

        for start in range(0, len(sentences), step):
            end = min(start + self.chunk_size, len(sentences))
            passage = " ".join(sentences[start:end]).strip()

            if not passage:
                continue

            units.append({"claim_id": claim_id, "candidate_id": (f"{claim_id}:" f"{source_index}:" f"{start}-{end - 1}"),
                "source_url": source_url, "text": passage, "source_index": source_index, "sentence_start": start,
                "sentence_end": end - 1, "source_type": source_type, "source_query": source_query
            })

            if end == len(sentences):
                break

        return units


    @staticmethod
    def _document_unit(claim_id, source_index, source_url, sentences, source_type, source_query) -> list[dict]:
        text = " ".join(sentences).strip()

        if not text:
            return []

        return [{"claim_id": claim_id, "candidate_id": (f"{claim_id}:" f"{source_index}:" "document"),
            "source_url": source_url, "text": text, "source_index": source_index, "sentence_start": 0, 
            "sentence_end": len(sentences) - 1, "source_type": source_type, "source_query": source_query,
        }]


    def has_claim(self, claim_id: int | str) -> bool:
        return (self.directory/ f"{claim_id}.json").exists()


    def available_claim_ids(self) -> list[int]:
        claim_ids = []

        for path in self.directory.glob("*.json"):
            try:
                claim_ids.append(int(path.stem))
            except ValueError:
                continue

        return sorted(claim_ids)


    def __len__(self) -> int:
        return len(self.available_claim_ids())