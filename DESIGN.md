# Design

This document describes the software and experimental structure of the evidence-retrieval project.

For installation and environment details, see [`ENVIRONMENT.md`](ENVIRONMENT.md). For a concise repository overview and quick-start instructions, see [`README.md`](README.md).

## Retrieval Pipeline

The project evaluates four retrieval stages over the same sentence-level AVeriTeC candidate collections:

```text
Claim + claim-specific candidate store
                |
        Candidate sentences
          /           \
       BM25          Dense
          \           /
       Reciprocal Rank Fusion
                |
        Cross-encoder reranking
```

The retained outputs are:

1. BM25 lexical ranking;
2. dense bi-encoder ranking;
3. hybrid ranking produced with Reciprocal Rank Fusion (RRF);
4. cross-encoder reranking of the hybrid candidates.

BM25, dense and hybrid retrieval retain up to 100 candidates. The primary cross-encoder configuration reranks the hybrid top 100 and retains the final top 20.

All stages operate over the same underlying candidate representation so differences can be attributed to retrieval architecture rather than different evidence segmentation.

## Candidate Representation

AVeriTeC provides a separate knowledge store for each claim. The project therefore performs retrieval over claim-specific candidate collections rather than one shared global corpus.

The final experiments use individual sentences as retrieval units.

Each candidate retains fields including:

```text
claim_id
candidate_id
source_url
text
sentence_start
sentence_end
```

Candidate construction is shared across all retrieval methods.

## Software Structure

Reusable implementation code is under `src/fact_verification/`:

```text
src/fact_verification/
├── data.py
├── evaluation.py
├── retrieval/
│   ├── baseRetriever.py
│   ├── bm25.py
│   ├── dense.py
│   ├── dpr.py
│   └── hybrid.py
└── reranking/
    ├── base.py
    └── cross_encoder.py
```

### Data layer

`data.py` provides the shared interfaces for loading claims, accessing claim-specific knowledge stores, and constructing candidate collections.

### Retrieval layer

The principal retrieval classes are:

- `BM25Retriever`
- `DenseRetriever`
- `HybridRetriever`

BM25 and dense retrieval produce independent rankings. Hybrid retrieval combines those existing rankings with RRF rather than modifying either underlying result.

`dpr.py` contains an implemented DPR retriever retained for reference, but it is not used in the final experiments.

### Reranking layer

`CrossEncoderReranker` receives an existing first-stage ranking and reorders a bounded candidate set using joint claim-candidate scoring.

It does not independently search the original knowledge store.

### Evaluation layer

`evaluation.py` contains the common source-level evaluation logic, including URL normalisation, annotated-source extraction, aggregate metrics and per-claim metrics.

Keeping evaluation separate from retrieval ensures that every configuration is assessed under the same criteria.

## Notebook Workflow

The repository contains three main notebooks.

### `0_repository_and_data_setup.ipynb`

Performs one-time repository and AVeriTeC dataset setup.

This notebook is not part of the retrieval experiment itself.

### `1_data_walkthrough.ipynb`

Provides an optional walkthrough of claims, annotated evidence, knowledge stores and sentence-level candidate construction.

It is explanatory rather than required for experimental execution.

### `2_retrieval_evaluation.ipynb`

Contains the complete experimental workflow:

1. configuration and reproducibility setup;
2. loading experimental data;
3. primary retrieval experiment;
4. candidate-source and reranking-depth experiment;
5. prepared-system computational benchmark;
6. effectiveness and computational evaluation;
7. persistence of outputs;
8. final validation.

## Primary Experiment

The primary experiment evaluates all 500 AVeriTeC development claims.

The four retained configurations are:

```text
BM25
Dense
BM25 + Dense -> RRF
BM25 + Dense -> RRF -> Cross-Encoder
```

Each stage is retained independently so later processing does not overwrite the rankings produced by earlier stages.

## Candidate Source and Reranking Depth

The reranking experiment varies two factors:

- first-stage candidate generator: BM25, dense or hybrid;
- number of candidates supplied to the cross-encoder: 20, 50 or 100.

This produces a 3 × 3 configuration matrix.

Every configuration retains a final top-20 reranked output.

The `Hybrid -> CE@100` configuration is identical to the primary cross-encoder run and is therefore reused rather than recomputed.

## Prepared-System Benchmark

The computational benchmark separates reusable preparation from work that must occur at query time.

For each sampled claim-specific store, the benchmark distinguishes:

### Preparation

- BM25 tokenisation and index construction;
- dense candidate encoding;
- serialisation of reusable representations.

### Loading

- candidate collection loading;
- prepared BM25 index loading;
- dense representation loading;
- dense CPU-to-GPU transfer where applicable.

### Online query processing

- BM25 retrieval;
- dense retrieval;
- hybrid fusion;
- cross-encoder reranking at depths 20, 50 and 100.

The benchmark uses a fixed seeded sample of claim-specific AVeriTeC stores.

Because AVeriTeC provides a separate store per claim, this experiment should not be interpreted as benchmarking one global production index. Its purpose is to isolate preparation, loading and query-time computation within the supplied retrieval setting.

## Checkpointing and Recovery

Long-running stages save intermediate or completed state where appropriate.

This includes:

- claim-level dense-retrieval checkpointing;
- saved reranking outputs;
- prepared BM25 and dense representations;
- prepared-system query and resource measurements.

Before resuming, the notebook checks which claims or configurations have already completed so that valid work is not unnecessarily repeated.

## Validation Strategy

Validation is treated separately from effectiveness evaluation.

### Primary outputs

Rankings are checked for expected claim coverage, ranking depth and valid rank values.

### Prepared retrieval

Prepared BM25 and dense retrieval must reproduce their corresponding frozen primary rankings exactly.

### Prepared pipelines

Prepared hybrid and cross-encoder configurations are compared with the corresponding primary or reranking-experiment outputs.

### Final validation

The final notebook section checks that the required experiment outputs, evaluation summaries, computational measurements and validation records are complete before the run is treated as finished.

The guiding rule is that an optimised or prepared implementation is not accepted merely because it produces plausible rankings: it must reproduce the established retrieval behaviour.

## Result Organisation

Experimental artefacts are stored beneath:

```text
results/<run_id>/
```

Primary, reranking and prepared-system outputs are kept in separate locations so later experiments do not overwrite earlier rankings or provenance.

Run-specific configuration and environment records are stored alongside the corresponding experimental outputs.

## Design Principles

The project follows five main design principles:

- **Shared inputs:** all retrieval architectures use the same candidate representation.
- **Independent outputs:** BM25, dense, hybrid and reranked rankings are retained separately.
- **Shared evaluation:** retrieval quality is measured through one common evaluation implementation.
- **Preparation/query separation:** reusable representation construction is not treated as unavoidable online cost.
- **Explicit validation:** prepared or optimised execution paths must reproduce the corresponding established rankings.

For exact environment, dataset revision and computational-reproduction information, see [`ENVIRONMENT.md`](ENVIRONMENT.md).
