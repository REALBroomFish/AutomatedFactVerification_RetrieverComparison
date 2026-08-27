# Design

## 1. Overview

This repository implements and evaluates evidence-retrieval approaches for automated fact verification using the AVeriTeC dataset.

The project compares lexical, dense, hybrid, and cross-encoder-reranked retrieval. In addition to retrieval effectiveness, it evaluates computational cost and distinguishes work that must occur at query time from work that can be prepared in advance.

The repository focuses specifically on the **evidence retrieval** stage of automated fact verification. Claim detection and final veracity classification are outside the scope of the implementation.

---

## 2. Experimental Pipeline

The primary experiment evaluates four retrieval configurations:

1. BM25 lexical retrieval
2. Dense bi-encoder retrieval
3. Hybrid BM25 + dense retrieval using Reciprocal Rank Fusion
4. Cross-encoder reranking of the hybrid results

All configurations operate over the same sentence-level candidate representation.

### 2.1 BM25 Retrieval

BM25 provides the lexical retrieval baseline.

For each claim:

- the corresponding claim-specific candidate collection is loaded;
- candidate sentences are scored using BM25;
- the highest-ranked 100 candidates are retained.

The primary configuration uses:

- `k1 = 1.5`
- `b = 0.75`
- `epsilon = 0.25`
- no additional lowercasing
- retrieval depth = 100

---

### 2.2 Dense Retrieval

Dense retrieval uses the Sentence Transformers model:

`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`

Claims and candidate sentences are encoded independently.

Candidate embeddings are normalised and ranked against the encoded claim using vector similarity.

The primary configuration uses:

- batch size = 64
- retrieval depth = 100
- CUDA when available, otherwise CPU

The independent encoding of claims and candidate sentences also permits candidate representations to be prepared in advance for the deployment benchmark.

---

### 2.3 Hybrid Retrieval

Hybrid retrieval combines the independently generated BM25 and dense rankings using Reciprocal Rank Fusion (RRF).

The primary configuration uses:

- BM25 depth = 100
- dense depth = 100
- fusion depth = 100
- RRF constant = 60
- final hybrid depth = 100

RRF operates on rank positions rather than raw relevance scores. This avoids the need to directly normalise BM25 and dense scores, which exist on different numerical scales.

---

### 2.4 Cross-Encoder Reranking

The primary reranking configuration uses:

`cross-encoder/ms-marco-MiniLM-L6-v2`

The cross-encoder jointly processes the claim and each candidate sentence.

The primary experiment:

1. obtains the top 100 hybrid candidates;
2. scores each claim-candidate pair with the cross-encoder;
3. reorders the candidates by cross-encoder score;
4. retains the final top 20.

The primary configuration uses:

- candidate depth = 100
- output depth = 20
- batch size = 32
- CUDA when available, otherwise CPU

---

## 3. Dataset and Candidate Representation

The experiments use the AVeriTeC development split.

The dataset revision used by the project is pinned during repository setup.

AVeriTeC supplies a separate knowledge store for each claim. The implementation therefore treats retrieval as occurring over **claim-specific candidate collections**, rather than over one shared global corpus.

### 3.1 Retrieval Unit

The final experiments use individual sentences as retrieval candidates.

Each candidate retains:

- `claim_id`
- `candidate_id`
- `source_url`
- `text`
- `sentence_start`
- `sentence_end`

All retrieval systems operate over the same candidate collections.

This ensures that differences in effectiveness are attributable to the retrieval architecture rather than differences in evidence segmentation.

---

## 4. Source-Level Evaluation

Retrieval effectiveness is evaluated against the source URLs associated with the annotated AVeriTeC evidence.

Candidate URLs and annotated evidence URLs are normalised before comparison.

The principal metrics are:

- Recall@k
- Hit@k
- Complete@k
- Mean Reciprocal Rank (MRR)

BM25, dense, and hybrid retrieval are evaluated up to rank 100.

Cross-encoder-reranked configurations retain and evaluate a maximum of 20 candidates.

Per-claim metrics are also retained to support later comparative and error analysis.

---

## 5. Software Structure

Reusable implementation code is stored under `src/fact_verification/`.

```text
src/fact_verification/
├── data.py
├── evaluation.py
├── retrieval/
│   ├── baseRetriever.py
│   ├── bm25.py
│   ├── dense.py
│   ├── dpr.py             <- implemented DPR retriever, not used in experimentation
│   └── hybrid.py
└── reranking/
    ├── base.py
    └── cross_encoder.py
```

### 5.1 Data Layer

`data.py` provides the shared data interfaces used by the notebooks and retrieval implementations.

Its principal responsibilities include:

- loading AVeriTeC claims;
- accessing claim-specific knowledge stores;
- constructing the candidate representation used by the retrieval systems.

The main interfaces are:

- `load_claims`
- `CandidateStore`

---

### 5.2 Retrieval Layer

The retrieval modules implement the first-stage retrieval approaches.

The principal classes are:

- `BM25Retriever`
- `DenseRetriever`
- `HybridRetriever`

BM25 and dense retrieval produce independent ranked outputs.

Hybrid retrieval consumes those existing rankings and combines them using RRF rather than modifying the underlying BM25 or dense results.

---

### 5.3 Reranking Layer

The reranking layer contains:

- `CrossEncoderReranker`

The cross-encoder operates on an existing first-stage ranking.

It does not search the original knowledge store independently. Instead, it receives a bounded candidate set and reorders those candidates according to joint claim-candidate relevance scores.

---

### 5.4 Evaluation Layer

`evaluation.py` contains the common evaluation logic.

Its responsibilities include:

- source URL normalisation;
- extraction of annotated evidence sources;
- aggregate retrieval evaluation;
- per-claim retrieval evaluation.

Using a shared evaluation implementation ensures that every retrieval configuration is assessed under the same source-level criteria.

---

## 6. Notebook Structure

The project contains three main notebooks with deliberately separate responsibilities.

### 6.1 `0_repository_and_data_setup.ipynb`

This notebook is responsible for one-time repository and dataset setup.

It:

- locates or clones the repository;
- prepares the expected local project structure;
- installs setup dependencies;
- downloads the pinned AVeriTeC annotations;
- optionally downloads and extracts the full development knowledge store;
- records the local dataset path and revision;
- creates the repository-level `.env` configuration.

It is not part of the retrieval experiment itself.

---

### 6.2 `1_data_walkthrough.ipynb`

This notebook documents and inspects the experimental inputs.

It examines:

- the AVeriTeC claim representation;
- annotated evidence;
- claim-specific knowledge stores;
- sentence-level candidate collections;
- the relationship between annotated source URLs and candidate source metadata.

It is intended as an explanatory and reproducibility aid rather than as part of the main experimental execution.

---

### 6.3 `2_retrieval_evaluation.ipynb`

This notebook contains the complete experimental workflow.

Its main stages are:

1. setup and imports;
2. experimental configuration and reproducibility controls;
3. loading the experimental data;
4. primary retrieval experiments;
5. secondary experiments;
6. comparative evaluation;
7. persistence of experimental outputs;
8. final experiment validation;

The secondary experiments contain:

- the cross-encoder reranking ablation;
- the prepared-system deployment benchmark.

---

## 7. Primary Experiment

The primary experiment evaluates all 500 AVeriTeC development claims.

The four primary configurations are:

```text
BM25
Dense
BM25 + Dense -> RRF
BM25 + Dense -> RRF -> Cross-Encoder
```

The first three systems retain their top 100 candidates.

The cross-encoder receives the hybrid top 100 and retains its final top 20.

The individual ranked outputs are preserved independently so that later stages do not overwrite earlier results.

---

## 8. Reranking Ablation Experiment

The reranking ablation examines whether cross-encoder behaviour depends on:

1. the first-stage candidate generator;
2. the number of candidates supplied to the cross-encoder.

### 8.1 Candidate Generators

Three candidate sources are evaluated:

- BM25
- dense
- hybrid

### 8.2 Candidate Depths

Each source is evaluated with cross-encoder candidate depths of:

- 20
- 50
- 100

This produces a 3 × 3 ablation matrix:

```text
BM25 -> CE@20
BM25 -> CE@50
BM25 -> CE@100

Dense -> CE@20
Dense -> CE@50
Dense -> CE@100

Hybrid -> CE@20
Hybrid -> CE@50
Hybrid -> CE@100
```

Every configuration produces a final top-20 ranking.

The `Hybrid -> CE@100` configuration is identical to the primary cross-encoder experiment and is therefore reused rather than recomputed.

---

## 9. Prepared-System Deployment Benchmark

The primary full experiment measures the cost of producing rankings from the original candidate stores.

A separate deployment benchmark examines the computational behaviour of a system whose reusable retrieval representations have already been prepared.

The purpose is to distinguish between:

- offline preparation cost;
- representation loading cost;
- online query-time cost.

### 9.1 Benchmark Sample

The deployment benchmark uses a fixed random sample of AVeriTeC development claims selected using the experiment seed.

This preserves reproducibility while avoiding manual selection of unusually small, large, easy, or difficult knowledge stores.

---

### 9.2 Preparation Stage

For each selected claim-specific knowledge store, reusable representations are created.

For BM25 this consists of:

- tokenising the candidate sentences;
- constructing the BM25 index;
- serialising the prepared index.

For dense retrieval this consists of:

- encoding every candidate sentence;
- storing the normalised candidate embeddings;
- serialising those embeddings for later reuse.

Preparation time and resource consumption are recorded separately from online retrieval.

---

### 9.3 Prepared Representation Loading

Before online retrieval can begin, prepared representations must still be loaded.

The benchmark measures:

- raw candidate collection loading;
- BM25 index loading;
- dense representation loading from disk;
- dense CPU-to-GPU transfer when CUDA is used.

These operations are kept separate from query latency.

---

### 9.4 Online Query Benchmark

Online query timing begins only after the required claim-specific representations have been loaded.

The benchmark measures configurations corresponding to:

- BM25
- dense
- hybrid
- BM25 -> cross-encoder
- dense -> cross-encoder
- hybrid -> cross-encoder

The reranked configurations are measured at candidate depths of 20, 50, and 100.

Warm-up queries are executed before recorded timing measurements.

Each measured configuration is repeated multiple times to reduce sensitivity to individual timing variation.

---

### 9.5 AVeriTeC Deployment Limitation

The AVeriTeC benchmark provides a separate knowledge store for each claim.

The deployment experiment therefore measures preparation and reuse of representations **within each supplied claim-specific store**.

It should not be interpreted as directly measuring one global reusable production index.

However, the benchmark still allows the computational distinction between representation construction and subsequent query processing to be measured explicitly.

---

## 10. Computational Measurement

The project evaluates computational performance alongside retrieval effectiveness.

Measurements include:

- wall-clock runtime;
- CPU utilisation;
- process RAM usage;
- GPU utilisation;
- GPU memory usage;
- prepared representation size;
- prepared representation loading time;
- query-stage latency.

Resource usage is sampled using `psutil`, PyTorch CUDA memory reporting, and NVIDIA Management Library bindings.

Hardware-dependent measurements are intended primarily for comparison between configurations executed within the same environment.

---

## 11. Experimental Reproducibility

The experimental design includes several safeguards intended to make long-running experiments reproducible and recoverable.

### 11.1 Random Seed

The experiment uses the fixed seed:

```text
67
```

Python, NumPy, and PyTorch random number generators are seeded from the same value.

---

### 11.2 Dataset Revision

The AVeriTeC dataset is downloaded from a pinned upstream revision rather than from an unspecified latest version.

The local dataset path and revision are stored in the repository-level `.env` file created by the setup notebook.

---

### 11.3 Experiment Configuration

Parameters capable of changing the experimental outcome are defined centrally in the retrieval notebook rather than being changed independently inside retrieval cells.

Separate configuration structures are retained for:

- the primary experiment;
- the reranking ablation;
- the deployment benchmark.

---

### 11.4 Environment Recording

The retrieval notebook records information including:

- execution timestamp;
- Python version;
- PyTorch version;
- CUDA availability;
- CUDA version;
- GPU model;
- Git commit;
- Transformers version;
- Sentence Transformers version;
- NLTK version.

The primary experiment environment is preserved independently from the later secondary-experiment environment.

---

## 12. Checkpointing and Recovery

Several experimental stages may take substantial time.

Long-running stages therefore save intermediate or completed state so that interrupted execution does not require all preceding work to be repeated.

Dense retrieval supports claim-level checkpointing during the primary experiment.

Reranking ablations save their completed rankings, resource measurements, and execution state.

The prepared-system benchmark also stores:

- completed prepared representations;
- preparation measurements;
- online query measurements;
- resource measurements;
- configuration-specific rankings.

Where partial state can exist, completed claim/configuration checks are used before resuming execution.

---

## 13. Validation Strategy

Validation is treated separately from effectiveness evaluation.

A configuration is not accepted merely because it successfully produces a ranking.

### 13.1 Primary Output Validation

Primary ranked outputs are checked for:

- expected claim coverage;
- expected number of candidates per claim;
- valid rank ranges.

---

### 13.2 Prepared Retrieval Validation

Prepared BM25 and dense retrieval are compared against their corresponding frozen primary rankings.

The candidate ordering must match exactly.

This ensures that moving expensive work into an offline preparation stage does not change retrieval behaviour.

---

### 13.3 Online Pipeline Validation

Prepared deployment configurations are compared against the completed primary and reranking-ablation experiments.

The first-stage deployment systems must reproduce:

- BM25
- dense
- hybrid

The reranked deployment configurations must reproduce their corresponding cross-encoder ablation rankings.

---

### 13.4 Final Experiment Validation

The final notebook section checks the completeness and consistency of the whole experimental dataset.

It validates:

- primary experiment coverage;
- the complete reranking ablation matrix;
- prepared representation availability;
- prepared retrieval equivalence;
- expected online query measurements;
- online ranking equivalence;
- resource measurement coverage;
- required saved result files.

The experimental run is considered complete only after these checks pass.

---

## 14. Experimental Output Structure

Experimental artefacts are stored under:

```text
results/<run_id>/
```

The primary experiment retains outputs such as:

```text
results/<run_id>/
├── bm25_results.pkl
├── dense_results.pkl
├── hybrid_results.pkl
├── cross_encoder_results.pkl
├── evaluation_summary.csv
├── resource_summary.csv
├── experiment_config.json
├── environment.json
├── ablations/
└── deployment_benchmark/
```

### 14.1 Ablation Outputs

The ablation directory contains:

- ablation configuration;
- reranked results;
- resource measurements;
- runtime state;
- aggregate evaluation summaries;
- per-claim evaluation outputs;
- secondary environment metadata.

### 14.2 Deployment Outputs

The deployment benchmark directory contains:

- deployment configuration;
- benchmark claim identifiers;
- prepared representations;
- preparation measurements;
- loading measurements;
- online query measurements;
- query resource measurements;
- validation tables;
- summary tables;
- secondary environment metadata.

---

## 15. Design Principles

The repository follows several design principles.

### Shared Inputs

All retrieval architectures use the same candidate representation and claim-specific knowledge stores.

### Independent Outputs

BM25, dense, hybrid, and reranked rankings are retained independently.

### Separation of Retrieval and Evaluation

Retrieval implementations produce rankings, while common evaluation code measures their quality.

### Separation of Preparation and Query Cost

The deployment benchmark does not treat dense candidate encoding as unavoidable query-time computation when those representations can be prepared beforehand.

### Explicit Validation

Prepared or optimised implementations must reproduce the corresponding established rankings before their computational measurements are accepted.

### Reproducibility

Dataset revision, experimental parameters, random seed, software environment, Git state, and output artefacts are retained wherever practical.

---

## 16. Scope

The repository is designed to answer questions about the effectiveness and computational requirements of evidence retrieval for automated fact verification.

It does not attempt to implement:

- claim detection;
- evidence-based veracity classification;
- a complete production fact-checking service;
- model compression or quantisation;
- a global vector database;
- iterative multi-hop retrieval.

These remain outside the experimental scope of the current project.