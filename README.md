# Automated Fact Verification: Evidence Retrieval Comparison

This repository contains the implementation and experimental workflow for an MSc dissertation on **evidence retrieval for Automated Fact Verification (AFV)**.

The project compares:

- BM25 lexical retrieval;
- dense bi-encoder retrieval;
- hybrid retrieval using Reciprocal Rank Fusion (RRF);
- cross-encoder reranking.

It also evaluates how reranking changes with different candidate generators and depths, and how retrieval cost changes when reusable representations are prepared in advance.

The repository focuses on **evidence retrieval only**. It does not implement claim detection or final veracity prediction.

## Quick Start

A lightweight demo runs the four retrieval stages over a small bundled candidate collection and does **not** require the AVeriTeC dataset.

Python 3.10 or later is required.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python demo.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python demo.py
```

The first run downloads the pretrained SentenceTransformer and cross-encoder models from Hugging Face if they are not already cached.

Useful options:

```bash
python demo.py --device cpu
python demo.py --download-models
python demo.py --candidates path/to/candidates.json
```

## Full Experimental Workflow

The dissertation experiments use the **AVeriTeC development split** and claim-specific sentence candidate collections.

Install the full notebook environment:

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:

1. `notebooks/0_repository_and_data_setup.ipynb`  
   Prepares the repository, downloads/configures the pinned AVeriTeC data, and creates the local dataset configuration.

2. `notebooks/1_data_walkthrough.ipynb`  
   Optional explanatory walkthrough of the claims, annotations, knowledge stores, and sentence candidates.

3. `notebooks/2_retrieval_evaluation.ipynb`  
   Runs the complete experimental workflow, including:
   - the primary retrieval comparison;
   - candidate-generator and reranking-depth experiments;
   - the prepared-system computational benchmark;
   - effectiveness and resource evaluation;
   - validation and persistence of outputs.

The AVeriTeC knowledge store is large (approximately 10.75 GiB compressed and 34 GiB extracted), so the setup notebook leaves the download and extraction steps configurable.

## Experimental Configuration

| Component | Configuration |
| --- | --- |
| Dataset | AVeriTeC development split |
| Claims | 500 |
| Retrieval unit | Sentence |
| Random seed | 67 |
| BM25 | `rank-bm25` / BM25Okapi |
| Dense retriever | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Hybrid retrieval | Reciprocal Rank Fusion |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| First-stage depth | 100 |
| Final reranked depth | 20 |

Detailed implementation choices and experimental rationale are documented in [`DESIGN.md`](DESIGN.md).

## Repository Structure

```text
.
├── demo.py
├── examples/
│   └── demo_candidates.json
├── notebooks/
│   ├── 0_repository_and_data_setup.ipynb
│   ├── 1_data_walkthrough.ipynb
│   └── 2_retrieval_evaluation.ipynb
├── src/
│   └── fact_verification/
│       ├── data.py
│       ├── evaluation.py
│       ├── retrieval/
│       │   ├── baseRetriever.py
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── dpr.py
│       │   └── hybrid.py
│       └── reranking/
│           ├── base.py
│           └── cross_encoder.py
├── requirements.txt
├── pyproject.toml
├── DESIGN.md
├── ENVIRONMENT.md
└── README.md
```

Local dataset files are stored under `.local_data/` and are not intended to be committed.

## Results and Reproducibility

Experiment outputs are written under:

```text
results/<run_id>/
```

Each run records its configuration, environment information, aggregate and per-claim evaluation, computational measurements, and saved retrieval outputs. Long-running stages use saved state/checkpointing where appropriate.

The experimental workflow also validates that prepared implementations reproduce the corresponding frozen retrieval rankings before computational measurements are accepted.

See:

- [`DESIGN.md`](DESIGN.md) for architecture, experimental design, validation, and output structure;
- [`ENVIRONMENT.md`](ENVIRONMENT.md) for software, hardware, dataset revision, and reproducibility details.

## Dataset

AVeriTeC is not redistributed with this repository. The setup notebook obtains the required files from the pinned upstream dataset revision.

The lightweight `demo.py` workflow uses only the bundled example candidates and can be run without AVeriTeC.
