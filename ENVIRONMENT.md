# Experimental Environment

This document summarises the environment and reproducibility information relevant to the evidence-retrieval experiments in this repository.

Run-specific `environment.json` and `experiment_config.json` files stored alongside experimental outputs are the authoritative records for individual runs.

## Recorded Experimental Environment

The primary experiment recorded the following software and hardware configuration:

| Component | Recorded value |
| --- | --- |
| Python | 3.14.4 |
| PyTorch | 2.13.0+cu132 |
| CUDA available | Yes |
| CUDA version reported by PyTorch | 13.2 |
| GPU | NVIDIA GeForce RTX 3060 |
| Transformers | 5.15.0 |
| Sentence Transformers | 5.7.0 |
| NLTK | 3.10.2 |

The experiments were executed in a Windows development environment.

Computational measurements such as runtime, CPU utilisation, RAM usage, GPU utilisation and GPU memory usage are hardware-dependent. They should therefore be interpreted primarily as comparisons between configurations executed under the same environment.

## Creating a Compatible Environment

A Python virtual environment is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` contains the notebook and experiment dependencies, while `pip install -e .` installs the local package under `src/fact_verification/`.

PyTorch is not pinned to a platform-specific CUDA wheel because installation requirements vary by operating system, GPU, driver and Python version. A compatible PyTorch installation should therefore be selected for the target machine.

BM25 runs on the CPU. Dense retrieval and cross-encoder reranking use CUDA when available and otherwise fall back to CPU.

## Dataset Configuration

The experiments use the AVeriTeC dataset from:

```text
chenxwh/AVeriTeC
```

Pinned dataset revision:

```text
2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031
```

The dataset is not committed to this repository. `notebooks/0_repository_and_data_setup.ipynb` prepares the local dataset and records its location in a repository-local `.env` file.

Typical configuration:

```text
AVERITEC_ROOT=<local AVeriTeC dataset root>
AVERITEC_REVISION=2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031
```

The `.env` file and `.local_data/` directory are local artefacts and should not be committed.

The complete AVeriTeC development knowledge store is approximately 10.75 GiB compressed and approximately 34 GiB after extraction.

## Neural Models

The experiments use:

```text
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
cross-encoder/ms-marco-MiniLM-L6-v2
```

The completed primary run recorded the model names but did not explicitly pin Hugging Face revisions during execution. The locally cached snapshot hashes were recovered afterwards and retained as supplementary provenance.

Future exact reproductions should explicitly pin model revisions where possible.

## Randomness and Run Provenance

The experiments use random seed:

```text
67
```

The retrieval notebook applies this to Python, NumPy, PyTorch and CUDA random-number generation where available.

Each run records environment and configuration information alongside its outputs, including items such as:

- execution timestamp;
- Python and PyTorch versions;
- CUDA availability and version;
- GPU model;
- relevant library versions;
- Git commit;
- experiment configuration.

Primary run records are stored under:

```text
results/<run_id>/
```

Secondary reranking and prepared-system experiments retain their own environment/configuration records beneath their corresponding result directories.

## Computational Reproduction

Runtime and resource measurements depend on the execution environment. Differences in GPU, CPU, RAM, storage, operating-system scheduling, drivers and library versions can change absolute measurements without changing retrieval effectiveness.

For meaningful comparison with the recorded computational experiment:

- use a CUDA-capable NVIDIA GPU where possible;
- avoid unrelated CPU/GPU-intensive workloads;
- ensure sufficient RAM, VRAM and disk space;
- avoid background activity that may interfere with timing measurements.

A reproduction on different hardware may reproduce rankings and effectiveness metrics while producing different runtime and resource measurements.

## Reproduction Workflow

For a new machine:

1. clone the repository;
2. create and activate a Python virtual environment;
3. install `requirements.txt`;
4. install the repository with `pip install -e .`;
5. run `notebooks/0_repository_and_data_setup.ipynb`;
6. confirm the pinned AVeriTeC dataset is available locally;
7. run `notebooks/2_retrieval_evaluation.ipynb`.

`notebooks/1_data_walkthrough.ipynb` is optional and is intended only for inspecting the prepared dataset.

For details of the retrieval architecture, experimental design, checkpointing and validation strategy, see [`DESIGN.md`](DESIGN.md).
