# Experimental Environment

## 1. Overview

This document records the software, hardware, dataset, and local configuration relevant to reproducing the evidence-retrieval experiments in this repository.

The repository also records environment information automatically when experiments are executed. Run-specific `environment.json` files stored alongside experimental outputs should be treated as the authoritative record of the environment used for a particular run.

Computational measurements such as runtime, CPU utilisation, RAM usage, GPU utilisation, and GPU memory usage are hardware-dependent. They should therefore be interpreted primarily as comparisons between retrieval configurations executed under the same environment.

---

## 2. Recorded Experimental Environment

The retrieval notebook records the execution environment automatically.

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

The experiment was executed in a Windows development environment.

The exact Windows version, CPU model, and installed system RAM were not recorded by the experiment notebook and are therefore not specified here.

---

## 3. Run-Specific Environment Records

The retrieval notebook records environment metadata programmatically.

The recorded fields include:

- UTC execution timestamp;
- Python version;
- PyTorch version;
- CUDA availability;
- CUDA version;
- GPU model;
- Git commit hash;
- Transformers version;
- Sentence Transformers version;
- NLTK version.

The resulting metadata is stored with the experimental outputs.

A typical environment record contains fields equivalent to:

```text
timestamp_utc
python_version
torch_version
cuda_available
cuda_version
gpu
git_commit
transformers_version
sentence_transformers_version
nltk_version
```

These run-specific records should be preferred over this document when determining the exact environment associated with an individual experimental result.

---

## 4. Python Environment

A Python virtual environment is recommended.

From the repository root, create the environment with:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The final command installs the package under `src/fact_verification/` in editable mode.

This allows the notebooks to import the local implementation directly while preserving changes made to the source modules.

---

## 5. Python Dependencies

Direct project dependencies are listed in:

```text
requirements.txt
```

The main dependency groups are described below.

### 5.1 Numerical and Data Processing

The project uses:

- NumPy;
- pandas;
- SciPy.

These libraries support candidate manipulation, metric calculation, result aggregation, and numerical operations.

---

### 5.2 Retrieval and Neural Models

The retrieval experiments depend on:

- `rank-bm25`;
- PyTorch;
- Transformers;
- Sentence Transformers.

BM25 provides the lexical baseline.

Sentence Transformers and PyTorch provide the dense bi-encoder and cross-encoder functionality used by the neural retrieval configurations.

---

### 5.3 Evaluation and Experiment Utilities

The notebooks additionally use:

- NLTK;
- tqdm;
- matplotlib;
- psutil.

These support text processing, progress reporting, visual inspection, and process-level resource measurement.

---

### 5.4 GPU Resource Monitoring

GPU resource monitoring uses NVIDIA Management Library bindings.

The installed package is:

```text
nvidia-ml-py
```

while the Python module is imported as:

```python
pynvml
```

This is intentional.

The retrieval notebook uses NVML to obtain process-level GPU utilisation measurements where supported.

---

### 5.5 Dataset Setup

The repository setup process uses:

- `huggingface-hub`;
- `hf-xet`;
- `python-dotenv`.

These are used to:

- obtain the pinned AVeriTeC dataset files;
- support downloading the large knowledge-store archive;
- maintain repository-local dataset configuration through `.env`.

---

## 6. CUDA and GPU Execution

CUDA is not required for every part of the project.

### BM25

BM25 retrieval runs on the CPU and does not require CUDA.

### Dense Retrieval

Dense retrieval selects:

```text
cuda
```

when CUDA is available, otherwise:

```text
cpu
```

Candidate sentence encoding is the most computationally expensive part of the dense retrieval pipeline.

### Cross-Encoder Reranking

The cross-encoder similarly uses CUDA when available and otherwise falls back to CPU.

GPU execution is strongly preferable for reproducing the computational characteristics of the dissertation experiments.

---

## 7. PyTorch and CUDA Installation

The exact recorded primary environment used:

```text
PyTorch 2.13.0+cu132
CUDA 13.2
```

These exact values are recorded for reproducibility, but `requirements.txt` does not force a CUDA-specific PyTorch wheel.

This is deliberate.

PyTorch installation can depend on:

- operating system;
- installed NVIDIA driver;
- available CUDA support;
- Python version;
- hardware.

A compatible PyTorch installation should therefore be selected for the target machine.

The retrieval code automatically checks:

```python
torch.cuda.is_available()
```

and falls back to CPU where necessary.

However, runtime and resource measurements from CPU execution should not be considered directly comparable with the recorded GPU-based experimental results.

---

## 8. Dataset Configuration

The experiments use the AVeriTeC dataset.

The upstream repository used for setup is:

```text
chenxwh/AVeriTeC
```

The exact dataset revision used by the experiments is:

```text
2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031
```

Using a fixed revision prevents later upstream changes from silently modifying the experimental data.

---

## 9. Local Dataset Storage

The AVeriTeC dataset is not committed to this repository.

For a normal local installation, notebook 0 stores the dataset beneath:

```text
.local_data/
```

with a revision-specific directory beneath the AVeriTeC data directory.

The exact resolved location is registered locally rather than being hard-coded into the experiment.

The complete development knowledge store is large:

- approximately 10.75 GiB compressed;
- approximately 34 GiB after extraction.

Adequate free disk space should therefore be available before downloading and extracting it.

---

## 10. `.env` Configuration

`0_repository_and_data_setup.ipynb` creates or updates a repository-level:

```text
.env
```

file.

The file contains:

```text
AVERITEC_ROOT=<local AVeriTeC dataset root>
AVERITEC_REVISION=2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031
```

The experiment notebooks load this file using `python-dotenv`.

This allows the same repository code to operate across machines without embedding machine-specific absolute paths in tracked source files.

The `.env` file contains local configuration and should not be committed to Git.

---

## 11. Expected Local Directory Layout

After dataset setup, a typical local repository resembles:

```text
repository/
├── .env
├── .local_data/
│   └── averitec/
│       └── <dataset revision>/
│           ├── data/
│           │   ├── train.json
│           │   └── dev.json
│           └── knowledge_store/
│               └── dev/
│                   └── output_dev/
├── notebooks/
├── results/
├── src/
├── requirements.txt
├── DESIGN.md
├── ENVIRONMENT.md
└── README.md
```

The `.local_data/` and `.env` entries are local artefacts rather than repository source files.

---

## 12. Model Configuration

Two pretrained neural models are used by the primary experiments.

### 12.1 Dense Retriever

The dense retriever uses:

```text
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
```

The primary configuration uses a batch size of:

```text
64
```

Candidate and query representations are normalised before similarity scoring.

---

### 12.2 Cross-Encoder Reranker

The cross-encoder uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The primary configuration uses a batch size of:

```text
32
```

The primary experiment reranks 100 candidates and retains the final top 20.

---

## 13. Model Revision Limitation

The primary experiment configuration records the model names but does not explicitly pin individual Hugging Face model revisions.

The relevant configuration therefore identifies:

```text
sentence-transformers/multi-qa-MiniLM-L6-cos-v1
```

and:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

without a fixed model commit.

This is a reproducibility limitation.

For the completed dissertation experiments, reproducibility is strengthened by retaining:

- the ranked experimental outputs;
- saved experiment configuration;
- saved environment information;
- the Git commit used to run the experiment;
- validation checks ensuring prepared and deployment implementations reproduce the established rankings.

Future exact reproductions should retain the downloaded model versions or explicitly pin upstream model revisions where possible.

---

## 14. Randomness and Reproducibility

The experiment uses the fixed random seed:

```text
67
```

The retrieval notebook applies this seed to:

- Python's `random` module;
- NumPy;
- PyTorch;
- PyTorch CUDA random-number generation when CUDA is available.

The deployment benchmark also uses the same seed when selecting its fixed sample of AVeriTeC development claims.

---

## 15. Resource Monitoring

Computational resource measurements are collected during the experiments.

### CPU

Process CPU utilisation is sampled using:

```text
psutil
```

### RAM

Process resident memory is measured through `psutil` and reported in GiB.

### GPU Utilisation

Process-level GPU utilisation is obtained through NVIDIA Management Library bindings where supported.

### GPU Memory

GPU memory usage is obtained using PyTorch CUDA memory measurements.

---

## 16. Timing Measurements

Wall-clock measurements use Python's high-resolution:

```python
time.perf_counter()
```

CUDA operations can execute asynchronously.

Where short GPU operations are measured, CUDA synchronisation is performed around timing boundaries so that recorded wall-clock measurements include the relevant GPU work.

Warm-up queries are excluded from the measured online-query benchmark.

---

## 17. Computational Comparability

Resource and runtime measurements are dependent on the execution environment.

Important variables include:

- GPU architecture;
- available VRAM;
- CPU performance;
- system RAM;
- storage performance;
- operating-system scheduling;
- background applications;
- PyTorch version;
- CUDA version;
- model-library versions.

For this reason, the computational results are primarily intended to compare retrieval configurations **within the same experimental environment**.

A reproduction on different hardware may reproduce ranking effectiveness while producing substantially different runtime and resource measurements.

---

## 18. Primary and Secondary Experiment Environments

The primary retrieval experiment and the later secondary experiments are treated separately for environment provenance.

The primary run retains its original:

```text
results/<run_id>/environment.json
```

The reranking-ablation and deployment experiments additionally record the environment used when those experiments are executed.

Their environment records are stored beneath their respective output directories.

This prevents later execution of secondary experiments from overwriting the environment metadata associated with the original primary experiment.

---

## 19. Git Provenance

The experiment records the current Git commit using:

```text
git rev-parse HEAD
```

This allows saved results to be associated with the repository state from which they were produced.

For a final reproducible run, experimental code and documentation should therefore be committed before execution.

Any uncommitted modifications present during an experiment are not represented by the commit hash alone.

---

## 20. Result Storage

Experimental outputs are stored beneath:

```text
results/<run_id>/
```

Run-specific environment and configuration records are stored with these results.

The primary experiment records:

```text
experiment_config.json
environment.json
```

Secondary experiment directories additionally contain their own relevant configuration and environment information.

These files should be retained together with the ranked outputs when archiving experimental results.

---

## 21. Reproducing the Environment

A new machine should be prepared in the following order:

1. clone the repository;
2. create and activate a Python virtual environment;
3. install `requirements.txt`;
4. install the repository with `pip install -e .`;
5. run `0_repository_and_data_setup.ipynb`;
6. download and extract the complete AVeriTeC development knowledge store;
7. confirm that `.env` points to the correct pinned dataset;
8. optionally run `1_data_walkthrough.ipynb` to inspect the prepared data;
9. run `2_retrieval_evaluation.ipynb`.

The setup notebook should report that the retrieval dataset setup is complete before the full retrieval experiment is started.

---

## 22. Recommended Conditions for Computational Reproduction

For computational measurements that are intended to be compared with the recorded dissertation experiment:

- use an NVIDIA CUDA-capable GPU where possible;
- close unrelated GPU-intensive applications;
- avoid running other computationally heavy processes simultaneously;
- ensure sufficient free RAM and VRAM;
- ensure sufficient disk space for prepared representations;
- prevent the machine from sleeping during long-running stages;
- avoid active cloud synchronisation of large experimental artefacts while latency or disk-loading measurements are being collected.

These conditions reduce external variation in runtime and resource measurements.

---

## 23. Authoritative Reproducibility Sources

The repository contains several complementary reproducibility records.

Use them in the following order.

### Run-Specific Results

```text
results/<run_id>/environment.json
results/<run_id>/experiment_config.json
```

These contain the environment and configuration associated with an actual run.

### Secondary Experiment Records

```text
results/<run_id>/ablations/
results/<run_id>/deployment_benchmark/
```

These contain configuration and environment information associated with the secondary experiments.

### `ENVIRONMENT.md`

This document explains the overall execution environment, setup procedure, and known reproducibility limitations.

### `requirements.txt`

This lists the direct Python dependencies required to construct a compatible environment.

### Git Commit

The recorded commit identifies the repository source state associated with the experiment.

---

## 24. Known Environment Limitations

The following information was not captured by the primary environment record:

- exact CPU model;
- total installed system RAM;
- exact Windows version;
- storage-device model;
- NVIDIA driver version.

These details may affect computational measurements but do not change the defined retrieval architecture or effectiveness evaluation.

The primary model revisions were also not explicitly pinned to individual Hugging Face commits.

These limitations should be considered when attempting exact computational reproduction.