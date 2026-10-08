# Project Explanation — Policy-Aware Code Compliance Auditor

This document explains the whole project: what it does, how it is laid out, how
data moves through the pipeline step by step, how the two model-training
notebooks became a Python package, and how the design compares with a real
distributed file system such as HDFS.

For a line-by-line walkthrough of the code and an interview preparation guide,
read [`details.md`](details.md).

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [From notebooks to a Python project](#2-from-notebooks-to-a-python-project)
3. [Architecture](#3-architecture)
4. [Project layout](#4-project-layout)
5. [The pipeline, stage by stage](#5-the-pipeline-stage-by-stage)
6. [The hybrid decision engine](#6-the-hybrid-decision-engine)
7. [The fine-tuning pipeline (CodeBERT + LoRA)](#7-the-fine-tuning-pipeline-codebert--lora)
8. [Configuration and runtime modes](#8-configuration-and-runtime-modes)
9. [Results from the original notebooks](#9-results-from-the-original-notebooks)
10. [Comparison with real HDFS](#10-comparison-with-real-hdfs)
11. [Limitations and honest notes](#11-limitations-and-honest-notes)
12. [Possible next steps](#12-possible-next-steps)

---

## 1. What the project does

Large codebases contain business rules that linters cannot check: payments must
be logged, discounts must come from the offer service, customer data must go
through the customer service, and so on. The auditor reads a Python codebase
and reports which functions or classes break which policy.

It answers, for every function and class in the code:

1. **What does this code do?** (a short business summary, written by an LLM)
2. **Which policies could apply?** (semantic search over the policy list)
3. **Does it violate them?** (a fine-tuned classifier decides the easy cases, and a stronger LLM decides the hard ones)
4. **Why?** (a natural-language explanation, a list of violations and a severity)

The output is a JSON file with one verdict per chunk and a PDF report for people
who do not read JSON.

The idea is that a **hybrid** system, with a small local model as a gatekeeper
in front of an expensive LLM, is cheaper and faster than sending every chunk to
the big model. It only pays for the hard cases.

---

## 2. From notebooks to a Python project

The project started as two Google Colab notebooks:

| Notebook (original) | Role | Where it lives now |
|---|---|---|
| `complete_code.ipynb` | Main pipeline: policies → Gemini embeddings → FAISS → AST chunking → LLaMA summaries → CodeBERT → Kimi K2 → PDF/JSON | Split into `src/auditix/` modules |
| `fineTuned_model.ipynb` | Trains the CodeBERT + LoRA classifier on public datasets and saves it | Split into `src/auditix/training/` |

The notebooks are kept unchanged in [`notebooks/legacy/`](notebooks/legacy) for
reference. The `git mv` command preserves their history.

### What changed and why

| Notebook problem | Change in the project |
|---|---|
| Policies hard-coded in two places, and they could drift apart | One source of truth: `policies/policies.txt` |
| Functions depended on global variables (`index`, `clf_model`, `policies`) | Explicit `AuditContext` object passed to the pipeline |
| API keys read from Colab secrets | Environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`), loaded from `.env` |
| Google Drive path hard-coded | `AUDITIX_CLASSIFIER_PATH`, defaulting to `models/policy_classifier/` |
| Many `print()` calls | `logging` with levels |
| `hash(text)` used to pick a policy for unmatched code | `hashlib.sha1`. Python's built-in `hash()` changes between runs, so the old labels were not reproducible |
| Dataset loaded in several cells, with the same snippets in several sources | Global de-duplication across all sources before the train/val/test split (see §7.4) |
| LLM verdict could override the classifier on the "trusted" fast path | Fast-path verdicts come from the classifier, as the README describes (see §6.3) |
| No tests, no dependency file | `requirements*.txt`, `pyproject.toml`, and 50 offline unit tests |
| Heavy imports (torch, transformers) at the top of every cell | Lazy imports. The audit path runs without a GPU stack |

The behaviour of the core pipeline is kept the same wherever possible. The
differences that change behaviour are listed in §11.

---

## 3. Architecture

### 3.1 Logical view

```
                 ┌──────────────────────────── OFFLINE (done once) ────────────────────────────┐
                 │                                                                              │
  policies/      │   policies.txt ──► Gemini embed (RETRIEVAL_DOCUMENT) ──► FAISS IndexFlatL2   │
  policies.txt   │                                                       models/policy_index/  │
                 │                                                                              │
  public         │   SecureCode v2 ┐                                                             │
  datasets  ───► │   CodeXGLUE     ├─► keyword policy assignment ─► dedupe ─► balance ─► split  │
                 │   clean Python ┘        (training/datasets.py)                 train/val/test│
                 │                                 │                                              │
                 │                                 ▼                                              │
                 │             CodeBERT + LoRA fine-tune (training/train.py)                     │
                 │                                 │                                              │
                 │                                 ▼                                              │
                 │                        models/policy_classifier/                               │
                 └──────────────────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────── ONLINE (per audit run) ───────────────────────────┐
  source code ─► │ chunker.py ─► summarizer (Groq LLaMA 8B) ─► Gemini embed (RETRIEVAL_QUERY)    │
  (folder or     │   (AST)              │                             │                          │
   git repo)     │                      │                             ▼                          │
                 │                      │                    FAISS search, top-3 policies        │
                 │                      │                             │                          │
                 │                      │                             ▼                          │
                 │                      │               CodeBERT p(violation) per policy        │
                 │                      │                             │                          │
                 │                      │          ┌──────────────────┴─────────────────┐        │
                 │                      │      fast path                          ambiguous       │
                 │                      │   (<35% or >65%)                        (35–65%)        │
                 │                      │          │                                  │          │
                 │                      └──► Kimi K2 explanation ◄──┘   Kimi K2 full verdict      │
                 │                                 │                                  │          │
                 │                                 └──────────────┬───────────────────┘          │
                 │                                                ▼                              │
                 │                                 verdict JSON per chunk                         │
                 │                                                ▼                              │
                 │                             audit_results.json + audit_report.pdf              │
                 └──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module dependencies

```
cli.py ──► pipeline.py ──┬─► chunker.py
                         ├─► policies.py ──► policy_index.py ──► embeddings.py (Gemini)
                         ├─► classifier.py        (torch, lazily)
                         ├─► hybrid_engine.py ──► config.py
                         ├─► llm/groq_client.py ──► llm/retry.py ──► requests (Groq)
                         └─► reporting.py         (reportlab)

cli.py ──► training/train.py ──► training/datasets.py ──► (datasets, huggingface_hub, lazily)
                             └─► training/config.py
```

The important rule is that `hybrid_engine.py` takes the classifier and the LLM as
parameters. It does not import them. This is what makes the routing logic
testable without a GPU or an API key.

### 3.3 Design principles

1. **Cheap before expensive.** Chunking, embedding search and the local classifier run first. The LLM only runs when needed.
2. **Degrade, don't crash.** A missing classifier means LLM-only mode, not an error. A rate-limited API call is retried after the time the API asks for. A file with a syntax error is still audited as one chunk.
3. **Semantic units, not line windows.** A verdict is about one function or class, which keeps the explanation specific.
4. **Configuration is data.** Model names, thresholds and paths are in `config.py` and environment variables, not scattered through the code.

---

## 4. Project layout

```
auditix-policy-aware-code-auditor/
├── README.md                    # overview, installation, usage
├── explanation.md               # this file: architecture and pipeline
├── details.md                   # code walkthrough and interview guide
├── requirements.txt             # runtime deps (audit pipeline)
├── requirements-train.txt       # fine-tuning deps (torch, transformers, peft, ...)
├── requirements-dev.txt         # pytest
├── pyproject.toml               # package metadata, `auditix` command
├── .env.example                 # template for API keys (copy to .env)
├── .gitignore                   # keeps keys, outputs and model weights out of Git
│
├── policies/
│   └── policies.txt             # the 12 compliance policies, one per line
│
├── src/auditix/                 # the Python package
│   ├── __init__.py              # version
│   ├── __main__.py              # `python -m auditix`
│   ├── cli.py                   # argparse commands: audit-local, audit-repo, build-index, train
│   ├── config.py                # all constants and environment settings
│   ├── policies.py              # load policies from .txt / .md / .pdf
│   ├── chunker.py               # AST chunking of Python files
│   ├── embeddings.py            # Gemini embedding wrapper (document / query modes)
│   ├── policy_index.py          # FAISS index build / save / load / search
│   ├── classifier.py            # CodeBERT + LoRA inference: p(violation) per (code, policy)
│   ├── hybrid_engine.py         # routing rules and verdict merging
│   ├── pipeline.py              # end-to-end orchestration and repo/folder runners
│   ├── reporting.py             # JSON and PDF output
│   ├── llm/
│   │   ├── retry.py             # Groq HTTP call with rate-limit-aware retries
│   │   └── groq_client.py       # summarise_chunk() and audit_chunk() prompts and parsing
│   └── training/
│       ├── config.py            # TrainConfig: hyperparameters and paths
│       ├── datasets.py          # loaders, keyword policy assignment, dedupe / balance / split
│       └── train.py             # tokenisation, LoRA model, Trainer, test metrics
│
├── tests/                       # 50 offline unit tests (no network, no GPU)
│
├── notebooks/legacy/            # the original notebooks, kept for reference
│   ├── complete_code.ipynb
│   └── fineTuned_model.ipynb
│
├── models/                      # generated: classifier, FAISS index, dataset cache (git-ignored)
│   └── .gitkeep
│
├── audit_report.pdf, paper.pdf, compliance_auditor_ppt.pdf   # project documents
├── code links.txt               # reference links
└── code-sample/                 # git submodule placeholder: the audit target used in the live run
```

---

## 5. The pipeline, stage by stage

The end-to-end run is `auditix audit-local <folder>` or `auditix audit-repo <url>`.
Both call `pipeline.run_audit_*`, which calls the stages below in order.

### Stage 0 — Setup (`pipeline.build_context`)

1. **Load policies** (`policies.load_policies`). The text file is split into lines. Comments, blank lines and duplicates are removed. The result is 12 statements.
2. **Build or reuse the FAISS index** (`PolicyIndex.load_or_build`). The index is embedded with Gemini in `RETRIEVAL_DOCUMENT` mode and saved under `models/policy_index/`. On later runs it is loaded from disk, and it is rebuilt only if the policy text has changed.
3. **Load the classifier** (`classifier.load_classifier`). If `models/policy_classifier/` exists and loads, the run is in **hybrid** mode. Otherwise it is in **LLM-only** mode and a warning is logged.

### Stage 1 — Discover files (`pipeline.collect_python_files`)

Walk the folder with `rglob("*.py")`. Skip directories named `.git`,
`__pycache__`, `.venv`, `venv`, `node_modules` and `migrations`. Skip files
shorter than 50 characters.

For a repository (`audit-repo`), the repo is first shallow-cloned with
`git clone --depth 1` into a temporary folder, which is deleted after the audit.

### Stage 2 — Chunk each file (`chunker.parse_code_chunks`)

Python's `ast` module parses the file. Every `FunctionDef`, `AsyncFunctionDef`
and `ClassDef` becomes a `CodeChunk` with its name, kind, and its exact source
text (`ast.get_source_segment`), dedented.

- A file that does not parse (syntax error, or a Python 2 file) becomes one chunk
  holding the whole file, so nothing is silently skipped.
- A file with no definitions (for example a settings file) also becomes one chunk.
- By default nested definitions are included, so a method is audited alone and also inside its class. `top_level_only=True` turns this off.

### Stage 3 — Summarise each chunk (`llm/groq_client.summarize_chunk`)

LLaMA 3.1 8B (`llama-3.1-8b-instant` on Groq) is asked for a 2–3 sentence
description that focuses on business logic, data access and external calls. The
summary is short and cheap, and its only job is to be a *search query* for
policy retrieval.

Example summary for a payment function:
> "This code processes a payment by updating the order status directly in the
> database and then calling the external charge API. It does not write an audit
> record."

### Stage 4 — Retrieve the relevant policies (`PolicyIndex.search`)

The summary is embedded with Gemini in `RETRIEVAL_QUERY` mode, which is the query
half of Gemini's asymmetric retrieval setup. FAISS then runs an exact L2 nearest-neighbour
search over the 12 policy vectors and returns the **top 3**.

The result is 3 policies per chunk, instead of 12, so the classifier runs 3 times
instead of 12. This is the "75% fewer classifier calls" claim in the README.

### Stage 5 — Classify each (chunk, policy) pair (`classifier.PolicyClassifier`)

For each of the 3 policies, CodeBERT is given `[CLS] code [SEP] policy [SEP]`
(truncated to 512 tokens). It returns two logits, and softmax turns them into
`p(COMPLIANT)` and `p(VIOLATION)`. Only `p(VIOLATION)` is used.

This is skipped in LLM-only mode.

### Stage 6 — Route and decide (`hybrid_engine.audit_chunk_hybrid`)

See §6 for the rules. The output of this stage is one verdict dict per chunk.

### Stage 7 — Explain and verdict with the LLM (`llm/groq_client.audit_chunk`)

Kimi K2 (`moonshotai/kimi-k2-instruct`) receives:

- the chunk name and code,
- the summary from stage 3,
- the relevant policies from stage 4,
- for hybrid mode, a hint with the classifier's probability.

It must return JSON with `compliant`, `violations`, `explanation` and `severity`.
The reply is parsed after removing Markdown fences. If parsing fails, the chunk
is recorded as non-compliant with `"JSON parse error"`, so that a human reviews it.

### Stage 8 — Aggregate and write (`pipeline._aggregate`, `reporting`)

Totals are computed per file and for the whole run:

- `chunks` — number of chunks audited,
- `violations` — chunks that are not compliant,
- `critical` — chunks with severity `critical`,
- `compliance_rate` — `(chunks − violations) / chunks × 100`, rounded.

Two files are written to the output folder (default `audit_output/`):

- `audit_results.json` — the full structured result, one entry per file and chunk.
- `audit_report.pdf` — a ReportLab document, grouped by file, with colour-coded COMPLIANT and VIOLATION lines.

### Example of one verdict

```json
{
  "name": "process_payment",
  "type": "FunctionDef",
  "summary": "Charges the customer and updates the order status directly in the database.",
  "policies_checked": [
    "No direct database updates to payment or order status are allowed.",
    "All payment transactions must be logged before and after execution using AuditLogger.",
    "All order state transitions must follow the defined workflow system."
  ],
  "compliant": false,
  "violations": ["Direct database update to order status without the workflow system"],
  "explanation": "Classifier p(violation)=91.0%. The function calls Order.objects.update(...) ...",
  "severity": "critical",
  "classifier_confidence": 0.91,
  "decision_path": "[Hybrid/high-confidence→LLM-explain]"
}
```

(The verdict above is illustrative, not output from a real run.)

---

## 6. The hybrid decision engine

### 6.1 The routing rules

Per chunk, the classifier produces three probabilities, one per retrieved policy.

| Condition over the 3 probabilities | Route | Who decides the verdict | LLM call |
|---|---|---|---|
| Any `p` in **[0.35, 0.65]** | **Ambiguous** | Kimi K2 | Full verdict, with a hint |
| No ambiguous `p`, and any `p > 0.65` | **Violation (fast path)** | CodeBERT | Explanation only (unless disabled) |
| No ambiguous `p`, and all `p < 0.35` | **Compliant (fast path)** | CodeBERT | Explanation only (unless disabled) |

The thresholds are in `config.py`: `AMBIGUITY_LOW = 0.35` and `AMBIGUITY_HIGH = 0.65`.

### 6.2 Severity bands

For a fast-path violation, severity comes from the highest probability:

| `p(violation)` | Severity |
|---|---|
| ≥ 0.92 | critical |
| 0.80 – 0.92 | high |
| 0.65 – 0.80 | medium |
| (fast-path compliant) | none |

### 6.3 Who has the final word

The README says the classifier verdict is trusted on the fast path, and that the
LLM only explains it. The notebook did not do this. It asked the LLM for a
verdict and used the LLM's `compliant` field, even when the classifier was
confident. The rewritten engine follows the README: **on the fast path, the
classifier's verdict is final, and the LLM writes the explanation and the list
of violations.** If the LLM returns no violations for a flagged chunk, the
engine lists the flagged policies instead.

### 6.4 Cost control

The LLM is called once for a summary and once more for the verdict or
explanation. The fast path skips the verdict work, not the explanation. Two cases:

- `AUDITIX_EXPLAIN_FAST_PATH=1` (default, notebook behaviour): every chunk gets an LLM explanation. The classifier saves nothing on the audit call, but its verdict is trusted.
- `AUDITIX_EXPLAIN_FAST_PATH=0`: fast-path chunks skip the audit call and get a template explanation. Only ambiguous chunks reach Kimi K2. This is the mode that delivers the "fewer LLM calls" saving. The actual saving depends on how many chunks fall in the ambiguous band, which has not been measured on the project's data.

---

## 7. The fine-tuning pipeline (CodeBERT + LoRA)

Module: `src/auditix/training/`. Entry point: `auditix train`.

### 7.1 Why fine-tune at all?

The pre-trained `microsoft/codebert-base` model knows code and natural language,
but it has never seen a policy-violation label. A classifier on top of it is
trained on labelled pairs of (code, policy), so it learns the decision the
auditor needs.

### 7.2 The model

- **Base:** `microsoft/codebert-base`, a RoBERTa encoder pre-trained on code and natural language.
- **Input:** a sentence pair, `[CLS] code [SEP] policy [SEP]`, up to 512 tokens. Attention can connect the code tokens and the policy tokens.
- **Head:** a two-way classification head (`classifier.dense` → `classifier.out_proj`), newly initialised. Label 0 is COMPLIANT and label 1 is VIOLATION.
- **LoRA adapter** (PEFT): small low-rank matrices are added to the **query** and **key** projections of the attention layers. The base weights are frozen.

| Setting | Value | Reason |
|---|---|---|
| `lora_r` | 16 | Enough capacity for a binary task |
| `lora_alpha` | 32 | 2 × r, a common scaling choice |
| `lora_dropout` | 0.1 | Regularisation |
| target modules | `query`, `key` | Attention matrices, as in the notebook |
| trainable parameters | **1,181,954 (0.94% of 125.8M)** | Measured in the notebook. The README's "~16M" figure is wrong |
| `lr` | 2e-4 | Typical for LoRA |
| `epochs` | 5 | |
| batch | 16 × 2 gradient accumulation = 32 | Fits a T4 GPU |
| schedule | cosine, 10% warm-up | |
| precision | fp16 on CUDA | |
| checkpoint choice | best epoch by validation macro-F1 | `load_best_model_at_end=True` |

### 7.3 Where the training data comes from

Every example is a (code, policy, label) triple.

| Source | Loader | What is used | Label |
|---|---|---|---|
| `scthornton/securecode-v2` (Hugging Face) | `load_securecode_v2` | Assistant turns in the JSONL files | 0 if `metadata.category == "secure"`, otherwise 1 |
| `code_x_glue_cc_defect_detection` (CodeXGLUE / Devign, Hugging Face) | `load_codexglue_defect` | `func` column, first 3000 rows | `target` (1 = defective) |
| `codeparrot/github-code` (streamed, Hugging Face) | `load_clean_python` | Python files ≥ 100 characters containing `def` | 0 (assumed clean) |

The notebook also listed CVEFixes and BigVul. CVEFixes is never loaded in the
code. BigVul failed in the notebook and was skipped. `bigcode/the-stack-smol`
returned HTTP 403 and was replaced with `codeparrot/github-code`. The README's
dataset table has been corrected to match.

**Policy pairing is keyword-based.** Each snippet is given the first policy whose
keywords appear in it (for example, `payment` → the AuditLogger policy). Code
that matches no keyword gets a stable hash-based policy. This is *weak
supervision*: the policy label was never checked by a human.

### 7.4 Merge, de-duplicate, balance, split

1. Concatenate the three sources.
2. Drop empty rows.
3. **Remove duplicates across all sources.** The notebook removed duplicates
   inside one loader only. The Devign loader and the CodeXGLUE loader used the
   same dataset, and the notebook also appended the first dataset a second time
   through a merged list. The same snippet could therefore land in both train and
   test. The project removes these duplicates globally.
4. Undersample the larger class so the data is 50/50. The notebook used 6,916 balanced samples.
5. Shuffle with seed 42, then split **80 / 10 / 10** into train, validation and test.

The result is cached in `models/ft_dataset.json`, so later runs skip the downloads.

### 7.5 Training and evaluation

- The `Trainer` evaluates on the validation split after each epoch, and keeps the epoch with the best **macro-F1**.
- After training, `evaluate_test` runs once on the held-out test split and writes `metrics.json`: accuracy, macro-F1, the confusion matrix and a per-class report.
- The adapter, tokenizer, `training_config.json`, `metrics.json` and `training_curves.png` are saved to `models/policy_classifier/`.

---

## 8. Configuration and runtime modes

All settings are in `src/auditix/config.py`, and can be overridden with environment variables.

| Variable | Default | Effect |
|---|---|---|
| `GROQ_API_KEY` (or `GROK_API_KEY`) | — | Groq API for LLaMA and Kimi K2 |
| `GEMINI_API_KEY` | — | Gemini embeddings |
| `HF_TOKEN` | — | Hugging Face downloads (training) |
| `AUDITIX_CLASSIFIER_PATH` | `models/policy_classifier` | Where the adapter is loaded from |
| `AUDITIX_INDEX_DIR` | `models/policy_index` | Where the FAISS index is cached |
| `AUDITIX_POLICIES_PATH` | `policies/policies.txt` | Policy source |
| `AUDITIX_OUTPUT_DIR` | `audit_output` | Where reports are written |
| `AUDITIX_EXPLAIN_FAST_PATH` | `1` | `0` skips LLM explanations on the fast path |
| `AUDITIX_USE_CLASSIFIER` | `1` | `0` forces LLM-only mode |

**Runtime modes**

| Mode | When | What runs |
|---|---|---|
| **Hybrid** | Classifier folder exists and loads | Summary → retrieval → CodeBERT → routing → Kimi K2 |
| **LLM-only** | No classifier, or `--no-classifier` | Summary → retrieval → Kimi K2 on every chunk |

The fallback is automatic and is recorded in the output as `audit_mode`.

---

## 9. Results from the original notebooks

These numbers come from the saved outputs of the two notebooks. They have not been re-run for this restructuring.

**Live audit (`complete_code.ipynb`, hybrid mode, Django e-commerce code):**

| Metric | Value |
|---|---|
| Python files audited | 9 |
| Chunks analysed | 337 |
| Violations | 24 |
| Critical | 0 |
| Compliance rate | 93% |

**Classifier test set (`fineTuned_model.ipynb`, T4 GPU, 5 epochs):**

| Metric | Value |
|---|---|
| Dataset: raw → balanced | 7,239 → 6,916 samples (50/50) |
| Split | 5,532 train / 691 validation / 693 test |
| Test accuracy | **0.817** |
| Macro-F1 | 0.81 |
| COMPLIANT precision / recall | 0.75 / 0.97 |
| VIOLATION precision / recall | 0.95 / 0.65 |
| Confusion matrix | TN 355, FP 11, FN 116, TP 211 |
| Training time | 959 s (about 16 minutes) |
| Final training loss | 0.40 |

**How to read these numbers**

- Violation **precision is high (0.95)** and **recall is low (0.65)**. The classifier rarely raises false alarms, but it misses about a third of the violations in its own test set. For an auditor that is a conservative trade-off, and it is the reason the ambiguous band exists.
- The test set has the same weak labels as the training set. The 0.81 macro-F1 measures how well the model learned those labels, not how well it finds real violations in unseen projects. That needs a human-labelled evaluation (see §12).

---

## 10. Comparison with real HDFS

The Hadoop Distributed File System (HDFS) is a storage system for very large
datasets across many machines. This project is a single-machine batch
pipeline. It does not use HDFS and it does not store large datasets. This section
explains where the two designs overlap, where they differ, and what would be
needed to scale this project the way HDFS scales.

### 10.1 HDFS, in brief

According to the Apache Hadoop documentation, HDFS has a master/slave design. A
cluster has one **NameNode**, which manages the namespace and the mapping of blocks
to DataNodes, and many **DataNodes**, which store the blocks. Files are split into blocks and
each block is replicated (three copies by default). DataNodes send heartbeats and block reports to the
NameNode, which decides when to re-replicate after a failure. HDFS is built for
write-once, read-many access to large files and for high throughput rather than low
latency. User data never passes through the NameNode.
(Source: [Apache Hadoop 3.3.5, HDFS Architecture](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html).)

### 10.2 Side-by-side

| Concern | Real HDFS | This project today | What it would take to scale |
|---|---|---|---|
| **Unit of storage** | Fixed-size blocks (128 MB by default in Hadoop 3) | Whole files, read into memory one at a time | Store source files as shards, e.g. Parquet/ORC with one row per chunk |
| **Unit of analysis** | Bytes; the system does not know what is inside a block | Semantic chunks: one function or class each (AST) | Same — semantic chunks are the right unit for the auditor, and blocks are just a storage detail |
| **Metadata server** | NameNode holds the namespace and block map in memory and persists it as an fsimage plus edit log | No server. `policies.json` next to `index.faiss` plays the metadata role for the policy index | Put the chunk catalogue (file, chunk name, hash, last audit) into a database |
| **Data servers** | DataNodes serve reads and writes | Local disk of the machine running the audit | Object storage or HDFS holds raw repos; workers read them |
| **Replication / durability** | Default factor 3, rack-aware placement, automatic re-replication | None. A lost index is rebuilt from `policies.txt` (cheap, a few API calls). Reports and the classifier are single copies | Keep the classifier and reports in versioned object storage. The index is cheap to regenerate, so it does not need replication |
| **Failure detection** | Heartbeats; a DataNode silent for a timeout is declared dead | Exceptions and retries only. Groq rate limits are handled by waiting the time the API asks for | Job-level retries and checkpoints per file, so a failed run resumes instead of restarting |
| **Write model** | Write once, then append or truncate only; one writer at a time | Each run overwrites `audit_results.json` and `audit_report.pdf`. Inputs are read-only | Immutable, dated run folders, for example `runs/2026-10-08T15-22/` |
| **Access model** | Hierarchical namespace, POSIX-like permissions, quotas, snapshots | Local filesystem permissions. Git gives history for the code under audit (the clone is `--depth 1`) | Keep the code under audit in git, and use HDFS or object-store permissions for raw data |
| **Data locality** | "Moving computation is cheaper than moving data": schedulers place tasks on nodes that hold the blocks | Not relevant at this size; the code is cloned to the local machine | Run one worker per data node, as Spark or MapReduce would |
| **Bottleneck** | Disk and network throughput | LLM API latency and rate limits (Groq returns HTTP 429) | Parallel workers with a shared rate limiter. Throughput is limited by the LLM, not by storage |
| **Scale** | Hundreds to thousands of nodes; tens of millions of files | One machine; 9 files and 337 chunks in the live run | Batch the chunks, and run many repositories in parallel |
| **Consistency** | Single writer per file; readers see closed files | Single process; no concurrent writers | Atomic writes (write to a temp name, then rename), and one writer per run |

### 10.3 What the comparison means

- **HDFS solves a storage problem and this project has no storage problem.**
  The inputs are source files of a few kilobytes. The project's own artifacts are
  small: the LoRA adapter is a few megabytes, and the FAISS index holds 12 vectors.
  The one big download is the base CodeBERT model (about 500 MB), which is cached
  by Hugging Face. A local folder is enough.
- **The idea that matters is data locality, and the bottleneck is the LLM.** At
  scale, the expensive work is API calls, not disk reads. Adding HDFS would not
  make the audit faster. Adding workers and a rate limiter would.
- **The most useful HDFS-style idea here is "make the expensive step rare".**
  HDFS keeps data where it is to avoid moving it. The hybrid engine keeps the
  LLM out of the path wherever the cheap classifier is confident.
- **The design already has the concepts for a distributed version.** Chunks are
  independent units, so they could be sharded. Verdicts are written per file, so
  a failed file can be retried. Policies and index are versioned by content, so
  workers can detect a change.

### 10.4 If you had to put it on a cluster

A realistic architecture for auditing, say, 10,000 repositories:

1. Raw repositories are stored as Parquet on HDFS or S3, one row per file.
2. A Spark job parses the files with the same `chunker.py` (the AST step is CPU-bound and parallel).
3. Chunks are written back as a table keyed by `(repo, file, chunk_hash)`. This makes repeat audits cheap, because unchanged chunks keep their verdict.
4. A worker pool calls Groq and Gemini with a shared token-bucket rate limiter.
5. Results go into a database and an immutable run folder.

The code in this repository is written so that steps 2 and 4 can be swapped in
without changing the routing logic, since `hybrid_engine.py` takes the LLM and
the classifier as parameters.

---

## 11. Limitations and honest notes

**Things that were wrong or unclear in the original material, and how they are handled now**

| Item | What the notebooks or README said | What the code does / what is true |
|---|---|---|
| LoRA size | "~16M parameters added" (README) | 1.18M trainable parameters, 0.94% of the model (notebook output). README corrected |
| LLM savings | "saves ~60% of LLM calls" (README) | Not measured. In the default mode the LLM still runs on every chunk to write the explanation. See §6.4 |
| Who decides on the fast path | README: classifier is trusted; notebook: LLM could override | Classifier decides now (§6.3). This changes the verdict in some cases compared with the notebook |
| Deduplication | Per-loader only, so duplicates could leak between train and test | Global dedupe before the split (§7.4). Results will differ from the notebook's 0.817 |
| Random policy fallback | `hash(text)` | `sha1`, so labels are reproducible across runs |
| Datasets | README listed BigCode/The Stack and "Vuln Detection" | Actual sources listed in §7.3 |
| Repo URL in README | `policy-aware-auditor` | Corrected to the actual repository |
| LICENSE link | Referenced but not in the repo | Still missing; see README |
| `code-sample` | Git submodule | The folder is an empty placeholder in this checkout. Run `git submodule update --init` to fetch it if you have access |

**Real limitations of the approach**

- **Weak labels.** The policy labels come from keyword rules, and the violation labels from datasets about security and defects (SecureCode, Devign). The classifier learns "this code looks like a vulnerable or defective function", which overlaps with, but is not the same as, "this code breaks *this* business rule". The 0.81 macro-F1 is measured on these same weak labels.
- **Language mismatch in one source.** CodeXGLUE defect detection is mostly C. It is used as a general defect signal.
- **"Clean" Python is assumed clean.** The compliant examples are random GitHub files with a `def`. Many of them probably contain the very patterns the policies forbid.
- **Recall is low on violations** (0.65 in the notebook run), so the classifier can miss real violations on the fast path.
- **Keyword policy assignment is noisy.** For example, the word `user` maps to the customer-PII policy, and `db` matches any word containing those letters.
- **LLM output is not guaranteed.** The JSON parser has fallbacks, but a malformed reply is recorded as a violation for human review.
- **The LLM is a judge, not a proof.** Verdicts should be read as suggestions with evidence, not as legal findings.
- **Classifier context is 512 tokens.** A long function plus a policy can be truncated. The end of the code is cut off silently.
- **PDF glyphs.** ReportLab's built-in font does not contain the tick, cross and warning symbols. They may render as boxes. The text is still readable.
- **Single-machine only.** See §10.

---

## 12. Possible next steps

1. **Human-labelled evaluation set.** Have a reviewer label 200–300 real chunks from the sample repository against the 12 policies. Report precision and recall on that set, not on the weak training labels.
2. **Measure the cost saving.** Log how many chunks take each route and how many LLM calls each mode makes, then report the real saving.
3. **Use the policies to generate labels.** Ask Kimi K2 to label a sample of training chunks against each policy, and train on those labels. This is still weak supervision, but it is closer to the real task.
4. **Retune thresholds for recall.** Lowering the violation threshold from 0.65 to around 0.5 trades some precision for better recall. Measure it on the human-labelled set before changing it.
5. **Chunk-level caching.** Key verdicts by a hash of `(chunk code, policy text, model version)`. Unchanged chunks would not be audited again.
6. **Long-context handling.** Split functions longer than 512 tokens into windows and take the maximum probability.
7. **Tests with real models.** Add a small integration test that loads a tiny checkpoint and checks the classifier path end to end.
