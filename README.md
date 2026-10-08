# 🛡️ Policy-Aware Code Compliance Auditor

> An AI-powered hybrid system that scans codebases for business policy violations using **CodeBERT**, **LLaMA 3.1**, **Kimi K2**, **Gemini Embeddings**, and **FAISS**.

---

## 👥 Authors

| Name | GitHub |
|------|--------|
| Rithvik Matta — *System Architect* | [@rtk5](https://github.com/rtk5) |
| Rishil Abhijit — *Pipeline & Integration Engineer* | [@RishilJalisatgi](https://github.com/RishilJalisatgi) |
| Rohan — *ML Engineer / Model Fine-Tuning* | [@Rohan-134v](https://github.com/Rohan-134v) |

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Compliance Policies](#-compliance-policies)
- [RAG Pipeline](#-rag-pipeline-policy-retrieval)
- [Hybrid Decision Engine](#-hybrid-decision-engine)
- [Fine-Tuning CodeBERT with LoRA](#-fine-tuning-codebert-with-lora)
- [Training Data & Dataset Strategy](#-training-data--dataset-strategy)
- [Audit Results](#-audit-results)
- [Key Design Decisions](#-key-design-decisions)
- [Documentation](#-documentation)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Training the Classifier](#-training-the-classifier)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations](#-known-limitations)
- [Contributing](#-contributing)

---

## 🚨 Problem Statement

### Manual Review Fails at Scale
Large codebases have hundreds of functions. Human review misses violations — especially subtle ones like bypassing service layers or missing audit logs.

### Policy Violations = Real Risk
GDPR non-compliance, unlogged transactions, hardcoded discounts — these cause regulatory fines, security breaches, and audit failures.

### No Automated Policy Enforcement
Linters check syntax. No tool checks whether your code follows your company's own **business rules and architecture policies**.

---

## 💡 Solution Overview

A **hybrid AI system** that automatically audits code against business policies at any scale. It combines:

- **Semantic policy retrieval** (FAISS + Gemini Embeddings) to find which policies are relevant to each function
- **Fine-tuned CodeBERT** as a fast binary classifier (compliant / violation)
- **Dual-LLM reasoning** using LLaMA 3.1 8B for summarization and Kimi K2 for deep compliance auditing
- **PDF/JSON report generation** for downloadable audit results

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Classifier** | CodeBERT (`microsoft/codebert-base`) — fine-tuned with LoRA |
| **LLM Summarizer** | LLaMA 3.1 8B Instant via Groq API |
| **LLM Auditor** | Kimi K2 Instruct (`moonshotai`) via Groq API |
| **Embeddings** | Gemini Embedding 2 Preview (3072-dim) |
| **Vector DB** | FAISS (`faiss-cpu`) — semantic policy search |
| **Code Parsing** | Python `ast` — Abstract Syntax Tree chunking |
| **Fine-Tuning** | HuggingFace `transformers` + `PEFT` + `LoRA` |
| **Reports** | ReportLab PDF · JSON |
| **Runtime** | Python 3.10+ · any machine for auditing · CUDA GPU (T4 recommended) for training |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Repository                         │
│                   (Python .py files)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              AST Chunking (Python ast module)                │
│         Extracts functions & classes as code chunks          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         LLaMA 3.1 8B — Business Summary (2–3 sentences)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│    Gemini Embedding (RETRIEVAL_QUERY mode, 3072-dim)         │
│         ↓ FAISS nearest-neighbor search                      │
│         → Top-3 semantically relevant policies               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         CodeBERT Classifier → p(violation) per policy        │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │  Hybrid Decision    │
              │      Engine         │
         ┌────┴────┐          ┌────┴────┐
         │  > 65%  │  35–65%  │  < 35%  │
         │  HIGH   │ AMBIGUOUS│  HIGH   │
         │VIOLATION│   ZONE   │COMPLIANT│
         └────┬────┘    │     └────┬────┘
              │         ▼         │
              │   Kimi K2 Full    │
              │   LLM Reasoning   │
              └────────┬──────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Structured JSON Verdict per chunk                  │
│    { compliant, violations[], explanation, severity }        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           PDF + JSON Audit Report (ReportLab)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 End-to-End Workflow

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **GitHub Repo** | Clone the target repo and discover all `.py` files |
| 2 | **AST Chunking** | Use Python's `ast` module to extract functions and classes as semantically complete chunks |
| 3 | **LLaMA Summary** | LLaMA 3.1 8B generates a 2–3 sentence business-level summary of each code chunk |
| 4 | **FAISS Retrieval** | The summary is embedded with Gemini and matched against the policy FAISS index → top-3 policies returned |
| 5 | **CodeBERT** | Fine-tuned classifier runs on (chunk, policy) pairs → outputs `p(violation)` per policy |
| 6 | **Hybrid Decision** | Route based on confidence: >65% fast-path violation, <35% fast-path compliant, 35–65% → full LLM |
| 7 | **Kimi K2 Audit** | Reasoning LLM produces structured JSON verdict with violations, explanation, and severity |
| 8 | **PDF Report** | ReportLab compiles a downloadable audit report from all chunk verdicts |

---

## 📋 Compliance Policies

The system enforces **12 policies across 5 domains**:

### 💳 Payments (3 policies)
- `AuditLogger` must be used on all transaction functions
- Only approved `PaymentService` providers may be used
- No direct database updates to payment or order status

### 🏷️ Discounts (2 policies)
- No hardcoded discount values anywhere in the codebase
- All discounts must go through `OfferService`

### 🔒 Customer PII (3 policies)
- Email and personal data must only be accessed via `CustomerService`
- No direct database queries for PII fields
- All access patterns must be GDPR-compliant

### 📧 Communications (2 policies)
- All outbound emails must go through the central email service
- Customer opt-out preferences must always be respected

### 🏛️ Architecture (2 policies)
- No direct ORM model manipulation (must go through service layer)
- All critical operations must have structured logging

---

## 🔍 RAG Pipeline: Policy Retrieval

### How FAISS Retrieval Works

1. **Index time:** All 12 policies are embedded using Gemini Embedding 2 (`RETRIEVAL_DOCUMENT` mode, 3072-dim) and stored in a FAISS index.
2. **Query time:** Each code chunk → LLaMA 8B summary → Gemini Embedding (`RETRIEVAL_QUERY` mode).
3. **Search:** FAISS nearest-neighbor search returns the top-3 semantically matching policies.
4. **Efficiency:** CodeBERT runs only on these 3 policies (not all 12) → so CodeBERT makes **75% fewer classifier calls per chunk** (3 of 12).

### Key Stats

| Metric | Value |
|--------|-------|
| Embedding dimensions | 3072 |
| Policies retrieved per chunk | Top-3 |
| CodeBERT calls per chunk vs. all policies | 3 of 12 (75% fewer) |

### Why Two LLMs?

**LLaMA 3.1 8B** — Fast and cheap. Used only for generating 2–3 sentence business summaries to drive retrieval. No compliance reasoning required at this step.

**Kimi K2 Instruct** — A reasoning model. Used for deep compliance judgment with structured JSON output. By default it is called for every chunk, to write the explanation, and it gives the full verdict for ambiguous chunks. With `AUDITIX_EXPLAIN_FAST_PATH=0` it is only called for ambiguous chunks.

> Separating concerns = lower cost + higher quality on the hard task.

---

## ⚙️ Hybrid Decision Engine

The core innovation: **CodeBERT as a fast gatekeeper** before calling the expensive LLM.

```
CodeBERT p(violation)
           │
    ┌──────┴──────────────────────┐
    │                             │
  > 65%                        < 35%
  FAST PATH                   FAST PATH
HIGH CONFIDENCE             HIGH CONFIDENCE
  VIOLATION                   COMPLIANT
    │                             │
    │        35% – 65%            │
    │      AMBIGUOUS ZONE         │
    │    Full Kimi K2 reasoning   │
    │    triggered with hint      │
    │                             │
    └──────────────┬──────────────┘
                   │
         Structured JSON output:
         compliant | violations[] | explanation | severity
         Prefixed: [Hybrid/high-confidence→LLM-explain]
                or [Hybrid/ambiguous→full-LLM]
```

- **>65%:** CodeBERT verdict is trusted. The LLM is asked to justify — not re-decide. Severity is assigned.
- **35–65%:** Full Kimi K2 reasoning is triggered. The classifier hint is passed as context.
- **<35%:** CodeBERT verdict is trusted. LLM confirms and explains the clean code.

The routing reduces expensive LLM work. Whether it saves LLM *calls* depends on the mode: with `AUDITIX_EXPLAIN_FAST_PATH=0` only ambiguous chunks reach Kimi K2 (the saving has not been measured on this project's data; see [explanation.md §6.4](explanation.md#64-cost-control)).

---

## 🤖 Fine-Tuning CodeBERT with LoRA

**Module:** `src/auditix/training/` (originally `notebooks/legacy/fineTuned_model.ipynb`)

### CodeBERT Architecture

- **Base model:** `microsoft/codebert-base` — a RoBERTa-based transformer pre-trained on natural language and code (Python, Java, JS, and more)
- **Input:** `(code_chunk, policy_text)` concatenated and tokenized (max 512 tokens)
- **Fine-tuned head:** `classifier.dense → classifier.out_proj`
- **Output:** 2 logits → softmax → `p(COMPLIANT)` and `p(VIOLATION)`
- **Task:** Binary sequence classification — `0 = compliant`, `1 = violation`

### LoRA Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lora_r` | 16 | Binary classifier — higher wastes VRAM |
| `lora_alpha` | 32 | Standard 2×r for stable gradients |
| `lora_targets` | Q, K | Query + Key attention matrices |
| `lr` | 2e-4 | LoRA standard; lower = no overfit |
| `epochs` | 5 | Converges on the ~6,900-sample balanced dataset |
| `scheduler` | cosine | Smooth decay vs linear |
| `fp16` | True | Half-precision on T4 GPU |
| `effective_batch` | 32 | 16 × grad_accum of 2 |

---

## 📊 Training Data & Dataset Strategy

**Real datasets only — NO synthetic data.** Four public HuggingFace datasets are used.

| Dataset | Source | Type | Usage |
|---------|--------|------|-------|
| **SecureCode-v2** | `scthornton/securecode-v2` | Security | Assistant code turns; `category == "secure"` → compliant, otherwise violation |
| **CodeXGLUE defect (Devign)** | `code_x_glue_cc_defect_detection` | Defects | `target` 1 (defective) → violation, 0 → compliant |
| **CodeParrot GitHub Code** | `codeparrot/github-code` (streamed) | Clean code | Python files with a `def` → compliant baseline |

Not used in the final code, despite earlier notes: CodeSearchNet, CVEFixes and BigVul (BigVul failed to load and was skipped; `bigcode/the-stack-smol` returned HTTP 403).

**Policy assignment strategy:** Keyword overlap with no API calls required. The labels are weak (not human-verified). Duplicates are removed across all sources before the 80/10/10 split. See [explanation.md §7](explanation.md#7-the-fine-tuning-pipeline-codebert--lora).

```
payment  → AuditLogger policy
email    → Communications policy
discount → Discount policy
customer → Customer PII policy
...
```

---

## 📈 Audit Results — Live Run

Live run on a real e-commerce Django codebase:

| Metric | Value |
|--------|-------|
| Code chunks audited | 337 |
| Violations found | 24 |
| Overall compliance rate | **93%** |
| Python files scanned | 9 |

### File-Level Breakdown

| File | Chunks | Violations | Severity |
|------|--------|-----------|----------|
| `payment/abstract_models.py` | 27 | 5 | 🔴 HIGH |
| `customer/abstract_models.py` | 26 | 6 | 🔴 HIGH |
| `offer/abstract_models.py` | 90 | 5 | 🟡 MEDIUM |
| `checkout/utils.py` | 33 | 3 | 🟡 MEDIUM |
| `payment/utils.py` | 19 | 1 | 🟡 MEDIUM |
| `customer/utils.py` | 8 | 2 | ⚪ LOW |
| `checkout/session.py` | 21 | 0 | ✅ CLEAN |
| `offer/utils.py` | 3 | 0 | ✅ CLEAN |

---

## 🧠 Key Design Decisions

### 01 — AST Chunking > Line Splitting
Python's `ast` module extracts semantically complete functions and classes — not arbitrary line blocks. Each chunk has one clear purpose, making it a meaningful unit for policy evaluation.

### 02 — Hybrid Routing = Cost Control
High-confidence cases (>65% or <35%) bypass deep LLM reasoning. Only ambiguous cases get a full Kimi K2 verdict. On clear-cut cases the classifier's verdict is used, and the LLM only writes an explanation (or is skipped entirely with `AUDITIX_EXPLAIN_FAST_PATH=0`). The accuracy cost on clear-cut cases has not been measured; the classifier's violation recall on its own test set was 0.65.

### 03 — FAISS Before CodeBERT
Running CodeBERT on all 12 policies = 12 inference passes per chunk. FAISS narrows to top-3 first → reduces classifier calls by **75%**.

### 04 — Rate-Limit Retry Logic
`call_groq_with_retry()` parses the exact wait time from Groq's error message and sleeps precisely that long — up to 5 attempts. Zero wasted time from fixed backoff.

### 05 — Graceful Degradation
If CodeBERT fails to load, the pipeline degrades to LLM-only mode automatically — same output schema, just without classifier confidence scores. The system always produces a result.

### 06 — LoRA for Efficiency
LoRA fine-tunes only the query+key attention matrices (not all weights). It trains 1.18M parameters (0.94% of the 125.8M model), which keeps memory and compute low enough for a single T4 GPU.

---

## 📚 Documentation

| Document | What it covers |
|----------|----------------|
| [**explanation.md**](explanation.md) | Whole-project explanation: architecture, module layout, the 8 pipeline stages with data shapes, the hybrid routing, the fine-tuning pipeline, configuration, results, and a detailed comparison with real HDFS |
| [**details.md**](details.md) | File-by-file code walkthrough, plus an interview guide: concepts to know, numbers to memorise, likely questions with answers, a demo script, and a knowledge checklist |
| [`policies/policies.txt`](policies/policies.txt) | The compliance policies the auditor enforces |
| [`notebooks/legacy/`](notebooks/legacy) | The original Colab notebooks, kept for reference (not used by the package) |

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.10+** (the project was developed on Python 3.11 and the notebooks ran on Colab's Python 3.12)
- **Git** on your `PATH` (needed for `auditix audit-repo`)
- **API keys**:
  - [Groq](https://console.groq.com/) — LLaMA 3.1 8B (summaries) and Kimi K2 (audit reasoning)
  - [Google AI Studio](https://aistudio.google.com/) — Gemini embeddings (policy retrieval)
  - [Hugging Face](https://huggingface.co/settings/tokens) — *optional* read token, only for training
- **GPU (optional)** — only needed for fine-tuning. Auditing runs on a CPU; a T4 GPU is recommended for training

### 1. Clone the repository

```bash
git clone https://github.com/rtk5/auditix-policy-aware-code-auditor.git
cd auditix-policy-aware-code-auditor
```

The `code-sample/` folder is a git submodule placeholder for the sample repository used in the live run. You do not need it to use the auditor. To fetch it (if you have access):

```bash
git submodule update --init
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt        # the audit pipeline
pip install -e .                       # installs the `auditix` command (recommended)
```

`requirements.txt` covers everything needed to audit code: `requests`, `numpy`
(< 2), `faiss-cpu`, `google-genai`, `reportlab` and `pypdf`.

If you do not run `pip install -e .`, use `python -m auditix` instead of
`auditix`, from the project root.

Optional extras:

```bash
pip install -r requirements-dev.txt    # pytest, to run the test suite
pip install -r requirements-train.txt  # torch, transformers, peft, ... to fine-tune CodeBERT
```

### 4. Set your API keys

Copy the template and fill it in:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY and GEMINI_API_KEY
```

`auditix` reads `.env` from the project root automatically. Values already set
in your shell take precedence over the file.

You can also export the variables directly:

```bash
export GROQ_API_KEY="your_groq_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

On Windows PowerShell: `$env:GROQ_API_KEY = "your_groq_api_key"`.

> ⚠️ Never commit `.env` or paste keys into code. `.gitignore` already excludes `.env`.

### 5. Verify the installation

```bash
auditix --help
python -m pytest -q        # needs requirements-dev.txt; the tests need no keys and no network
```

### 6. Build the FAISS policy index

The index is built automatically on the first audit. To build it ahead of time:

```bash
auditix build-index
# Indexed 12 policies into models/policy_index
```

This embeds the 12 policies with Gemini and writes `models/policy_index/`. It is rebuilt
automatically whenever `policies/policies.txt` changes.

### 7. (Optional) Get the fine-tuned classifier

The classifier is optional. Without it the auditor runs in **LLM-only** mode.
You have two options:

- **Train your own** (needs a GPU and `requirements-train.txt`): see [Training the Classifier](#-training-the-classifier).
- **Use a checkpoint you already have** (for example, the folder saved by the Colab notebook): copy it to `models/policy_classifier/`, or point to it with `AUDITIX_CLASSIFIER_PATH` / `--classifier`.

The classifier folder holds a LoRA adapter, so loading it also downloads the base
model `microsoft/codebert-base` from Hugging Face the first time.

---

## ⚙️ Configuration

Settings are in `src/auditix/config.py` and can be overridden by environment
variables or command-line flags. Command-line flags win over environment
variables, which win over defaults.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Groq API key (`GROK_API_KEY` is accepted as an alias) |
| `GEMINI_API_KEY` | — | Gemini embedding key |
| `HF_TOKEN` | — | Hugging Face token, used by training |
| `AUDITIX_CLASSIFIER_PATH` | `models/policy_classifier` | Classifier folder (`--classifier`) |
| `AUDITIX_INDEX_DIR` | `models/policy_index` | FAISS index folder |
| `AUDITIX_POLICIES_PATH` | `policies/policies.txt` | Policy file (`--policies`) |
| `AUDITIX_OUTPUT_DIR` | `audit_output` | Report folder (`-o`) |
| `AUDITIX_EXPLAIN_FAST_PATH` | `1` | `0` skips the LLM on confident chunks (`--no-explain-fast-path`) |
| `AUDITIX_USE_CLASSIFIER` | `1` | `0` forces LLM-only mode (`--no-classifier`) |

Model names, thresholds (`0.35` / `0.65`) and the number of retrieved policies
(`3`) are constants in `config.py`.

---

## 🖥️ Usage

### Audit a local folder

```bash
auditix audit-local path/to/your/project -o audit_output
```

### Audit a GitHub repository

```bash
auditix audit-repo https://github.com/org/repo -o audit_output
```

The repository is shallow-cloned into a temporary folder, audited, and then deleted.

### Useful flags

```bash
auditix audit-local ./project --no-classifier          # LLM-only mode
auditix audit-local ./project --no-explain-fast-path   # fewer LLM calls (template explanations)
auditix audit-local ./project --policies my_policies.pdf
auditix audit-local ./project --pause 2                # slower, gentler on rate limits
auditix -v audit-local ./project                       # debug logging
```

A run prints a summary like this:

```
============================================================
FULL AUDIT COMPLETE
============================================================
  Mode             : hybrid
  Files audited    : 9
  Chunks analyzed  : 337
  Total violations : 24
  Critical issues  : 0
  Compliance rate  : 93%
  Output folder    : audit_output
============================================================
```

It writes two files to the output folder:

- `audit_results.json` — the full verdict for every chunk
- `audit_report.pdf` — a readable report grouped by file

### Use it from Python

```python
from auditix.pipeline import build_context, run_audit_local

ctx = build_context()                         # reads .env and the environment
results = run_audit_local("path/to/project", output_dir="audit_output", ctx=ctx)
print(results["summary"])
```

To run without any API calls, inject fakes (this is how the tests work):

```python
from auditix.pipeline import build_context, audit_directory

ctx = build_context(embedder=my_embedder, classifier=None)  # LLM-only, custom embedder
ctx.llm_audit = my_llm_function                             # any callable with the same signature
results = audit_directory("path/to/project", ctx)
```

### Output format

Each code chunk produces a verdict in this schema:

```json
{
  "name": "process_payment",
  "type": "FunctionDef",
  "compliant": false,
  "violations": [
    "Missing AuditLogger on transaction",
    "Direct DB update to order status"
  ],
  "explanation": "Classifier p(violation)=91.0%. This function updates order status directly via ORM without going through PaymentService and without any AuditLogger call, violating both the payment logging and architecture policies.",
  "severity": "critical",
  "classifier_confidence": 0.91,
  "policies_checked": ["..."],
  "summary": "...",
  "decision_path": "[Hybrid/high-confidence→LLM-explain]"
}
```

`decision_path` is one of:

| Value | Meaning |
|-------|---------|
| `[Hybrid/high-confidence→LLM-explain]` | Classifier was confident; the verdict is the classifier's. This label is used for both fast-path verdicts (violation and compliant), so read the `compliant` field to tell them apart |
| `[Hybrid/ambiguous→full-LLM]` | Probability in the 35–65% band; Kimi K2 decided |
| `[LLM-only]` | No classifier loaded; Kimi K2 decided |

### Run the original notebook (legacy)

The notebooks were the first version of the pipeline. They are kept in
`notebooks/legacy/` for reference. New work should use the package.

---

## 🎓 Training the Classifier

```bash
pip install -r requirements-train.txt
auditix train --epochs 5
```

Useful options:

```bash
auditix train --output-dir models/policy_classifier   # where the adapter is saved
auditix train --cache models/ft_dataset.json          # dataset cache file
auditix train --max-per-source 1000                   # smaller run, for a quick check
auditix train --rebuild-dataset                       # ignore the cache and reload the datasets
```

What the command does:

1. Loads the three public datasets (see [Training Data & Dataset Strategy](#-training-data--dataset-strategy)) and writes them to a cache.
2. Removes duplicate code across all sources, balances the classes 50/50, and splits 80/10/10.
3. Fine-tunes `microsoft/codebert-base` with a LoRA adapter (r=16, alpha=32, query and key projections).
4. Evaluates on the held-out test split and writes `metrics.json`.
5. Saves the adapter, tokenizer, `training_config.json`, `metrics.json` and `training_curves.png` to `models/policy_classifier/`.

The output folder is git-ignored, so the trained weights are not committed. Share them through
Hugging Face Hub or a release asset.

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

There are 52 tests. They cover policy parsing, AST chunking, retry logic,
hybrid routing (every threshold boundary), the FAISS index with a fake embedder,
report generation, the training-data logic, and the CLI. None of them needs an
API key, a GPU or network access. A full run takes about one second.

---

## 📁 Project Structure

```
auditix-policy-aware-code-auditor/
│
├── README.md                    # this file
├── explanation.md               # architecture and pipeline (read this first)
├── details.md                   # code walkthrough and interview guide
├── requirements.txt             # runtime dependencies
├── requirements-train.txt       # fine-tuning dependencies (torch, transformers, peft, ...)
├── requirements-dev.txt         # pytest
├── pyproject.toml               # package metadata and the `auditix` command
├── .env.example                 # API key template (copy to .env)
├── .gitignore
│
├── policies/
│   └── policies.txt             # 12 compliance policies, one per line
│
├── src/auditix/                 # the Python package
│   ├── cli.py                   # auditix audit-local | audit-repo | build-index | train
│   ├── config.py                # constants, settings, .env loader
│   ├── pipeline.py              # end-to-end orchestration
│   ├── chunker.py               # AST chunking of Python files
│   ├── policies.py              # policy loading (.txt / .md / .pdf)
│   ├── embeddings.py            # Gemini embeddings (document / query)
│   ├── policy_index.py          # FAISS index build / cache / search
│   ├── classifier.py            # CodeBERT + LoRA inference
│   ├── hybrid_engine.py         # routing rules and verdict merging
│   ├── reporting.py             # JSON and PDF output
│   ├── llm/
│   │   ├── retry.py             # Groq calls with rate-limit retries
│   │   └── groq_client.py       # summary and audit prompts, JSON parsing
│   └── training/
│       ├── config.py            # training hyperparameters
│       ├── datasets.py          # dataset loading, dedupe, balance, split
│       └── train.py             # LoRA fine-tuning and test evaluation
│
├── tests/                       # 52 offline unit tests
│
├── notebooks/
│   └── legacy/                  # original Colab notebooks (reference only)
│
├── models/                      # generated artifacts (git-ignored, except .gitkeep)
│   └── .gitkeep
│
├── audit_report.pdf, paper.pdf, compliance_auditor_ppt.pdf   # project documents
├── code links.txt               # reference links
└── code-sample/                 # submodule placeholder for the sample audit target
```

---

## 🩺 Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `error: GEMINI_API_KEY is not set` | Add the key to `.env` or export it. Check you are in the project root, since `.env` is read from there |
| Groq returns `invalid_api_key` | The key is wrong or expired. Create a new one in the Groq console |
| `Classifier folder not found … running LLM-only` | Expected if you have not trained or copied a classifier. Not an error |
| `Could not load classifier (…) — running LLM-only` | Usually a `transformers` version mismatch, or no internet access to download `microsoft/codebert-base`. Install `requirements-train.txt`, which pins the versions the adapter was trained with |
| `ModuleNotFoundError: No module named 'auditix'` | Run `pip install -e .`, or run from the project root with `PYTHONPATH=src` |
| `faiss` fails to install on your platform | Use Python 3.10–3.12 with a recent `pip`. `faiss-cpu` ships wheels for Linux, macOS and Windows on those versions |
| `audit-repo` fails with a git error | Check `git --version`. Private repositories need credentials configured for git |
| Many `Rate limit — waiting …` messages | Normal. Retries wait the time Groq asks for. Use `--pause 2` to slow down |
| The PDF shows boxes instead of ✔ ✘ ⚠ | ReportLab's default font has no such glyphs. The text is still correct. Register a Unicode TTF font to fix it |
| PowerShell ignores `export` | Use `$env:GROQ_API_KEY = "..."` or put the keys in `.env` |

---

## ⚠️ Known Limitations

- **Weak labels.** The classifier's training labels come from keyword rules and from security and defect datasets, not from human review. Its 0.81 macro-F1 measures how well it learned those labels.
- **Low violation recall.** On its own test set the classifier caught 65% of violations (precision 0.95). The ambiguous band and the LLM help, but real misses remain.
- **Unmeasured savings.** The "fewer LLM calls" benefit depends on the mode and on the share of ambiguous chunks, which has not been measured.
- **Single machine.** The auditor is a batch tool, not a distributed system. See [explanation.md §10](explanation.md#10-comparison-with-real-hdfs) for what scaling would require.
- **LLM output is advisory.** Verdicts are judgments with evidence, not legal findings. Review violations before acting on them.
- **Long functions are truncated.** The classifier sees at most 512 tokens.

The full list, with the corrections made to the original notebooks and README, is in [explanation.md §11](explanation.md#11-limitations-and-honest-notes).

---

## 🧠 Contributing

1. Create a branch and make your change.
2. Run `python -m pytest -q`. Add a test for any new routing rule or parser.
3. Keep keys, outputs and model weights out of Git. `.gitignore` covers the usual paths.
4. Open a pull request with a short description of what changed and why.

Please keep `explanation.md` and `details.md` in step with the code. If you
change the routing thresholds, the prompts, or the training settings, update the
tables there too.

---

## 📜 License

MIT License. The `LICENSE` file is not yet present in this repository, so add one (for example from [choosealicense.com](https://choosealicense.com/licenses/mit/)) before publishing.

---

*Built as part of a university AI systems project. Live-tested on a real Django e-commerce codebase with 337 code chunks and 93% compliance rate.*
