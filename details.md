# Code Details and Interview Guide

This document has two parts:

1. **A walkthrough of every file** in the project: what it does, its main functions, and why it is written the way it is.
2. **An interview guide**: what you need to know, the numbers to memorise, likely questions with answers, and a demo script.

Read [`explanation.md`](explanation.md) first for the big picture. This file goes
into the code.

---

## Part 1 — Code walkthrough

### 1.1 Repository root

| File | Purpose |
|---|---|
| `README.md` | Overview, installation, usage, configuration, testing, troubleshooting |
| `explanation.md` | Architecture, pipeline, HDFS comparison |
| `details.md` | This file |
| `requirements.txt` | Runtime dependencies (pinned to ranges). Used by `pip install -r` |
| `requirements-train.txt` | Fine-tuning dependencies: torch, transformers 4.41.2, peft 0.11.1, accelerate 0.30.1, datasets 2.19.1, scikit-learn, matplotlib. The versions match the notebook |
| `requirements-dev.txt` | `pytest` |
| `pyproject.toml` | Package metadata. Installs the `auditix` command. Reads dependencies from `requirements.txt` so the list is not duplicated. Sets `pythonpath` for pytest |
| `.env.example` | Template for API keys and overrides. Copy to `.env` |
| `.gitignore` | Ignores `.env` (keys), generated outputs, model weights, the dataset cache. Keeps `models/.gitkeep` |
| `policies/policies.txt` | The 12 policies, one per line, with domain comments |
| `notebooks/legacy/` | The original notebooks. Not used by the package |
| `models/` | Empty except `.gitkeep`. Classifier, FAISS index and dataset cache are written here at run time |
| `tests/` | 50 unit tests, all offline |

### 1.2 `src/auditix/__init__.py` and `__main__.py`

- `__init__.py` defines `__version__`. It is the only thing that runs on import.
- `__main__.py` lets you run `python -m auditix ...`. It calls `cli.main()`.

### 1.3 `config.py` — configuration in one place

**Key ideas**

- `PROJECT_ROOT = Path(__file__).resolve().parents[2]`. The file is `src/auditix/config.py`, so `parents[0]` is `auditix`, `[1]` is `src`, and `[2]` is the repo root. Paths are built from this, so the project works from any current directory.
- Constants: API URLs, model names (`llama-3.1-8b-instant`, `moonshotai/kimi-k2-instruct`, `models/gemini-embedding-2-preview`), thresholds (`AMBIGUITY_LOW = 0.35`, `AMBIGUITY_HIGH = 0.65`, `CRITICAL_THRESHOLD = 0.92`), `TOP_K_POLICIES = 3`, `CLF_MAX_LENGTH = 512`.
- `SKIP_DIRS` and `MIN_FILE_CHARS` control file discovery.
- `get_groq_api_key()` accepts `GROQ_API_KEY` and falls back to `GROK_API_KEY`, the name used in the notebooks' Colab secrets.
- `Settings` is a dataclass. `Settings.from_env()` reads `AUDITIX_*` variables. The explicit `cls()` call inside it gives defaults first, and then the environment overrides them.

**Why:** a single file means that changing a model or a threshold does not require searching the code.

### 1.4 `policies.py` — loading the policy list

- `parse_policy_text(text)` turns free text into a list of policies. It skips comments and short lines, removes numbering and bullets with regular expressions, and removes case-insensitive duplicates while keeping the first occurrence.
- `load_policies(path)` reads a `.txt`, `.md` or `.pdf` file. The PDF branch imports `pypdf` only when it is used.

**Interview point:** the notebook's dynamic parser is reused, so any policy document can be used, not just the hard-coded list.

### 1.5 `chunker.py` — AST chunking

- `CodeChunk` is a frozen dataclass with `name`, `kind`, `code` and `path`.
- `parse_code_chunks(source, filename, top_level_only=False)`:
  1. `ast.parse(source)`. On `SyntaxError`, return the whole file as one chunk.
  2. Walk the tree: `ast.walk(tree)` (all depths), or `tree.body` (top level only).
  3. For each `FunctionDef`, `AsyncFunctionDef` or `ClassDef`, take `ast.get_source_segment(source, node)`. This returns the exact text, including decorators and comments inside the body.
  4. `textwrap.dedent` removes the indentation of nested definitions, so each chunk is valid on its own.
  5. If nothing matched, return the whole file.

**Why AST and not lines:** a line window can cut a function in half. An AST chunk is a complete, meaningful unit. `ast.walk` includes methods, so a method is audited on its own and inside its class. This matches the notebook and is the behaviour behind the 337-chunk result. The cost is that the class body is counted twice. `top_level_only=True` avoids that.

### 1.6 `llm/retry.py` — the Groq HTTP call

- `parse_wait_seconds(message)` uses the regex `try again in\s+([\d.]+)\s*(ms|s)`. It returns the number of seconds plus one second of padding. Milliseconds are converted. With no match it returns 11 seconds (10 + 1).
- `call_groq_with_retry(payload, max_retries=5, api_key=None, sleep=time.sleep)`:
  - Sends a POST with `timeout=60` (the notebook had no timeout).
  - If the reply has `choices`, return it.
  - If the error code is `rate_limit_exceeded`, sleep and retry.
  - Any other error is returned at once, so the caller sees it.
  - `sleep` is a parameter, so tests can replace it and run instantly.

**Interview point:** "I retry on the server's own wait time instead of a fixed exponential backoff. The API tells us when the rate limit resets, so we wait exactly that long." Mention the trade-off: it depends on the message format, so the regex fallback matters.

### 1.7 `llm/groq_client.py` — the two LLM roles

- `summarize_chunk(code)` sends the code to LLaMA 3.1 8B with a system prompt asking for 2–3 sentences. Returns the text, or an "unavailable" message on error.
- `build_audit_prompt(...)` assembles the Kimi K2 prompt: the code, the summary, the policies, and an optional classifier hint. The prompt demands JSON in a fixed shape.
- `parse_llm_json(raw)`:
  1. Strips ```` ```json ```` fences.
  2. Tries `json.loads`.
  3. If that fails, extracts the first `{...}` block with a regex and tries again.
  4. Otherwise returns `{"compliant": False, "violations": ["JSON parse error"], ...}`. Failing closed means a human sees it.
- `audit_chunk(...)` calls Kimi K2 with `max_tokens=500`, then parses.

**Interview point:** LLM output is untrusted input. Always parse defensively, always have a fallback, and never let a parse error silently count as compliant.

### 1.8 `embeddings.py` — Gemini embeddings

- `GeminiEmbedder` creates the `google.genai` client on first use, so importing the module does not need the SDK or a key.
- `embed_documents(texts)` uses `RETRIEVAL_DOCUMENT` and returns a float32 array of shape `(n, dim)`. Used for policies.
- `embed_query(text)` uses `RETRIEVAL_QUERY` and returns shape `(1, dim)`. Used for summaries.
- The model is `models/gemini-embedding-2-preview`, which gives 3072-dimensional vectors (printed in the notebook).

**Interview point:** asymmetric embedding. Documents and queries are different kinds of text, and the task type tells the model how to match them. Using the same mode for both would weaken the match.

### 1.9 `policy_index.py` — FAISS

- `PolicyIndex` wraps a FAISS `IndexFlatL2` and the list of policy strings. The row order matches the policy order.
- `build(policies, embedder)` embeds the policies and adds them to the index.
- `save(dir)` writes `index.faiss` and `policies.json`.
- `load_or_build(policies, dir, embedder)`: if the saved `policies.json` equals the current list, load the index. Otherwise rebuild and save. This is cache invalidation by content.
- `search(query, top_k)` embeds the query, runs `index.search`, and maps the returned row numbers to policy text. `top_k` is capped at the index size.
- `Embedder` is a `typing.Protocol`. Any object with `embed_documents` and `embed_query` will do, which is how the tests inject a fake.

**Interview point:** `IndexFlatL2` is exact, brute-force search. With 12 vectors it is the right choice. For millions of vectors you would use an approximate index (IVF, HNSW), and accept some recall loss for speed.

### 1.10 `classifier.py` — CodeBERT inference

- `PolicyClassifier.from_pretrained(path)` imports `torch` and `transformers` inside the method. It loads the tokenizer and the model from the folder, moves the model to CUDA if present, and sets eval mode.
- `violation_probability(code, policy)`:
  1. Tokenises the pair with `max_length=512`, `padding="max_length"`, `truncation=True`.
  2. Runs the model under `torch.no_grad()`.
  3. Applies softmax to the logits and returns index 1, the VIOLATION probability.
- `load_classifier(path, enabled)` returns `None` if the folder is missing, the flag is off, or loading raises any exception. It logs a warning. This is the graceful degradation.

**Interview point:** the `torch.no_grad()` context stops autograd from storing activations, which saves memory and time at inference.

### 1.11 `hybrid_engine.py` — the decision logic

This is the file to understand best.

- `route_probabilities(probs)` returns a `Routing(route, max_probability, is_violation)`:
  - If any probability is in `[0.35, 0.65]`, the route is **ambiguous**.
  - Otherwise if any is `> 0.65`, the route is **violation** (fast path).
  - Otherwise the route is **compliant** (fast path).
- `severity_from_probability(p)` maps probability to a label: ≥0.92 critical, ≥0.80 high, ≥0.65 medium, ≥0.35 low, else none.
- `audit_chunk_hybrid(...)`:
  1. **No classifier:** call the LLM with an empty hint. `decision_path = "[LLM-only]"`.
  2. **Classifier present:** compute one probability per policy, then route.
     - **Ambiguous:** call the LLM with a hint, and use its verdict.
     - **Fast path:** the verdict is the classifier's. If `explain_fast_path` is on, call the LLM for an explanation with a hint, and use its explanation and violation list. If it is off, use a template and make no LLM call.
  3. Build the result dict with `compliant`, `violations`, `explanation`, `severity`, `classifier_confidence` and `decision_path`.
- `normalize_severity` accepts only the five valid labels. Anything else becomes `unknown`, or `none` for a compliant chunk.
- Dependencies are injected as parameters: `classifier` (anything with `violation_probability`) and `llm_audit` (a function). The module has no network imports.

**Interview point:** "The routing is a pure function of the probabilities, so I can test every boundary without a model. The LLM is passed in, so the same code runs in production with the real API and in tests with a fake."

### 1.12 `pipeline.py` — orchestration

- `AuditContext` is a dataclass holding `settings`, `policies`, `index`, `classifier`, plus the `summarize` and `llm_audit` functions. The last two default to the real Groq functions. Replacing them is how the smoke test ran offline.
- `build_context(settings, embedder, classifier, pause_seconds)` does the setup. `classifier="auto"` loads from disk. `None` forces LLM-only mode.
- `audit_source(code, filename, ctx)`: for each chunk, summarise → retrieve → route → build the report entry. It also counts totals and sleeps `pause_seconds` between chunks, to respect rate limits.
- `collect_python_files(root)` uses `rglob` and skips folders in `SKIP_DIRS`.
- `audit_directory(root, ctx)` reads each file and skips short ones. Paths are stored relative to the root with `as_posix()`, so reports look the same on Windows.
- `_aggregate(reports, ctx)` computes the totals and the compliance rate. `rate = round((total - violations) / total * 100)`.
- `write_outputs(results, dir)` writes JSON and PDF.
- `clone_repo(url, dest)` runs `git clone --depth 1` through `subprocess.run(check=True)`.
- `run_audit_repo` clones into a `tempfile.TemporaryDirectory`, so the clone is deleted afterwards.

**Interview point:** "The pipeline is a sequence of pure-ish steps. Each step can be swapped, which is why I could test the whole run offline."

### 1.13 `reporting.py` — output

- `write_json` writes the result with `indent=4` and `ensure_ascii=False`, so non-ASCII characters are kept.
- `write_pdf` builds a ReportLab `SimpleDocTemplate` with a `Paragraph` per line. It uses `html.escape` for every piece of user-controlled text. This matters because ReportLab's `Paragraph` parses a small HTML-like markup, so a function named `<script>` would otherwise break the PDF or inject formatting.

### 1.14 `cli.py` — command-line interface

- `build_parser()` defines four sub-commands: `audit-local`, `audit-repo`, `build-index` and `train`.
- `_settings_from_args` starts from `Settings.from_env()` and applies command-line flags with `dataclasses.replace`. Command-line flags win over environment variables, which win over defaults.
- `main()` configures logging, dispatches, and turns `FileNotFoundError`, `RuntimeError` and `ValueError` into a one-line error with exit code 2. For example, a missing `GEMINI_API_KEY` prints a readable message rather than a traceback.
- `train` imports the training module inside a `try`. If torch is missing, the user gets an install hint, not a crash.

### 1.15 `training/config.py` — training hyperparameters

`TrainConfig` is a dataclass with the notebook's `CFG` values. `to_dict()` converts `Path` objects to strings so the config can be written as JSON.

### 1.16 `training/datasets.py` — building the training set

Most of this file is pure Python and is tested offline.

- `KEYWORD_RULES` is the ordered list of keyword → policy rules.
- `stable_index(text, n)` gives a deterministic bucket from SHA-1.
- `assign_policy(code, policies)` returns the first matching rule, or a stable hash bucket.
- `make_sample(code, policy, label, source)` builds the record.
- `load_securecode_v2`, `load_codexglue_defect`, `load_clean_python` are the three network loaders. They import `datasets` and `huggingface_hub` inside the function.
- `merge_dedupe_balance(samples, seed)`:
  1. Drops rows with empty code or policy.
  2. Removes duplicate code across all sources, using a `set`.
  3. Undersamples to 50/50 with a seeded `random.Random`.
  4. Shuffles.
- `split_dataset(balanced, train, val)` slices the list 80/10/10.
- `build_dataset(policies, cfg, token, rebuild)` loads, merges, splits and writes `ft_dataset.json`. On later runs it reads the cache.

**Interview point:** the duplicate-leak fix. Show the test `test_merge_removes_duplicates_across_sources_and_balances`. Explain why a dataset split must be made *after* deduplication, not before.

### 1.17 `training/train.py` — fine-tuning

- `PolicyPairDataset` is a `torch.utils.data.Dataset`. `__getitem__` tokenises one pair and returns `input_ids`, `attention_mask` and `labels`.
- `build_lora_model(cfg)` loads `AutoModelForSequenceClassification` with `num_labels=2` and the label names, then wraps it with `get_peft_model` using a `LoraConfig` (`TaskType.SEQ_CLS`, r=16, alpha=32, targets `query` and `key`).
- `compute_metrics` is called by the `Trainer` on each evaluation. It returns accuracy and macro-F1 from scikit-learn.
- `train_model(cfg, splits)` sets `TrainingArguments` (cosine schedule, warm-up 10%, fp16 on CUDA, `load_best_model_at_end`, `metric_for_best_model="f1"`), then runs `trainer.train()`.
- `evaluate_test(...)` runs the model on the test split with `torch.no_grad()` and returns the confusion matrix and a classification report.
- `save_artifacts(...)` writes the adapter, tokenizer, configs, metrics and the training-curve plot. Plotting is optional and is skipped if matplotlib is missing.
- `run(cfg, policies_path, ...)` is the top-level flow. `main(args)` is the CLI entry point.

**Interview point:** `save_pretrained` on a PEFT model saves only the adapter (a few MB), not the full model. Loading needs the base model name, which is in `adapter_config.json`. This is why the classifier folder is small.

### 1.18 Tests (`tests/`)

| File | What it checks |
|---|---|
| `test_policies.py` | Parsing rules, duplicates, the shipped file has 12 policies, error cases |
| `test_chunker.py` | Functions, classes and methods are found; `top_level_only`; dedent; syntax-error fallback |
| `test_retry.py` | Wait-time parsing (s and ms); success; rate-limit retry with a fake `sleep`; no retry on other errors; missing key |
| `test_hybrid_engine.py` | Every routing boundary (0.35 and 0.65 inclusive); fast-path verdict is the classifier's; ambiguous uses the LLM; `explain_fast_path=False` makes no LLM call; LLM-only mode; severity normalisation |
| `test_policy_index.py` | Nearest-neighbour order with one-hot vectors; `top_k` cap; cache reuse; rebuild on change |
| `test_reporting.py` | JSON round trip; PDF is a valid PDF |
| `test_training_data.py` | Keyword assignment; determinism; cross-source dedupe; balance; split sizes; the notebook's hyperparameters |
| `test_cli.py` | Flags override environment; env var controls fast path; subcommands exist |

Run them with `python -m pytest -q`. They take under a second and need no keys.

---

## Part 2 — Interview guide

### 2.1 What you must be able to explain in two minutes

> "The auditor checks a Python codebase against company policies. It splits each file into functions and classes using the AST. An LLM summarises each chunk, and that summary is embedded with Gemini and searched in a FAISS index of 12 policies to find the top 3. A fine-tuned CodeBERT classifier, with a LoRA adapter, scores each chunk against those 3 policies. If the scores are clearly high or low, the classifier's verdict is used and Kimi K2 writes the explanation. If a score is between 35% and 65%, Kimi K2 decides. The result is a JSON verdict per chunk and a PDF report. On a held-out test set the classifier reached 81.7% accuracy and 0.81 macro-F1."

Practise it out loud until it takes under two minutes.

### 2.2 Numbers to memorise

| Fact | Value |
|---|---|
| Policies | 12, across 5 domains |
| Retrieval | FAISS `IndexFlatL2`, top-3 |
| Embedding dimension | 3072 (Gemini Embedding 2 Preview) |
| Classifier base | `microsoft/codebert-base` (RoBERTa, ~125M params) |
| LoRA | r=16, alpha=32, dropout 0.1, query+key, 1.18M trainable params (0.94%) |
| Training | 5 epochs, lr 2e-4, batch 16 × 2 accumulation = 32, cosine, 10% warm-up, fp16 |
| Dataset | 7,239 raw → 6,916 balanced (50/50) → 5,532 / 691 / 693 |
| Test results | accuracy 0.817, macro-F1 0.81, TN 355, FP 11, FN 116, TP 211 |
| Violation precision / recall | 0.95 / 0.65 |
| Thresholds | ambiguous band 0.35–0.65 |
| Live audit | 9 files, 337 chunks, 24 violations, 93% compliance |
| Training time | 959 s on a Tesla T4 |
| Retries | up to 5 attempts, wait = server's stated time + 1 s |

### 2.3 Concepts you should be able to define

**Machine learning and NLP**

- **Transformer encoder** and **self-attention**: each token attends to every other token.
- **Tokenisation** and the `[CLS]` / `[SEP]` tokens. Sentence-pair input and segment handling.
- **Sequence classification**: take the `[CLS]` representation, pass it through a head, get logits.
- **Softmax** turns logits into probabilities. Be able to say why we use index 1.
- **Fine-tuning vs. training from scratch.** Starting from pre-trained weights.
- **LoRA**: freeze the base weights and learn `W + (α/r)·B·A` for low-rank `A` and `B`. Parameters go from 125M to 1.2M. Explain why `alpha/r` scaling.
- **Mixed precision (fp16)**: faster and less memory on a GPU, with loss scaling.
- **Learning-rate schedule**: warm-up, then cosine decay. Why warm-up matters for transformers.
- **Gradient accumulation**: simulate batch 32 with batches of 16.
- **Overfitting and early stopping**: `load_best_model_at_end` keeps the best epoch by validation F1.
- **Metrics**: accuracy, precision, recall, F1 (macro vs. micro), confusion matrix. Know when accuracy misleads (imbalance). Here the data is balanced, but the real-world violation rate is not.
- **Weak supervision and label noise**: labels from keyword rules, not from humans.
- **Data leakage**: duplicates across train and test inflate scores. Explain the fix.
- **Class balancing by undersampling**: why it throws data away, and when to oversample or weight the loss instead.

**Retrieval and embeddings (RAG)**

- **Embedding**: a fixed-length vector for text. Similar meaning, nearby vectors.
- **Asymmetric retrieval**: `RETRIEVAL_DOCUMENT` for policies, `RETRIEVAL_QUERY` for summaries.
- **L2 distance** vs. cosine similarity. Note that `IndexFlatL2` uses L2 on the raw vectors.
- **Exact vs. approximate nearest neighbours**: flat search versus HNSW or IVF.
- **Why retrieve at all?** To cut the number of classifier calls from 12 to 3 and keep the prompt focused.
- **Why summarise before embedding?** Raw code and policy prose live in different parts of the space. A business-language summary is closer to the policy text.

**LLM engineering**

- **Structured output**: asking for JSON, and handling replies that are not valid JSON.
- **Rate limits (HTTP 429)** and **retry with server-provided delay**.
- **Prompt design**: role, task, format constraints, and a classifier hint as extra context.
- **Temperature and determinism**: the code does not set temperature, so replies can vary. Say how you would fix that (set temperature 0, cache verdicts).
- **Cost and latency**: the number of LLM calls per chunk is 2 (summary and audit).

**Systems and software engineering**

- **Hybrid cascade**: cheap model first, expensive model for hard cases. Same idea as spam filtering or a cascade of classifiers.
- **Graceful degradation**: LLM-only fallback.
- **Dependency injection**: passing the classifier and LLM as parameters, which makes the logic testable.
- **Lazy imports**: keeping heavy dependencies out of the import path.
- **Configuration via environment variables** and the 12-factor approach.
- **Packaging**: `pyproject.toml`, console scripts, editable installs, `src/` layout.
- **Testing without the network**: fakes for the embedder, classifier and LLM.
- **Content-based cache invalidation**: rebuild the index when `policies.json` changes.
- **Security**: keys in environment variables, `.env` ignored by Git, HTML-escaping in the PDF. Mention that the notebooks used Colab secrets.

**Distributed systems (for the HDFS question)**

- The NameNode/DataNode split, replication factor 3, heartbeats, block reports, write-once-read-many, data locality. See `explanation.md` §10.

### 2.4 Likely questions and good answers

**Q1. Why hybrid? Why not just use the LLM for everything?**
The LLM is the most accurate judge but the most expensive and slowest. Most chunks are clear-cut, and a small classifier can decide them with good precision (0.95 on violations). The cascade sends only uncertain cases to the LLM. The tradeoff is recall: the classifier misses about a third of violations, so the fast path has a real error rate.

**Q2. Why AST chunking rather than fixed-size windows?**
A window can split a function, so the verdict would be about half a function. AST nodes are complete units with a name, which also makes the report readable ("process_payment violates…"). The cost is that nested methods are counted twice, which `top_level_only` addresses.

**Q3. Why FAISS if there are only 12 policies?**
Honestly, brute force would be fine at 12. I used FAISS because the design has to scale to a larger policy library, and the index is cached on disk so it is not rebuilt on each run. For a few hundred thousand policies I would move to an approximate index.

**Q4. Why embed the summary and not the code?**
The policies are written in business language. A summary is also business language, so it is closer to them in embedding space. Raw code is full of identifiers that do not match the policy wording.

**Q5. Why `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`?**
Gemini's embedding models support task types. Documents and queries are encoded to be matched against each other. Using the same mode for both is a common mistake that weakens retrieval.

**Q6. Why LoRA? Why only query and key?**
Full fine-tuning updates 125M parameters, which needs much more memory and is slower. LoRA trains 1.2M. Query and key determine which tokens attend to which, which is the part most useful for matching code to policy. Value and output projections could be added with a higher rank if recall is too low. I did not run that comparison, so I cannot say it helps.

**Q7. How did you choose the thresholds 0.35 and 0.65?**
They were set in the notebook and are symmetric around 0.5, so the ambiguous band is 30 points wide. I would tune them on a human-labelled validation set, since the current labels are weak. The honest answer is that they were chosen by judgment, not optimised.

**Q8. Your validation says 0.81 F1. Is the classifier good?**
It is good on its own labels. The labels are weak: keyword policy assignment and labels from security and defect datasets. So 0.81 measures how well it learned those labels, not whether it finds real violations in a real codebase. The next step is a human-labelled set from the sample repository.

**Q9. Why is violation recall only 0.65? How would you improve it?**
The classifier is conservative: precision is high, so it rarely raises false alarms. Options: lower the decision threshold (measured on a labelled set), class-weighted loss, more violation examples with real labels, and a longer context window. I would check each against a labelled validation set before changing it.

**Q10. What happens to code longer than 512 tokens?**
The tokeniser truncates it silently, so the end is lost. The classifier might miss a violation in the truncated part. The fix is to split long functions into windows, score each window, and take the maximum. This is listed as a next step.

**Q11. How do you know the LLM's verdict is reliable?**
I don't, fully. That is why the fast path does not let the LLM override the classifier, and why a JSON parse error is recorded as a violation for review. The LLM's explanation is useful, but it is a judgment, not a proof. A real evaluation would compare LLM verdicts with human labels.

**Q12. What if Groq rate-limits you?**
`call_groq_with_retry` reads the wait time from the error message ("try again in 2.5s"), sleeps for that long plus one second, and retries up to five times. Other errors are not retried. The pipeline also waits one second between chunks.

**Q13. What if the classifier fails to load?**
`load_classifier` catches any exception, logs a warning, and returns `None`. The pipeline then runs in LLM-only mode, and the report says so in `audit_mode`.

**Q14. Why did you change the deduplication?**
The notebook removed duplicates inside each loader, but the same snippets appeared in more than one source, and one source was appended twice. Those duplicates could land in both train and test. Deduplicating across all sources before splitting removes that leak. The numbers will differ from the notebook's 0.817, and I would report the new numbers.

**Q15. Why `hashlib` instead of `hash()`?**
Python randomises `hash()` for strings between runs (unless `PYTHONHASHSEED` is set), so the fallback policy assignment would change from run to run. `sha1` is stable.

**Q16. How would you run this on 10,000 repositories?**
See `explanation.md` §10.4: store the raw code in object storage or HDFS, parse with Spark, cache verdicts by chunk hash, use a shared rate limiter for the APIs, and write immutable run folders.

**Q17. How is this different from HDFS?**
HDFS is a distributed file system for storing large files with replication across many machines, with a NameNode for metadata. This project stores nothing large and runs on one machine. Its bottleneck is LLM latency, not storage. The useful HDFS idea is to keep expensive work rare, which the hybrid engine does. Full comparison in `explanation.md` §10.

**Q18. What would you test first if you had more time?**
The LLM prompts against a labelled set, measured for JSON validity and verdict agreement. Then an end-to-end test with a real small classifier checkpoint. Then the chunker on a corpus of real Python files, including Python 2 and syntax-error cases.

**Q19. Why separate the training package from the audit package?**
They have different dependencies (torch and transformers are heavy), different hardware (a GPU for training, a CPU for audits), and different lifecycles. A user who only audits should not need torch. Lazy imports and `requirements-train.txt` make that possible.

**Q20. What is `load_best_model_at_end` doing?**
After each epoch the Trainer evaluates on the validation set and saves a checkpoint. At the end it reloads the checkpoint with the best validation F1, not the last epoch. That protects against overfitting in later epochs.

**Q21. Why is the PDF escaping important?**
ReportLab's `Paragraph` interprets a subset of HTML-like tags. A function name or message containing `<b>` or `<script>` would change the layout or break the document. `html.escape` turns those characters into text.

**Q22. Why `ast.get_source_segment` instead of `ast.unparse`?**
`get_source_segment` returns the original text, including comments and formatting. `unparse` rebuilds code from the tree and loses comments, which would change what the LLM sees.

**Q23. What does the `.squeeze()` in the notebook do, and why did you change it?**
`squeeze()` removes all size-1 dimensions. Indexing with `[0]` removes only the batch dimension of one item, so it is safer if the sequence length ever becomes 1.

**Q24. What are the risks of using an LLM as the judge?**
Non-determinism, prompt sensitivity, hallucinated violations, and cost. Mitigations: structured output, a fixed prompt, stored verdicts, human review of the violations list, and measuring agreement with human labels.

**Q25. What is the compliance rate and is it a fair metric?**
It is `(chunks − violations) / chunks`. It is only as good as the verdicts behind it. The classifier's recall on violations was 0.65 in its test set, so about 35% of real violations were missed on that split. The true compliance rate of a codebase may be lower than reported. It also counts every chunk equally, so a trivial getter counts the same as a payment function. Severity-weighted scoring would be fairer.

### 2.5 Weak points to own, not hide

- The **labels are weak**, and the metrics are measured on those labels.
- **Recall on violations is 0.65.** Be ready to say how you would raise it.
- The **README had errors** (parameter count, savings claim, dataset list) which were corrected. Saying so shows care.
- The **live run numbers** come from a single run of the notebook. There is no error bar.
- The **chunk compliance rate** is unweighted.
- **No human evaluation** yet.
- **The PDF symbols** may render as boxes with the default font.

### 2.6 Demo script (about 5 minutes)

1. **Show the layout** (30 s): `src/auditix/`, the `policies/policies.txt` file, the tests folder.
2. **Show the routing** (90 s): open `hybrid_engine.py`, walk through `route_probabilities`, then run `pytest -q tests/test_hybrid_engine.py -v` to show the boundary tests.
3. **Run an offline audit** (90 s): `python -m pytest -q`, then describe the smoke test. If you have keys, run `auditix audit-local code-sample -o demo_out` and open the PDF.
4. **Show the training data fix** (60 s): `merge_dedupe_balance` and its test.
5. **Show the numbers** (30 s): the confusion matrix from §2.2.
6. **Close with the limitations** (30 s): weak labels, recall, and the human-evaluation next step.

### 2.7 Commands to know by heart

```bash
pip install -r requirements.txt            # audit pipeline
pip install -e .                           # installs the `auditix` command
auditix --help
auditix build-index                        # embed the policies (needs GEMINI_API_KEY)
auditix audit-local ./some_project -o audit_output
auditix audit-repo https://github.com/org/repo -o audit_output
auditix audit-local ./some_project --no-classifier        # LLM-only mode
auditix audit-local ./some_project --no-explain-fast-path # fewer LLM calls
pip install -r requirements-train.txt
auditix train --epochs 5                   # needs a GPU and HF_TOKEN for gated data
python -m pytest -q                        # 50 offline tests
```

---

## Part 3 — Knowledge checklist

Tick each one only when you can explain it without notes.

- [ ] Walk through the 8 pipeline stages from memory
- [ ] Draw the routing rules and explain both boundaries (0.35 and 0.65)
- [ ] Explain why the summary is embedded, not the code
- [ ] Explain asymmetric retrieval (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`)
- [ ] Explain LoRA with the formula `W + (α/r)·BA`, and say which matrices are adapted
- [ ] Explain why 0.94% trainable parameters is the point of LoRA
- [ ] Explain the dataset pipeline: sources, keyword labels, dedupe, balance, split
- [ ] Explain the duplicate leak and its fix
- [ ] Interpret the confusion matrix and the precision/recall gap
- [ ] Explain graceful degradation and the LLM-only fallback
- [ ] Explain the retry logic and why it reads the server's wait time
- [ ] Explain the JSON parsing fallback and why it fails closed
- [ ] Explain AST chunking and the nested-method double count
- [ ] Explain dependency injection and how the tests use fakes
- [ ] Explain the HDFS comparison from `explanation.md` §10 in two minutes
- [ ] Name three things the original README got wrong, and how they were corrected
