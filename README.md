# financial-complaints

Compare how three LLMs respond to real consumer finance complaints.

Two scripts, run in order:

1. **`build_dataset.py`** curates a high-quality, category-diverse subset of the CFPB
   consumer complaint dataset (default 10,000 rows) and engineers analysis features.
2. **`generate_llm_responses.py`** sends every curated complaint through **three prompt
   variants** to **three models** (ChatGPT, Claude, Mistral): 9 responses per complaint,
   **90,000 responses** for the 10k set. It is async, checkpointed, and resumable.

## Setup

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add your API keys

Copy the template and fill in your own keys:

```powershell
Copy-Item .env.example .env
```

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
```

`.env` is gitignored and must stay that way — never commit real keys. Real environment
variables take precedence over `.env`. All three are paid APIs; **make sure each account
has credit before a full run** (the script detects a no-credit / bad-key error on the first
call and stops using that provider instead of failing 30,000 times).

Get keys from [OpenAI](https://platform.openai.com/api-keys),
[Anthropic](https://console.anthropic.com/settings/keys), and
[Mistral](https://console.mistral.ai/api-keys).

### 3. Get the dataset

Download **US Consumer Finance Complaints** from Kaggle:

<https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints>

Unzip it and place `consumer_complaints.csv` at `dataset/consumer_complaints.csv`:

```
financial-complaints/
└── dataset/
    └── consumer_complaints.csv
```

The file is ~167 MB, over GitHub's 100 MB per-file limit, so it is gitignored. To keep it
elsewhere:

```powershell
$env:CONSUMER_COMPLAINTS_CSV = "D:\data\consumer_complaints.csv"
```

The current CFPB export (`Date received`, `Sub-issue`, `Consumer complaint narrative`, …)
works too; column names are matched ignoring case and separators.

## Step 1 — curate the 10,000 rows

```powershell
python build_dataset.py
```

Writes `dataset/curated_complaints.csv` and `dataset/curated_complaints_summary.json`.

### Filtering

Rows are kept only when they provide real context:

| Rule | Why |
| --- | --- |
| `issue` and `sub_issue` both present | The sub-issue is what makes the complaint specific. |
| `sub_issue` is not generic (`Other`, `N/A`, …) and not a verbatim repeat of `issue` | Those add no information. |
| `product` present | Needed for the category key. |
| Exact-duplicate narratives dropped | Re-submitted complaints would give the LLMs identical prompts. |

### Feature engineering

Each curated row carries these engineered columns in addition to the raw fields:

| Column | Meaning |
| --- | --- |
| `category` | `product \| issue \| sub_issue` — the stratification key |
| `issue_pair` | `issue \| sub_issue` |
| `has_narrative`, `narrative_chars`, `narrative_words` | Consumer narrative presence and size |
| `narrative_redactions`, `narrative_redaction_ratio` | Count / share of `XXXX` redaction tokens |
| `issue_words`, `sub_issue_words`, `sub_issue_specificity` | How descriptive the labels are |
| `has_sub_product`, `has_company_response` | Extra context flags |
| `year`, `month` | Parsed from `date_received` |
| `timely_response_flag`, `consumer_disputed_flag` | Booleans from the Yes/No columns |
| `quality_score` | Context-richness score (≈0–10) used for ranking |

`quality_score` rewards a narrative (+4), a narrative in the 500–3,000 character band (+2.5),
a specific sub-issue, and a sub-product, and penalises heavy redaction. Within a category,
rows are taken in score order, so complaints with a consumer narrative are always picked
before label-only rows.

### Selection: coverage first, then prevalence

1. Every `category` gets at least `--min-per-category` rows (default 5) if that many exist.
2. The remaining budget is spread across categories in proportion to
   **√(available rows)** — frequent categories stay prominent, but the long tail is not
   drowned out the way a plain proportional sample would.

`--target`, `--min-per-category`, and `--seed` are all flags; the selection is
deterministic for a given seed. The summary JSON reports how many products / issues /
sub-issues / categories made it in, the narrative share, and the smallest categories.

## Step 2 — generate the 90,000 responses

Smoke test first (45 calls):

```powershell
python generate_llm_responses.py --limit 5
```

Full run:

```powershell
python generate_llm_responses.py
```

Every response is appended to `dataset/llm_responses_raw.jsonl` the moment it arrives.
**If the run stops for any reason — crash, laptop sleep, a provider running out of
credit — just run the same command again and it resumes where it left off.**

Useful flags:

| Flag | Effect |
| --- | --- |
| `--limit N` | Only the first N complaints (or `LLM_SAMPLE_ROWS`) |
| `--models Claude,Mistral` | Subset of providers |
| `--retry-errors` | Redo calls that were recorded as errors |
| `--dry-run` | No API calls; exercises scheduling, checkpointing and outputs |
| `--finalize-only` | Rebuild the CSVs from the checkpoint without calling anything |
| `--concurrency-openai/claude/mistral` | Parallel requests per provider (defaults 8 / 4 / 2) |
| `--max-attempts`, `--base-backoff`, `--max-backoff` | Retry policy for 429 / 5xx / timeouts |

Raise the concurrency once your rate-limit tier allows it; at the defaults the Claude leg
(30,000 calls) is the slowest at several hours. Models default to `gpt-4o-mini`,
`claude-sonnet-5`, `mistral-small-latest`; override with `OPENAI_MODEL`, `CLAUDE_MODEL`,
`MISTRAL_MODEL`.

### Outputs (all in `dataset/`, all gitignored)

| File | Contents |
| --- | --- |
| `llm_responses_raw.jsonl` | Append-only checkpoint: one JSON object per call (key, response, tokens, latency, attempts, error) |
| `llm_prompts.csv` | One row per complaint × variant: the exact prompt text sent |
| `llm_responses_long.csv` | One row per response, 9 per complaint. Use this for `groupby` / `pivot_table`. |
| `llm_responses_wide.csv` | One row per complaint: 3 prompt columns + 9 response columns, side by side |

`llm_responses_long.csv` columns: `Complaint_ID`, `Product`, `Sub_Product`, `Issue`,
`Sub_Issue`, `Category`, `has_narrative`, `narrative_chars`, `Prompt_Variant`, `Model`,
`Model_ID`, `Response`, `Response_Chars`, `Is_Error`, `Error_Type`, `Finish_Reason`,
`Input_Tokens`, `Output_Tokens`, `Attempts`, `Latency_ms`.

The run ends with a response-count table, a mean-length table per variant × model, token
usage with an approximate cost per model (edit `PRICES` in the script), and an error
breakdown.

## The three prompt variants

All three receive the same complaint block and differ only in framing, so differences in
the output are attributable to the prompt. They live in `PROMPT_VARIANTS` near the top of
`generate_llm_responses.py`.

| Variant | Framing | `max_tokens` |
| --- | --- | --- |
| `v1_terse` | Bare instruction, no persona, no tone guidance. The baseline. | 200 |
| `v2_empathetic` | Adds a persona and asks for acknowledgement, ownership, and one concrete next step in plain language. | 400 |
| `v3_structured` | Same persona, but a rigid `Acknowledgement / Next step / What we need from you` format plus compliance constraints. | 450 |

### How the complaint block is built

```
Product: Debt collection
Sub-product: Medical
Issue: Disclosure verification of debt
Sub-issue: Right to dispute notice not received

Customer's description (personal details redacted as XXXX):
On XX/XX/XXXX I received a letter stating that I owed ...
```

Lines are omitted when the field is blank or (for `Sub-issue`) when it repeats the issue.
Narratives longer than 2,500 characters are cut at a word boundary with a `[...]` marker
(`LLM_MAX_NARRATIVE_CHARS` to change).

## Tests

No dataset or API key needed — a schema-faithful synthetic fixture is generated on the fly:

```powershell
python -m pytest tests/ -q
```

`tests/make_fixture.py` can also write a fixture CSV to try the scripts end to end:

```powershell
python tests/make_fixture.py --rows 20000 --output dataset/fixture_complaints.csv
python build_dataset.py --input dataset/fixture_complaints.csv --output dataset/fixture_curated.csv --target 1000
python generate_llm_responses.py --dry-run --rows-file dataset/fixture_curated.csv --limit 20
```

## Notes

- `mistralai` 2.x moved the `Mistral` class to `mistralai.client`. On 1.x the import is
  `from mistralai import Mistral` instead.
- Claude is called with extended thinking disabled so the whole `max_tokens` budget goes
  to the visible reply; its `content` list is scanned for text blocks rather than reading
  `content[0].text`.
- Fatal provider errors (bad key, no credit) are not written to the checkpoint, so the
  affected calls are retried automatically on the next run. Non-fatal errors are written
  with `Is_Error=True`; use `--retry-errors` to redo them.
- These are LLM-generated drafts for research and comparison, not compliance-reviewed
  customer communications.
