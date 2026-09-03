# financial-complaints

Compare how three LLMs respond to real consumer finance complaints.

The pipeline has two steps:

1. **`feature_engineering.py`** reads the full CFPB consumer complaint database, keeps
   only complaints with real context (an issue, a distinct sub-issue, and the consumer's
   own narrative), adds engineered features, and picks **10,000 rows spread across as many
   product / issue / sub-issue categories as possible**.
2. **`generate_llm_responses.py`** builds **three different prompts** from each selected
   complaint and sends each one to ChatGPT, Claude, and Mistral. That is **9 responses per
   complaint, 90,000 in total**, so you can compare how much the wording of a prompt
   changes the answer, and how differently each model reacts to the same change.

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
variables take precedence over `.env`, so you can also just set them in your shell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Get keys from [OpenAI](https://platform.openai.com/api-keys),
[Anthropic](https://console.anthropic.com/settings/keys), and
[Mistral](https://console.mistral.ai/api-keys). All three are paid APIs; the script
reports a per-model error inline if a key is missing credits rather than crashing.

**Running inside Claude Code on the web?** Its cloud environment reserves
`ANTHROPIC_API_KEY` for Claude Code itself, so a value you set under that name never
reaches the session (and `ANTHROPIC_BASE_URL` is pointed at a local proxy). Add the
Anthropic key to the environment as `CLAUDE_API_KEY` instead; the script accepts either
name and always calls the public Anthropic API.

### 3. Get the dataset

The raw data is the CFPB's public complaint database, downloadable without an account:

```powershell
Invoke-WebRequest https://files.consumerfinance.gov/ccdb/complaints.csv.zip -OutFile complaints.csv.zip
Expand-Archive complaints.csv.zip dataset
```

It is ~9 GB unzipped (about 8 million complaints), so it is gitignored rather than
committed. Put it at `dataset/complaints.csv`, or point at it with
`CFPB_COMPLAINTS_CSV`.

The selected subset, `dataset/complaints_10k.csv`, **is** committed, so you can skip the
download and step 1 entirely if you only want to run the LLM comparison.

## Step 1: feature engineering

```powershell
python feature_engineering.py
```

Streams the export in chunks (it never loads the whole file), so it runs on a laptop.
Takes a few minutes. Output goes next to the input file:

- `complaints_10k.csv` — the selected rows with engineered features
- `complaints_10k_summary.md` — coverage statistics for the selection

### Filters

Rows are kept only if all of the following hold:

| Filter | Why |
| --- | --- |
| `Issue` and `Sub-issue` both present, and `Sub-issue` is not a copy of `Issue` | The sub-issue is what makes the category specific |
| Narrative present and at least 200 characters | Shorter narratives rarely describe the actual problem |
| At most 25% of the narrative is `XXXX` redaction | Heavily redacted text has little left to respond to |
| Narrative is not an exact duplicate of another row | The export contains many repeated submissions |

### Engineered features

| Column | Contents |
| --- | --- |
| `row_id` | 1-based id used as `Row` in the LLM output |
| `narrative_chars`, `narrative_words` | Narrative length |
| `issue_words`, `sub_issue_words`, `detail_score` | How specific the category labels are (`detail_score` is their sum) |
| `redaction_ratio` | Share of narrative characters inside `XXXX` redactions |
| `year`, `month`, `days_to_company` | Timing, from `Date received` and `Date sent to company` |
| `has_tags`, `has_public_response`, `timely_response` | Boolean flags from the source columns |
| `response_category`, `is_disputed_or_relief` | Company outcome, and whether it involved relief |
| `quality_score` | Ranking score: narrative length in the 300–2500 char sweet spot, detailed labels, low redaction |

### Selection

Every `(Product, Sub-product, Issue, Sub-issue)` combination in the pool gets up to `CAP`
rows, where `CAP` is the smallest number that reaches the target. Inside a combination the
highest `quality_score` rows win. Rare categories are therefore always represented and
the dominant credit-reporting categories cannot crowd them out. `TARGET_ROWS` (default
10000) and `SEED` (default 42) are environment overrides.

## Step 2: LLM responses

Test on a handful of rows first — every row is 9 paid API calls:

```powershell
$env:LLM_SAMPLE_ROWS = "5"
python generate_llm_responses.py
```

Then run everything:

```powershell
$env:LLM_SAMPLE_ROWS = "all"
python generate_llm_responses.py
```

Calls run concurrently, `LLM_CONCURRENCY` per provider (default 4, so 12 in flight).
Rate limits, timeouts, and 5xx errors retry with backoff; an unrecoverable error (no
credits, bad key) disables that provider for the rest of the run instead of failing the
same way thousands of times.

Every reply is appended to `llm_responses_long.csv` the moment it arrives. **If the run
is interrupted, just start it again**: rows already recorded are skipped, and rows that
only hold an error are retried. Delete the file to start over.

Two CSVs are written next to the input file — the same data in both shapes, because each
is convenient for a different job.

**`llm_responses_long.csv`** — one row per response, 9 rows per complaint. Use this for
analysis (`groupby`, `pivot_table`).

| Column | Contents |
| --- | --- |
| `Row` | `row_id` from the input |
| `Complaint_ID`, `Product`, `Sub_Product`, `Issue`, `Sub_Issue` | The source fields |
| `Prompt_Variant` | `v1_terse`, `v2_empathetic`, or `v3_structured` |
| `Prompt_Text` | The full prompt actually sent |
| `Model` | `ChatGPT`, `Claude`, or `Mistral` |
| `Response` | The reply, or the error message if the call failed |
| `Response_Chars` | Reply length, `0` on error |
| `Is_Error` | Boolean, for filtering failures out of aggregates |
| `Prompt_Tokens`, `Completion_Tokens` | Usage as reported by the provider, for cost tracking |
| `Latency_s` | Wall-clock seconds for the call |

**`llm_responses_wide.csv`** — one row per complaint: the source fields, the 3 rendered
prompts (`prompt__v1_terse`, …) and the 9 responses (`v1_terse__Claude`, …). Use this to
read the variants side by side.

The run ends with a mean-response-length table per variant × model and total tokens per
model.

## The three prompt variants

All three receive the same complaint text and differ only in framing, so differences in
the output are attributable to the prompt. They are defined in `PROMPT_VARIANTS` near the
top of the script — edit the templates there to run your own comparison.

| Variant | Framing | `max_tokens` |
| --- | --- | --- |
| `v1_terse` | Bare instruction, no persona, no tone guidance. The baseline. | 150 |
| `v2_empathetic` | Adds a persona and asks for acknowledgement, ownership, and one concrete next step in plain language. | 300 |
| `v3_structured` | Same persona, but a rigid `Acknowledgement / Next step / What we need from you` format plus compliance constraints (no promised outcomes, no admitted liability, no invented figures). | 350 |

### How the complaint text is built

```
Issue: Incorrect information on your report
Sub-issue: Information belongs to someone else
Complaint: I have disputed this account three times with XXXX and ...
```

The `Sub-issue:` line is omitted when it is blank or duplicates `Issue`. The narrative is
truncated at a word boundary to `LLM_NARRATIVE_CHARS` (default 1500) so a handful of very
long complaints do not dominate the token budget; the prompt itself is what the models saw,
and it is saved in full in `Prompt_Text`.

Column lookup ignores case and `-`/`_`/space differences, so the script also runs on the
old Kaggle export (`issue`/`sub_issue`, no narrative for most rows) via
`LLM_INPUT_CSV=path/to/consumer_complaints.csv`.

## Notes

- `mistralai` 2.x moved the `Mistral` class to `mistralai.client`. On 1.x the import is
  `from mistralai import Mistral` instead.
- Claude's `content` list can begin with a non-text block, so the script collects the
  text blocks rather than reading `content[0].text`.
- Leave a key blank in `.env` and it is set to `""`, not unset — the script uses
  `os.getenv(...) or default` so blank behaves as absent.
- The Anthropic SDK honours `ANTHROPIC_BASE_URL` if it is set in your environment, so the
  script pins the public API URL explicitly. Set `LLM_ANTHROPIC_BASE_URL` to route Claude
  calls elsewhere on purpose.
- Narratives contain `XXXX` where the CFPB redacted names, dates, and amounts. The models
  see them as-is.
- These are LLM-generated drafts for research and comparison, not compliance-reviewed
  customer communications.
