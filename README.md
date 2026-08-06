# financial-complaints

Compare how three LLMs respond to real consumer finance complaints.

`generate_llm_responses.py` reads the CFPB consumer complaint dataset, builds **three
different prompts** from each complaint's `issue` and `sub_issue` fields, and sends each
one to ChatGPT, Claude, and Mistral. That is **9 responses per complaint**, so you can
compare how much the wording of a prompt changes the answer, and how differently each
model reacts to the same change. Every rendered prompt is saved alongside its response.

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

### 3. Get the dataset

Download **US Consumer Finance Complaints** from Kaggle:

<https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints>

Unzip it and place `consumer_complaints.csv` at `dataset/consumer_complaints.csv`:

```
financial-complaints/
└── dataset/
    └── consumer_complaints.csv
```

The file is ~167 MB, which is over GitHub's 100 MB per-file limit, so it is gitignored
rather than committed. If you keep it somewhere else, point at it instead:

```powershell
$env:CONSUMER_COMPLAINTS_CSV = "D:\data\consumer_complaints.csv"
```

## Run

```powershell
python generate_llm_responses.py
```

Each response is printed as it arrives, and two CSVs are written next to the dataset —
the same data in both shapes, because each is convenient for a different job.

**`llm_responses_long.csv`** — one row per response, 9 rows per complaint. Use this for
analysis (`groupby`, `pivot_table`).

| Column | Contents |
| --- | --- |
| `Row` | 1-based row number from the dataset |
| `Issue` / `Sub_Issue` | The source fields; `Sub_Issue` blank where absent |
| `Prompt_Variant` | `v1_terse`, `v2_empathetic`, or `v3_structured` |
| `Prompt_Text` | The full prompt actually sent |
| `Model` | `ChatGPT`, `Claude`, or `Mistral` |
| `Response` | The reply, or the error message if the call failed |
| `Response_Chars` | Reply length, `0` on error |
| `Is_Error` | Boolean, for filtering failures out of aggregates |

**`llm_responses_wide.csv`** — one row per complaint with 15 columns: the 3 rendered
prompts (`prompt__v1_terse`, …) and the 9 responses (`v1_terse__Claude`, …). Use this to
read the variants side by side.

The run ends with a mean-response-length table per variant × model, which is a quick
signal that the prompts actually landed differently.

Row count is controlled by `LLM_SAMPLE_ROWS` (default 10). At 9 calls per complaint that
is 90 API calls, so raise it deliberately — the dataset has ~555,000 rows.

```powershell
$env:LLM_SAMPLE_ROWS = "50"
```

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

The dataset splits a complaint's category across two columns, and `sub_issue` is messy:
it is empty for a large share of rows, and sometimes just repeats `issue` verbatim. The
script handles both, so the prompt gets the extra detail only when it adds something:

```
Issue: Incorrect information on credit report
Sub-issue: Account status
```

When `sub_issue` is blank or duplicates `issue`, the `Sub-issue:` line is omitted
entirely. Column lookup ignores case and `-`/`_`/space differences, so `issue`/`Issue`
and `sub_issue`/`Sub-issue` all resolve — the Kaggle export and the current CFPB export
disagree on this.

Each call is wrapped individually, so one provider being down leaves the other two
intact; the error text lands in that cell with `Is_Error` set. If a provider returns an
unrecoverable error (no credits, bad key) it is skipped for the rest of the run instead
of failing the same way dozens of times.

## Notes

- `mistralai` 2.x moved the `Mistral` class to `mistralai.client`. On 1.x the import is
  `from mistralai import Mistral` instead.
- Claude's `content` list can begin with a non-text block, so the script collects the
  text blocks rather than reading `content[0].text`.
- Leave a key blank in `.env` and it is set to `""`, not unset — the script uses
  `os.getenv(...) or default` so blank behaves as absent.
- These are LLM-generated drafts for research and comparison, not compliance-reviewed
  customer communications.
