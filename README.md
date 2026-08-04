# financial-complaints

Compare how three LLMs respond to real consumer finance complaints.

`generate_llm_responses.py` reads the CFPB consumer complaint dataset, builds a
customer-service prompt from each complaint's `issue` and `sub_issue` fields, sends it to
ChatGPT, Claude, and Mistral, and writes all three replies side by side to a CSV for
comparison.

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

Each complaint is printed as it is processed, and the results are saved next to the
dataset as `llm_responses_sample.csv` with columns:

| Column | Contents |
| --- | --- |
| `Row` | 1-based row number from the dataset |
| `Issue` | The `issue` field |
| `Sub_Issue` | The `sub_issue` field, blank where absent |
| `ChatGPT` | Response from `gpt-4o-mini` |
| `Claude` | Response from `claude-sonnet-5` |
| `Mistral` | Response from `mistral-small-latest` |

By default only the first 10 rows are processed, so a run costs a few cents. Change
`df.head(10)` to widen the sample — the dataset has ~555,000 rows, so run the full thing
only if you mean to.

## How the prompt is built

The dataset splits a complaint's category across two columns, and `sub_issue` is messy:
it is empty for a large share of rows, and sometimes just repeats `issue` verbatim. The
script handles both, so the prompt gets the extra detail only when it adds something:

```
You are a customer service representative for a financial institution.
Briefly respond to the following consumer complaint in 2 sentences.
Issue: Incorrect information on credit report
Sub-issue: Account status
```

When `sub_issue` is blank or duplicates `issue`, the `Sub-issue:` line is omitted
entirely. Column lookup ignores case and `-`/`_`/space differences, so `issue`/`Issue`
and `sub_issue`/`Sub-issue` all resolve — the Kaggle export and the current CFPB export
disagree on this.

Each model is called in its own `try`/`except`, so one provider being down or out of
credits leaves the other two intact; the error text lands in that model's cell.

## Notes

- `mistralai` 2.x moved the `Mistral` class to `mistralai.client`. On 1.x the import is
  `from mistralai import Mistral` instead.
- `max_tokens=150` keeps replies to roughly the requested two sentences.
- These are LLM-generated drafts for research and comparison, not compliance-reviewed
  customer communications.
