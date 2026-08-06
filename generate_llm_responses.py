import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import Mistral

# Load keys from a local .env file (real environment variables win over it)
load_dotenv()

# 1. Dataset Path (override with the CONSUMER_COMPLAINTS_CSV env var)
DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "consumer_complaints.csv"
)
# `or` rather than a getenv default: a blank line in .env sets the variable to ""
dataset_path = os.getenv("CONSUMER_COMPLAINTS_CSV") or DEFAULT_DATASET

# How many complaints to process. Each one costs 9 API calls
# (3 prompt variants x 3 models), so raise this deliberately.
NUM_ROWS = int(os.getenv("LLM_SAMPLE_ROWS") or "10")


# 2. API keys come from the environment only -- never hardcode them in this file.
def require_key(name: str) -> str:
    key = os.getenv(name)
    if not key:
        raise SystemExit(
            f"Missing {name}. Set it before running, e.g. in PowerShell:\n"
            f'    $env:{name} = "your-key-here"'
        )
    return key


OPENAI_API_KEY = require_key("OPENAI_API_KEY")
ANTHROPIC_API_KEY = require_key("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = require_key("MISTRAL_API_KEY")

# Initialize SDK Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)


# 3. Define Generator Functions
def get_chatgpt_response(prompt: str, max_tokens: int) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def get_claude_response(prompt: str, max_tokens: int) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # content can start with a non-text block (e.g. ThinkingBlock), so collect
    # the text blocks rather than assuming content[0] is one.
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    return text.strip()


def get_mistral_response(prompt: str, max_tokens: int) -> str:
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


MODELS = {
    "ChatGPT": get_chatgpt_response,
    "Claude": get_claude_response,
    "Mistral": get_mistral_response,
}

# 4. Three prompt variants, differing on persona, tone, and output structure.
# Each takes a {complaint} placeholder. Vary one axis at a time so differences
# in the responses are attributable.
PROMPT_VARIANTS = [
    {
        # Baseline: minimal instruction, no persona, no tone guidance.
        "name": "v1_terse",
        "max_tokens": 150,
        "template": (
            "Respond to the following consumer complaint in 2 sentences.\n\n"
            "{complaint}"
        ),
    },
    {
        # Adds a persona plus explicit empathy and ownership guidance.
        "name": "v2_empathetic",
        "max_tokens": 300,
        "template": (
            "You are an experienced customer service representative at a financial "
            "institution, known for being warm and genuinely helpful.\n\n"
            "A customer has raised the complaint below. Reply directly to them in 3-4 "
            "sentences. Acknowledge the frustration this has caused, take ownership on "
            "behalf of the institution, and describe one concrete next step you will "
            "take. Write in plain language, avoid corporate jargon, and do not ask the "
            "customer to repeat information they have already provided.\n\n"
            "{complaint}"
        ),
    },
    {
        # Same persona, but a rigid labelled format with compliance constraints.
        "name": "v3_structured",
        "max_tokens": 350,
        "template": (
            "You are a customer service representative at a financial institution "
            "operating under CFPB complaint-handling rules.\n\n"
            "Draft a reply to the complaint below using exactly this format:\n"
            "Acknowledgement: <one sentence restating the issue>\n"
            "Next step: <one sentence on what the institution will do, and by when>\n"
            "What we need from you: <one sentence, or 'Nothing at this time.'>\n\n"
            "Constraints: do not promise a specific outcome, do not admit legal "
            "liability, do not give legal or tax advice, and do not invent account "
            "numbers, dates, or dollar amounts.\n\n"
            "{complaint}"
        ),
    },
]

# 5. Load Dataset
df = pd.read_csv(dataset_path, low_memory=False)


# Handle column naming variations (e.g. 'Issue'/'issue', 'sub_issue'/'Sub-issue')
def find_col(*candidates):
    """Match a column ignoring case and any -, _ or space separators."""
    norm = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


issue_col = find_col("issue")
sub_issue_col = find_col("sub_issue", "subissue")

sample_df = df.head(NUM_ROWS)

# Errors that will not fix themselves on retry (no credits, bad key). Once a
# model hits one, reuse the message instead of making the remaining calls.
FATAL_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "authentication_error",
    "invalid_api_key",
    "unauthorized",
)
blocked = {}


def call_model(model_name, fn, prompt, max_tokens):
    """Return (response_text, is_error). Never raises."""
    if model_name in blocked:
        return blocked[model_name], True
    try:
        return fn(prompt, max_tokens), False
    except Exception as e:
        message = f"{model_name} Error: {e}"
        if any(marker in str(e).lower() for marker in FATAL_MARKERS):
            blocked[model_name] = message
            print(f"  ! {model_name} disabled for the rest of the run: {e}")
        return message, True


long_rows = []
wide_rows = []

total_calls = len(sample_df) * len(PROMPT_VARIANTS) * len(MODELS)
print(
    f"--- {len(sample_df)} complaints x {len(PROMPT_VARIANTS)} prompts x "
    f"{len(MODELS)} models = {total_calls} responses ---\n"
)

for idx, row in sample_df.iterrows():
    issue_text = str(row[issue_col]).strip()

    sub_issue_raw = row[sub_issue_col]
    if pd.isna(sub_issue_raw) or not str(sub_issue_raw).strip():
        sub_issue_text = ""
    else:
        sub_issue_text = str(sub_issue_raw).strip()
        if sub_issue_text.lower() == issue_text.lower():
            sub_issue_text = ""

    # Complaint description shared by all three prompt variants
    if sub_issue_text:
        complaint = f"Issue: {issue_text}\nSub-issue: {sub_issue_text}"
    else:
        complaint = f"Issue: {issue_text}"

    print(f"================ Row {idx + 1} ================")
    print(f"Issue: {issue_text}")
    print(f"Sub-issue: {sub_issue_text or '(none)'}")

    wide_row = {"Row": idx + 1, "Issue": issue_text, "Sub_Issue": sub_issue_text}

    for variant in PROMPT_VARIANTS:
        prompt = variant["template"].format(complaint=complaint)
        wide_row[f"prompt__{variant['name']}"] = prompt

        print(f"\n  --- prompt {variant['name']} ---")

        for model_name, fn in MODELS.items():
            text, is_error = call_model(model_name, fn, prompt, variant["max_tokens"])

            print(f"\n  [{model_name}]:\n  {text}")

            long_rows.append({
                "Row": idx + 1,
                "Issue": issue_text,
                "Sub_Issue": sub_issue_text,
                "Prompt_Variant": variant["name"],
                "Prompt_Text": prompt,
                "Model": model_name,
                "Response": text,
                "Response_Chars": 0 if is_error else len(text),
                "Is_Error": is_error,
            })
            wide_row[f"{variant['name']}__{model_name}"] = text

    wide_rows.append(wide_row)
    print("\n" + "-" * 60)

# 6. Save both shapes: long for analysis, wide for eyeballing side by side
out_dir = os.path.dirname(dataset_path)
long_df = pd.DataFrame(long_rows)
wide_df = pd.DataFrame(wide_rows)

long_path = os.path.join(out_dir, "llm_responses_long.csv")
wide_path = os.path.join(out_dir, "llm_responses_wide.csv")
long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")

errors = int(long_df["Is_Error"].sum())
print(f"\nCollected {len(long_df)} responses ({errors} errors)")
print(f"  long (one row per response): {long_path}")
print(f"  wide (9 responses per row):  {wide_path}")

# Mean response length per prompt variant x model, ignoring errors
ok = long_df[~long_df["Is_Error"]]
if not ok.empty:
    print("\nMean response length in characters:")
    print(
        ok.pivot_table(
            index="Prompt_Variant",
            columns="Model",
            values="Response_Chars",
            aggfunc="mean",
        ).round(0).to_string()
    )
