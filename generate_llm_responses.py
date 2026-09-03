"""Send every complaint through 3 prompt variants x 3 LLMs and record the replies.

Input is the feature-engineered subset from feature_engineering.py by default
(dataset/complaints_10k.csv). The complaint shown to each model is built from
the Issue and Sub-issue labels plus the consumer's own narrative when present.

Built for long runs:
  - calls run concurrently, with a separate concurrency limit per provider
  - transient errors (rate limits, timeouts, 5xx) retry with backoff;
    unrecoverable ones (no credits, bad key) disable that provider for the run
  - every reply is appended to llm_responses_long.csv as it arrives, and a
    rerun skips anything already recorded, so an interrupted run resumes

Environment:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY   required
  LLM_INPUT_CSV        input file (default dataset/complaints_10k.csv)
  LLM_SAMPLE_ROWS      rows to process, or "all" (default 10)
  LLM_CONCURRENCY      parallel calls per provider (default 4)
  LLM_NARRATIVE_CHARS  narrative truncation length (default 1500)
  LLM_VERBOSE          1 to print every reply (default: only for <=100 calls)
"""

import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import Mistral

# Load keys from a local .env file (real environment variables win over it)
load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
# `or` rather than a getenv default: a blank line in .env sets the variable to ""
input_path = (os.getenv("LLM_INPUT_CSV") or os.getenv("CONSUMER_COMPLAINTS_CSV")
              or os.path.join(HERE, "dataset", "complaints_10k.csv"))

_rows_env = (os.getenv("LLM_SAMPLE_ROWS") or "10").strip().lower()
NUM_ROWS = None if _rows_env == "all" else int(_rows_env)
CONCURRENCY = int(os.getenv("LLM_CONCURRENCY") or "4")
NARRATIVE_CHARS = int(os.getenv("LLM_NARRATIVE_CHARS") or "1500")
MAX_RETRIES = 6


# 1. API keys come from the environment only -- never hardcode them in this file.
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

openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=90)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=0, timeout=90)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)


# 2. Generator functions. Each returns (text, prompt_tokens, completion_tokens).
def get_chatgpt_response(prompt: str, max_tokens: int):
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    u = r.usage
    return r.choices[0].message.content.strip(), u.prompt_tokens, u.completion_tokens


def get_claude_response(prompt: str, max_tokens: int):
    r = anthropic_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # content can start with a non-text block, so collect the text blocks
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text.strip(), r.usage.input_tokens, r.usage.output_tokens


def get_mistral_response(prompt: str, max_tokens: int):
    r = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    u = r.usage
    return (r.choices[0].message.content.strip(),
            getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))


MODELS = {
    "ChatGPT": get_chatgpt_response,
    "Claude": get_claude_response,
    "Mistral": get_mistral_response,
}

# 3. Three prompt variants, differing on persona, tone, and output structure.
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

# 4. Load dataset
df = pd.read_csv(input_path, low_memory=False, dtype=str)


def find_col(*candidates, required=True):
    """Match a column ignoring case and any -, _ or space separators."""
    norm = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"None of {candidates} found in {list(df.columns)}")
    return None


issue_col = find_col("issue")
sub_issue_col = find_col("sub_issue", "subissue")
narrative_col = find_col("consumer_complaint_narrative", "narrative", required=False)
row_id_col = find_col("row_id", required=False)
complaint_id_col = find_col("complaint_id", required=False)
product_col = find_col("product", required=False)
sub_product_col = find_col("sub_product", required=False)

sample_df = df if NUM_ROWS is None else df.head(NUM_ROWS)


def cell(row, col):
    if col is None:
        return ""
    v = row[col]
    return "" if pd.isna(v) else str(v).strip()


def build_complaint(row):
    issue = cell(row, issue_col)
    sub_issue = cell(row, sub_issue_col)
    if sub_issue.lower() == issue.lower():
        sub_issue = ""
    narrative = cell(row, narrative_col)
    if len(narrative) > NARRATIVE_CHARS:
        cut = narrative[:NARRATIVE_CHARS]
        narrative = cut[: cut.rfind(" ")] + " [...]"

    lines = [f"Issue: {issue}"]
    if sub_issue:
        lines.append(f"Sub-issue: {sub_issue}")
    if narrative:
        lines.append(f"Complaint: {narrative}")
    return issue, sub_issue, "\n".join(lines)


# 5. Resilient calling
FATAL_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "authentication_error",
    "invalid_api_key",
    "unauthorized",
    "incorrect api key",
)
TRANSIENT_MARKERS = (
    "rate_limit", "rate limit", "429", "overloaded", "timeout", "timed out",
    "connection", "500", "502", "503", "504", "server_error", "temporarily",
)
blocked = {}
blocked_lock = threading.Lock()


def call_model(model_name, fn, prompt, max_tokens):
    """Return (text, prompt_tokens, completion_tokens, is_error, latency). Never raises."""
    if model_name in blocked:
        return blocked[model_name], None, None, True, 0.0
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            text, p_tok, c_tok = fn(prompt, max_tokens)
            return text, p_tok, c_tok, False, round(time.monotonic() - t0, 2)
        except Exception as e:
            err = str(e)
            low = err.lower()
            if any(m in low for m in FATAL_MARKERS):
                message = f"{model_name} Error: {err}"
                with blocked_lock:
                    if model_name not in blocked:
                        blocked[model_name] = message
                        print(f"  ! {model_name} disabled for the rest of the run: {err}",
                              flush=True)
                return message, None, None, True, 0.0
            transient = any(m in low for m in TRANSIENT_MARKERS)
            if attempt == MAX_RETRIES or not transient:
                return f"{model_name} Error: {err}", None, None, True, 0.0
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)


# 6. Job list and checkpoint
out_dir = os.path.dirname(os.path.abspath(input_path))
long_path = os.path.join(out_dir, "llm_responses_long.csv")
wide_path = os.path.join(out_dir, "llm_responses_wide.csv")

LONG_COLUMNS = [
    "Row", "Complaint_ID", "Product", "Sub_Product", "Issue", "Sub_Issue",
    "Prompt_Variant", "Prompt_Text", "Model", "Response", "Response_Chars",
    "Is_Error", "Prompt_Tokens", "Completion_Tokens", "Latency_s",
]

done = set()
if os.path.exists(long_path):
    prev = pd.read_csv(long_path, dtype=str, keep_default_na=False)
    if set(LONG_COLUMNS) <= set(prev.columns):
        # Redo rows that only hold an error so a rerun fills them in
        ok_prev = prev[prev["Is_Error"].str.lower() != "true"]
        done = set(zip(ok_prev["Row"], ok_prev["Prompt_Variant"], ok_prev["Model"]))
        if len(done) < len(prev):
            ok_prev.to_csv(long_path, index=False, encoding="utf-8-sig")
        print(f"Resuming: {len(done):,} responses already recorded in {long_path}")
    else:
        print(f"Existing {long_path} has a different layout; starting fresh")
        os.remove(long_path)

jobs = []
for i, (_, row) in enumerate(sample_df.iterrows()):
    row_key = cell(row, row_id_col) or str(i + 1)
    issue, sub_issue, complaint = build_complaint(row)
    meta = {
        "Row": row_key,
        "Complaint_ID": cell(row, complaint_id_col),
        "Product": cell(row, product_col),
        "Sub_Product": cell(row, sub_product_col),
        "Issue": issue,
        "Sub_Issue": sub_issue,
    }
    for variant in PROMPT_VARIANTS:
        prompt = variant["template"].format(complaint=complaint)
        for model_name, fn in MODELS.items():
            if (row_key, variant["name"], model_name) in done:
                continue
            jobs.append((meta, variant, prompt, model_name, fn))

total_calls = len(sample_df) * len(PROMPT_VARIANTS) * len(MODELS)
VERBOSE = (os.getenv("LLM_VERBOSE") or "").strip() == "1" or total_calls <= 100
print(
    f"--- {len(sample_df):,} complaints x {len(PROMPT_VARIANTS)} prompts x "
    f"{len(MODELS)} models = {total_calls:,} responses; {len(jobs):,} to do, "
    f"{CONCURRENCY} parallel calls per provider ---\n", flush=True
)

# 7. Run
write_lock = threading.Lock()
new_file = not os.path.exists(long_path)
long_file = open(long_path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8")
writer = csv.DictWriter(long_file, fieldnames=LONG_COLUMNS)
if new_file:
    writer.writeheader()
    long_file.flush()

semaphores = {name: threading.Semaphore(CONCURRENCY) for name in MODELS}
progress = {"done": 0, "errors": 0}
started = time.monotonic()


def run_job(job):
    meta, variant, prompt, model_name, fn = job
    with semaphores[model_name]:
        text, p_tok, c_tok, is_error, latency = call_model(
            model_name, fn, prompt, variant["max_tokens"])
    record = {
        **meta,
        "Prompt_Variant": variant["name"],
        "Prompt_Text": prompt,
        "Model": model_name,
        "Response": text,
        "Response_Chars": 0 if is_error else len(text),
        "Is_Error": is_error,
        "Prompt_Tokens": "" if p_tok is None else p_tok,
        "Completion_Tokens": "" if c_tok is None else c_tok,
        "Latency_s": latency,
    }
    with write_lock:
        writer.writerow(record)
        long_file.flush()
        progress["done"] += 1
        progress["errors"] += int(is_error)
        n = progress["done"]
        if VERBOSE:
            print(f"[row {meta['Row']} | {variant['name']} | {model_name}] "
                  f"({latency}s)\n{text}\n", flush=True)
        elif n % 100 == 0 or n == len(jobs):
            elapsed = time.monotonic() - started
            rate = n / elapsed if elapsed else 0
            remaining = (len(jobs) - n) / rate if rate else 0
            print(f"  {n:,}/{len(jobs):,} done, {progress['errors']} errors, "
                  f"{rate:.1f}/s, ~{remaining / 60:.0f} min left", flush=True)
    return record


try:
    with ThreadPoolExecutor(max_workers=CONCURRENCY * len(MODELS)) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for f in as_completed(futures):
            f.result()
finally:
    long_file.close()

# 8. Save both shapes: long for analysis, wide for eyeballing side by side
long_df = pd.read_csv(long_path, dtype=str, keep_default_na=False)
long_df["Is_Error"] = long_df["Is_Error"].str.lower() == "true"
long_df["Response_Chars"] = pd.to_numeric(long_df["Response_Chars"], errors="coerce")

keys = ["Row", "Complaint_ID", "Product", "Sub_Product", "Issue", "Sub_Issue"]
prompts = (long_df.drop_duplicates(["Row", "Prompt_Variant"])
           .pivot(index="Row", columns="Prompt_Variant", values="Prompt_Text"))
prompts.columns = [f"prompt__{c}" for c in prompts.columns]
responses = long_df.pivot_table(index="Row", columns=["Prompt_Variant", "Model"],
                                values="Response", aggfunc="first")
responses.columns = [f"{v}__{m}" for v, m in responses.columns]
wide_df = (long_df.drop_duplicates("Row")[keys].set_index("Row")
           .join(prompts).join(responses).reset_index())
wide_df["_order"] = pd.to_numeric(wide_df["Row"], errors="coerce")
wide_df = wide_df.sort_values("_order").drop(columns="_order")
wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")

errors = int(long_df["Is_Error"].sum())
elapsed_min = (time.monotonic() - started) / 60
print(f"\nCollected {len(long_df):,} responses ({errors} errors) in {elapsed_min:.1f} min")
print(f"  long (one row per response): {long_path}")
print(f"  wide (9 responses per row):  {wide_path}")

ok = long_df[~long_df["Is_Error"]]
if not ok.empty:
    print("\nMean response length in characters:")
    print(ok.pivot_table(index="Prompt_Variant", columns="Model",
                         values="Response_Chars", aggfunc="mean").round(0).to_string())
    tok = ok.copy()
    for c in ("Prompt_Tokens", "Completion_Tokens"):
        tok[c] = pd.to_numeric(tok[c], errors="coerce")
    print("\nTotal tokens per model:")
    print(tok.groupby("Model")[["Prompt_Tokens", "Completion_Tokens"]].sum().to_string())
if errors:
    print("\nErrors per model:")
    print(long_df[long_df["Is_Error"]].groupby("Model").size().to_string())
