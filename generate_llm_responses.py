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
  CLAUDE_API_KEY       accepted instead of ANTHROPIC_API_KEY, for hosts that
                       reserve that name (e.g. Claude Code on the web)
  LLM_INPUT_CSV        input file (default dataset/complaints_10k.csv)
  LLM_SAMPLE_ROWS      rows to process, or "all" (default 10)
  LLM_CONCURRENCY      parallel calls per provider (default 4)
  LLM_NARRATIVE_CHARS  narrative truncation length (default 1500)
  LLM_VERBOSE          1 to print every reply (default: only for <=100 calls)
  LLM_VARIANTS         comma-separated prompt names to run (default: the three
                       Study 1 prompts); see FACTORIAL_VARIANTS for Study 2
  LLM_ROWS_FILE        text file of row_ids to restrict the run to
  LLM_OUTPUT_NAME      output file stem (default llm_responses)
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
def require_key(name: str, *aliases: str) -> str:
    for candidate in (name, *aliases):
        key = os.getenv(candidate)
        if key:
            return key
    raise SystemExit(
        f"Missing {name}. Set it before running, e.g. in PowerShell:\n"
        f'    $env:{name} = "your-key-here"'
    )


OPENAI_API_KEY = require_key("OPENAI_API_KEY")
# Claude Code on the web reserves ANTHROPIC_API_KEY for itself, so the key can
# also be supplied as CLAUDE_API_KEY.
ANTHROPIC_API_KEY = require_key("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
MISTRAL_API_KEY = require_key("MISTRAL_API_KEY")
# The same host points ANTHROPIC_BASE_URL at a local proxy, which the SDK would
# otherwise pick up; always use the public API unless overridden for this script.
ANTHROPIC_URL = os.getenv("LLM_ANTHROPIC_BASE_URL") or "https://api.anthropic.com"

openai_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=90)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_URL,
                             max_retries=0, timeout=90)
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

# 3b. Study 2: a 2x2x2 factorial around v2_empathetic, plus a hygiene cell and two
# paraphrases. Factors: A = empathetic framing (v2's persona + directives), B =
# the three-field format, C = the compliance constraints. All free-text cells ask
# for 3-4 sentences so length is not confounded. v2_empathetic is cell A00; the
# other seven cells are below. Select with LLM_VARIANTS (comma-separated names).
_PERSONA_A = ("You are an experienced customer service representative at a financial "
              "institution, known for being warm and genuinely helpful.\n\n")
_DIRECTIVES_A_FREE = (
    "A customer has raised the complaint below. Reply directly to them in 3-4 "
    "sentences. Acknowledge the frustration this has caused, take ownership on "
    "behalf of the institution, and describe one concrete next step you will "
    "take. Write in plain language, avoid corporate jargon, and do not ask the "
    "customer to repeat information they have already provided.\n\n")
_FORMAT_SPEC = (
    "Acknowledgement: <one sentence restating the issue>\n"
    "Next step: <one sentence on what the institution will do, and by when>\n"
    "What we need from you: <one sentence, or 'Nothing at this time.'>\n\n")
_FORMAT_PLAIN = "Draft a reply to the complaint below using exactly this format:\n" + _FORMAT_SPEC
_DIRECTIVES_A_FORMAT = (
    "A customer has raised the complaint below. Draft a reply to them using exactly "
    "this format:\n" + _FORMAT_SPEC +
    "Acknowledge the frustration this has caused, take ownership on behalf of the "
    "institution, and describe one concrete next step you will take. Write in plain "
    "language, avoid corporate jargon, and do not ask the customer to repeat "
    "information they have already provided.\n\n")
_CONSTRAINTS = (
    "Constraints: do not promise a specific outcome, do not admit legal liability, "
    "do not give legal or tax advice, and do not invent account numbers, dates, or "
    "dollar amounts.\n\n")
_BASE_FREE = "Reply to the following consumer complaint in 3-4 sentences.\n\n"
_HYGIENE = ("Write plain text only: no salutation, no sign-off, no markdown, and no "
            "bracketed placeholders such as [Customer's Name] or [date].\n\n")

FACTORIAL_VARIANTS = [
    {"name": "f_000_base", "max_tokens": 300, "template": _BASE_FREE + "{complaint}"},
    {"name": "f_0B0_format", "max_tokens": 350, "template": _FORMAT_PLAIN + "{complaint}"},
    {"name": "f_00C_constraints", "max_tokens": 300, "template": _BASE_FREE + _CONSTRAINTS + "{complaint}"},
    {"name": "f_0BC_format_constraints", "max_tokens": 350, "template": _FORMAT_PLAIN + _CONSTRAINTS + "{complaint}"},
    {"name": "f_A0C_empathetic_constraints", "max_tokens": 300,
     "template": _PERSONA_A + _DIRECTIVES_A_FREE + _CONSTRAINTS + "{complaint}"},
    {"name": "f_AB0_empathetic_format", "max_tokens": 350,
     "template": _PERSONA_A + _DIRECTIVES_A_FORMAT + "{complaint}"},
    {"name": "f_ABC_empathetic_format_constraints", "max_tokens": 350,
     "template": _PERSONA_A + _DIRECTIVES_A_FORMAT + _CONSTRAINTS + "{complaint}"},
    # mitigation cell: v2 plus output hygiene
    {"name": "h_A00_hygiene", "max_tokens": 300,
     "template": _PERSONA_A + _DIRECTIVES_A_FREE + _HYGIENE + "{complaint}"},
    # surface paraphrases of v2 (same content, reworded) to estimate wording noise
    {"name": "p_A00_para1", "max_tokens": 300, "template": (
        "You work in customer service at a financial institution and have a reputation "
        "for being kind and genuinely useful to customers.\n\n"
        "Below is a complaint from one of your customers. Write your response to them in "
        "three or four sentences. Recognise how frustrating this has been, accept "
        "responsibility on the institution's behalf, and set out one specific thing you "
        "will do next. Keep the language simple and free of corporate phrasing, and don't "
        "ask for details the customer has already given.\n\n{complaint}")},
    {"name": "p_A00_para2", "max_tokens": 300, "template": (
        "Take the role of a seasoned customer-service agent at a bank or lender, someone "
        "customers describe as warm and truly helpful.\n\n"
        "A customer has sent the complaint shown below. Answer them directly, using 3 to 4 "
        "sentences. Show that you understand the frustration involved, own the problem on "
        "behalf of the institution, and name one concrete action you will take. Use "
        "everyday words rather than jargon, and avoid asking the customer to repeat "
        "anything they have already told you.\n\n{complaint}")},
]
PROMPT_VARIANTS += FACTORIAL_VARIANTS
DEFAULT_VARIANTS = ["v1_terse", "v2_empathetic", "v3_structured"]
_variants_env = (os.getenv("LLM_VARIANTS") or "").strip()
_selected = [v.strip() for v in _variants_env.split(",") if v.strip()] if _variants_env else DEFAULT_VARIANTS
_unknown = set(_selected) - {v["name"] for v in PROMPT_VARIANTS}
if _unknown:
    raise SystemExit(f"Unknown LLM_VARIANTS: {sorted(_unknown)}")
PROMPT_VARIANTS = [v for v in PROMPT_VARIANTS if v["name"] in _selected]

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
# LLM_ROWS_FILE: restrict to the row_ids listed in a text file (one per line)
_rows_file = os.getenv("LLM_ROWS_FILE")
if _rows_file and row_id_col is not None:
    _keep = {ln.strip() for ln in open(_rows_file) if ln.strip()}
    sample_df = sample_df[sample_df[row_id_col].astype(str).str.strip().isin(_keep)]
    print(f"Restricted to {len(sample_df):,} rows listed in {_rows_file}")


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
    "credit balance is too low",
    # A per-day request cap will not clear during the run; rerun tomorrow to fill in
    "requests per day",
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
# LLM_OUTPUT_NAME lets a separate study write its own files (default llm_responses)
_out_name = os.getenv("LLM_OUTPUT_NAME") or "llm_responses"
long_path = os.path.join(out_dir, f"{_out_name}_long.csv")
wide_path = os.path.join(out_dir, f"{_out_name}_wide.csv")

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
