"""Send every curated complaint through 3 prompt variants x 3 LLMs.

10,000 complaints x 3 prompts x 3 models = 90,000 responses.

Design
------
* Async with a per-provider concurrency limit, so rate limits on one provider
  never stall the others.
* Every response is appended to a JSONL checkpoint as soon as it arrives. Re-run
  the script after a crash, a rate-limit lockout, or a top-up of credits and it
  resumes exactly where it stopped. Use --retry-errors to redo failed calls.
* Retryable failures (429, 5xx, overloaded, timeouts, connection resets) are
  retried with exponential backoff. Fatal failures (bad key, no credits) block
  that provider for the rest of the run so the same error is not paid for
  30,000 times; those rows are simply not written, so the next run retries them.
* --dry-run exercises the whole pipeline (scheduling, checkpointing, resume,
  output files) with canned text and no API calls.

Usage
-----
    python generate_llm_responses.py                 # full run on dataset/curated_complaints.csv
    python generate_llm_responses.py --limit 5       # smoke test: 45 calls
    python generate_llm_responses.py --dry-run --limit 20
    python generate_llm_responses.py --models Claude,Mistral --retry-errors
    python generate_llm_responses.py --finalize-only # rebuild CSVs from the checkpoint

Outputs (in --out-dir, default dataset/):
    llm_responses_raw.jsonl   append-only checkpoint, one JSON object per call
    llm_prompts.csv           one row per complaint x variant, the exact prompt sent
    llm_responses_long.csv    one row per response (9 per complaint)
    llm_responses_wide.csv    one row per complaint, 9 response columns
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
# Real environment variables win over .env.
load_dotenv(os.path.join(HERE, ".env"))

DEFAULT_ROWS = os.path.join(HERE, "dataset", "curated_complaints.csv")
DEFAULT_OUT_DIR = os.path.join(HERE, "dataset")

# Narratives longer than this are cut (with a marker) to keep prompts bounded.
MAX_NARRATIVE_CHARS = int(os.getenv("LLM_MAX_NARRATIVE_CHARS") or "2500")

# Model ids per provider; override via env without touching the code.
MODEL_IDS = {
    "ChatGPT": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
    "Claude": os.getenv("CLAUDE_MODEL") or "claude-sonnet-5",
    "Mistral": os.getenv("MISTRAL_MODEL") or "mistral-small-latest",
}
KEY_ENV = {"ChatGPT": "OPENAI_API_KEY", "Claude": "ANTHROPIC_API_KEY", "Mistral": "MISTRAL_API_KEY"}

# Approximate list prices, USD per 1M tokens (input, output). Edit as needed;
# only used for the cost estimate printed at the end.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "mistral-small-latest": (0.10, 0.30),
}

# Three prompt variants. Each takes a {complaint} placeholder. Only the framing
# changes between them, so differences in the replies are attributable to it.
PROMPT_VARIANTS = [
    {
        # Baseline: minimal instruction, no persona, no tone guidance.
        "name": "v1_terse",
        "max_tokens": 200,
        "template": (
            "Respond to the following consumer complaint in 2 sentences.\n\n"
            "{complaint}"
        ),
    },
    {
        # Adds a persona plus explicit empathy and ownership guidance.
        "name": "v2_empathetic",
        "max_tokens": 400,
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
        "max_tokens": 450,
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
VARIANTS_BY_NAME = {v["name"]: v for v in PROMPT_VARIANTS}

FATAL_MARKERS = (
    "insufficient_quota",
    "credit balance",
    "credit_balance",
    "billing",
    "authentication",
    "invalid_api_key",
    "invalid api key",
    "unauthorized",
    "permission",
)
RETRY_MARKERS = ("rate limit", "rate_limit", "overloaded", "timeout", "timed out",
                 "connection", "temporarily", "try again", "server error", "capacity")


# --------------------------------------------------------------------------- prompts
def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "<na>", "none") else s


def build_complaint(row: pd.Series) -> str:
    """Render the complaint block shared by all three prompt variants."""
    lines = []
    product = _text(row.get("product"))
    sub_product = _text(row.get("sub_product"))
    issue = _text(row.get("issue"))
    sub_issue = _text(row.get("sub_issue"))
    narrative = _text(row.get("narrative"))

    if product:
        lines.append(f"Product: {product}")
    if sub_product:
        lines.append(f"Sub-product: {sub_product}")
    lines.append(f"Issue: {issue}")
    if sub_issue and sub_issue.lower() != issue.lower():
        lines.append(f"Sub-issue: {sub_issue}")
    if narrative:
        if len(narrative) > MAX_NARRATIVE_CHARS:
            narrative = narrative[:MAX_NARRATIVE_CHARS].rsplit(" ", 1)[0] + " [...]"
        lines.append(
            "\nCustomer's description (personal details redacted as XXXX):\n"
            f"{narrative}"
        )
    return "\n".join(lines)


def render_prompt(variant: dict, complaint: str) -> str:
    return variant["template"].format(complaint=complaint)


def job_key(complaint_id: str, variant: str, model: str) -> str:
    return f"{complaint_id}|{variant}|{model}"


# --------------------------------------------------------------------------- providers
@dataclass
class Result:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish: str | None = None


class ProviderBlocked(Exception):
    """Raised once a provider hits a fatal error; stops scheduling for it."""


class Provider:
    name: str

    def __init__(self, name: str, model: str, concurrency: int):
        self.name = name
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.blocked: str | None = None
        self.done = 0
        self.errors = 0

    async def generate(self, prompt: str, max_tokens: int) -> Result:  # pragma: no cover
        raise NotImplementedError

    def classify(self, exc: Exception) -> str:
        """'fatal' (block provider), 'retry', or 'error' (record and move on)."""
        status = getattr(exc, "status_code", None)
        if status is None:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
        msg = str(exc).lower()
        if status in (401, 403) or any(m in msg for m in FATAL_MARKERS):
            return "fatal"
        if status in (408, 409, 425, 429, 500, 502, 503, 504, 529):
            return "retry"
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
            return "retry"
        if any(m in msg for m in RETRY_MARKERS):
            return "retry"
        return "error"


class OpenAIProvider(Provider):
    def __init__(self, model, concurrency):
        super().__init__("ChatGPT", model, concurrency)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(max_retries=0, timeout=90.0)

    async def generate(self, prompt, max_tokens):
        r = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        choice = r.choices[0]
        usage = r.usage
        return Result(
            (choice.message.content or "").strip(),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            choice.finish_reason,
        )


class ClaudeProvider(Provider):
    def __init__(self, model, concurrency):
        super().__init__("Claude", model, concurrency)
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(max_retries=0, timeout=120.0)

    async def generate(self, prompt, max_tokens):
        r = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            # Short customer-service replies: no extended thinking, so the
            # whole max_tokens budget goes to the visible answer.
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        # content can start with a non-text block; collect the text blocks.
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return Result(text.strip(), r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)


class MistralProvider(Provider):
    def __init__(self, model, concurrency):
        super().__init__("Mistral", model, concurrency)
        from mistralai.client import Mistral  # 2.x location of the client

        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async def generate(self, prompt, max_tokens):
        r = await self.client.chat.complete_async(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        choice = r.choices[0]
        content = choice.message.content
        if isinstance(content, list):  # content chunks in newer SDKs
            content = "".join(getattr(c, "text", "") for c in content)
        usage = getattr(r, "usage", None)
        return Result(
            (content or "").strip(),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(choice, "finish_reason", None),
        )


class DryRunProvider(Provider):
    """No network: canned replies, plus occasional fake transient errors."""

    def __init__(self, name, model, concurrency, fail_rate=0.05, seed=0):
        super().__init__(name, model, concurrency)
        self.rng = random.Random(f"{name}-{seed}")
        self.fail_rate = fail_rate

    async def generate(self, prompt, max_tokens):
        await asyncio.sleep(self.rng.uniform(0.001, 0.01))
        if self.rng.random() < self.fail_rate:
            raise RuntimeError("429 rate limit exceeded (simulated)")
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        return Result(f"[{self.name} dry-run {digest}] " + prompt.splitlines()[0][:60],
                      len(prompt) // 4, 40, "end_turn")


def make_providers(models: list[str], args) -> dict[str, Provider]:
    conc = {"ChatGPT": args.concurrency_openai, "Claude": args.concurrency_claude,
            "Mistral": args.concurrency_mistral}
    providers: dict[str, Provider] = {}
    for name in models:
        if args.dry_run:
            providers[name] = DryRunProvider(name, MODEL_IDS[name], conc[name], seed=args.seed)
            continue
        if not os.getenv(KEY_ENV[name]):
            raise SystemExit(f"Missing {KEY_ENV[name]} (needed for {name}). Set it in .env or the environment.")
        cls = {"ChatGPT": OpenAIProvider, "Claude": ClaudeProvider, "Mistral": MistralProvider}[name]
        providers[name] = cls(MODEL_IDS[name], conc[name])
    return providers


# --------------------------------------------------------------------------- checkpoint
def load_checkpoint(path: str) -> dict[str, dict]:
    """Return {key: record}, last record per key wins."""
    records: dict[str, dict] = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial last line after a crash
            records[rec["key"]] = rec
    return records


# --------------------------------------------------------------------------- runner
async def run_job(provider: Provider, job: dict, args, write, progress) -> None:
    prompt, max_tokens = job["prompt"], job["max_tokens"]
    attempt = 0
    started = time.time()
    while True:
        if provider.blocked:
            return  # not recorded, so the next run retries it
        attempt += 1
        try:
            async with provider.sem:
                if provider.blocked:
                    return
                res = await provider.generate(prompt, max_tokens)
            rec = {
                "key": job["key"], "complaint_id": job["complaint_id"],
                "variant": job["variant"], "model": provider.name,
                "model_id": provider.model, "response": res.text,
                "is_error": False, "error_type": None,
                "finish_reason": res.finish,
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                "attempts": attempt, "latency_ms": int((time.time() - started) * 1000),
                "ts": time.time(),
            }
            provider.done += 1
            await write(rec)
            progress.update(1)
            return
        except Exception as exc:  # noqa: BLE001 - every failure must be classified
            kind = provider.classify(exc)
            if kind == "fatal":
                if not provider.blocked:
                    provider.blocked = f"{type(exc).__name__}: {exc}"
                    progress.write(f"\n! {provider.name} disabled for the rest of the run: {exc}\n")
                return
            if kind == "retry" and attempt < args.max_attempts:
                delay = min(args.max_backoff, args.base_backoff * (2 ** (attempt - 1)))
                delay *= random.uniform(0.7, 1.3)
                await asyncio.sleep(delay)
                continue
            rec = {
                "key": job["key"], "complaint_id": job["complaint_id"],
                "variant": job["variant"], "model": provider.name,
                "model_id": provider.model,
                "response": f"{provider.name} Error: {type(exc).__name__}: {exc}",
                "is_error": True, "error_type": type(exc).__name__,
                "finish_reason": None, "input_tokens": None, "output_tokens": None,
                "attempts": attempt, "latency_ms": int((time.time() - started) * 1000),
                "ts": time.time(),
            }
            provider.errors += 1
            await write(rec)
            progress.update(1)
            return


async def run(jobs: list[dict], providers: dict[str, Provider], args, ckpt_path: str) -> None:
    from tqdm import tqdm

    fh = open(ckpt_path, "a", encoding="utf-8")
    lock = asyncio.Lock()

    async def write(rec: dict) -> None:
        async with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    progress = tqdm(total=len(jobs), unit="resp", dynamic_ncols=True)
    try:
        tasks = [run_job(providers[j["model"]], j, args, write, progress) for j in jobs]
        # Bound the number of in-flight coroutines; the semaphores bound the API calls.
        window = 2000
        for i in range(0, len(tasks), window):
            await asyncio.gather(*tasks[i:i + window])
    finally:
        progress.close()
        fh.close()


# --------------------------------------------------------------------------- outputs
def finalize(rows: pd.DataFrame, prompts: pd.DataFrame, ckpt_path: str, out_dir: str,
             models: list[str]) -> None:
    records = load_checkpoint(ckpt_path)
    if not records:
        print("No responses in checkpoint; nothing to finalize.")
        return
    resp = pd.DataFrame(list(records.values()))
    resp["complaint_id"] = resp["complaint_id"].astype(str)

    meta_cols = ["complaint_id", "product", "sub_product", "issue", "sub_issue",
                 "category", "has_narrative", "narrative_chars"]
    meta = rows[[c for c in meta_cols if c in rows.columns]].copy()
    meta["complaint_id"] = meta["complaint_id"].astype(str)

    long_df = resp.merge(meta, on="complaint_id", how="left")
    long_df["response_chars"] = long_df["response"].where(~long_df["is_error"], "").str.len()
    long_df = long_df.rename(columns={
        "complaint_id": "Complaint_ID", "product": "Product", "sub_product": "Sub_Product",
        "issue": "Issue", "sub_issue": "Sub_Issue", "category": "Category",
        "variant": "Prompt_Variant", "model": "Model", "model_id": "Model_ID",
        "response": "Response", "response_chars": "Response_Chars", "is_error": "Is_Error",
        "error_type": "Error_Type", "finish_reason": "Finish_Reason",
        "input_tokens": "Input_Tokens", "output_tokens": "Output_Tokens",
        "attempts": "Attempts", "latency_ms": "Latency_ms",
    })
    order = ["Complaint_ID", "Product", "Sub_Product", "Issue", "Sub_Issue", "Category",
             "has_narrative", "narrative_chars", "Prompt_Variant", "Model", "Model_ID",
             "Response", "Response_Chars", "Is_Error", "Error_Type", "Finish_Reason",
             "Input_Tokens", "Output_Tokens", "Attempts", "Latency_ms"]
    long_df = long_df[[c for c in order if c in long_df.columns]]
    long_df = long_df.sort_values(["Complaint_ID", "Prompt_Variant", "Model"])

    wide = long_df.pivot_table(index="Complaint_ID", columns=["Prompt_Variant", "Model"],
                               values="Response", aggfunc="first")
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    wide = wide.reset_index()
    prompt_wide = prompts.pivot(index="complaint_id", columns="variant", values="prompt_text")
    prompt_wide.columns = [f"prompt__{v}" for v in prompt_wide.columns]
    prompt_wide = prompt_wide.reset_index().rename(columns={"complaint_id": "Complaint_ID"})
    prompt_wide["Complaint_ID"] = prompt_wide["Complaint_ID"].astype(str)
    wide = meta.rename(columns={"complaint_id": "Complaint_ID", "product": "Product",
                                "issue": "Issue", "sub_issue": "Sub_Issue"})[
        ["Complaint_ID", "Product", "Issue", "Sub_Issue"]
    ].merge(prompt_wide, on="Complaint_ID", how="inner").merge(wide, on="Complaint_ID", how="inner")

    long_path = os.path.join(out_dir, "llm_responses_long.csv")
    wide_path = os.path.join(out_dir, "llm_responses_wide.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    errors = int(long_df["Is_Error"].sum())
    print(f"\nCollected {len(long_df):,} responses ({errors:,} errors)")
    print(f"  long (one row per response): {long_path}")
    print(f"  wide (9 responses per row):  {wide_path}")

    ok = long_df[~long_df["Is_Error"]]
    if not ok.empty:
        print("\nResponses per variant x model:")
        print(ok.pivot_table(index="Prompt_Variant", columns="Model", values="Response",
                             aggfunc="count").fillna(0).astype(int).to_string())
        print("\nMean response length in characters:")
        print(ok.pivot_table(index="Prompt_Variant", columns="Model", values="Response_Chars",
                             aggfunc="mean").round(0).to_string())
        usage = ok.groupby("Model_ID")[["Input_Tokens", "Output_Tokens"]].sum(min_count=1)
        if usage.notna().any().any():
            usage["est_cost_usd"] = [
                round((r.Input_Tokens or 0) / 1e6 * PRICES.get(m, (0, 0))[0]
                      + (r.Output_Tokens or 0) / 1e6 * PRICES.get(m, (0, 0))[1], 2)
                for m, r in usage.iterrows()
            ]
            print("\nToken usage and approximate cost (edit PRICES to adjust):")
            print(usage.to_string())
    if errors:
        print("\nErrors by model:")
        print(long_df[long_df["Is_Error"]].groupby(["Model", "Error_Type"]).size().to_string())
        print("Re-run with --retry-errors to retry them.")


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-file", default=os.getenv("CURATED_COMPLAINTS_CSV") or DEFAULT_ROWS,
                    help="curated CSV from build_dataset.py")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=int(os.getenv("LLM_SAMPLE_ROWS") or "0"),
                    help="only the first N complaints (0 = all)")
    ap.add_argument("--models", default="ChatGPT,Claude,Mistral")
    ap.add_argument("--variants", default=",".join(VARIANTS_BY_NAME))
    ap.add_argument("--concurrency-openai", type=int, default=int(os.getenv("LLM_CONCURRENCY_OPENAI") or "8"))
    ap.add_argument("--concurrency-claude", type=int, default=int(os.getenv("LLM_CONCURRENCY_CLAUDE") or "4"))
    ap.add_argument("--concurrency-mistral", type=int, default=int(os.getenv("LLM_CONCURRENCY_MISTRAL") or "2"))
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--base-backoff", type=float, default=2.0)
    ap.add_argument("--max-backoff", type=float, default=60.0)
    ap.add_argument("--retry-errors", action="store_true", help="redo calls recorded as errors")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; canned responses")
    ap.add_argument("--finalize-only", action="store_true", help="rebuild CSVs from the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_IDS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(MODEL_IDS)}")
    variants = [VARIANTS_BY_NAME[v.strip()] for v in args.variants.split(",") if v.strip()]

    if not os.path.exists(args.rows_file):
        raise SystemExit(f"{args.rows_file} not found. Run build_dataset.py first.")
    rows = pd.read_csv(args.rows_file, dtype=str, keep_default_na=False)
    if "complaint_id" not in rows.columns:
        rows["complaint_id"] = [f"row{i}" for i in range(len(rows))]
    rows["complaint_id"] = rows["complaint_id"].astype(str)
    if args.limit:
        rows = rows.head(args.limit)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "llm_responses_raw.jsonl")
    prompts_path = os.path.join(args.out_dir, "llm_prompts.csv")

    # Render every prompt once; save them so analysis can see exactly what was sent.
    prompt_rows = []
    for _, row in rows.iterrows():
        complaint = build_complaint(row)
        for v in variants:
            prompt_rows.append({"complaint_id": row["complaint_id"], "variant": v["name"],
                                "prompt_text": render_prompt(v, complaint),
                                "max_tokens": v["max_tokens"]})
    prompts = pd.DataFrame(prompt_rows)
    prompts.to_csv(prompts_path, index=False, encoding="utf-8-sig")

    if args.finalize_only:
        finalize(rows, prompts, ckpt_path, args.out_dir, models)
        return 0

    done = load_checkpoint(ckpt_path)
    jobs = []
    skipped = 0
    for p in prompt_rows:
        for m in models:
            key = job_key(p["complaint_id"], p["variant"], m)
            prev = done.get(key)
            if prev is not None and (not prev["is_error"] or not args.retry_errors):
                skipped += 1
                continue
            jobs.append({"key": key, "complaint_id": p["complaint_id"], "variant": p["variant"],
                         "model": m, "prompt": p["prompt_text"], "max_tokens": p["max_tokens"]})
    # Interleave providers so all three run from the first second.
    jobs.sort(key=lambda j: (j["complaint_id"], j["variant"], j["model"]))

    total = len(rows) * len(variants) * len(models)
    print(f"--- {len(rows):,} complaints x {len(variants)} prompts x {len(models)} models "
          f"= {total:,} responses; {skipped:,} already done, {len(jobs):,} to run "
          f"{'(DRY RUN)' if args.dry_run else ''} ---")
    print("models: " + ", ".join(f"{m}={MODEL_IDS[m]}" for m in models))

    if jobs:
        providers = make_providers(models, args)
        asyncio.run(run(jobs, providers, args, ckpt_path))
        for p in providers.values():
            status = f"BLOCKED ({p.blocked})" if p.blocked else "ok"
            print(f"  {p.name}: {p.done:,} responses, {p.errors:,} errors, {status}")
        if any(p.blocked for p in providers.values()):
            print("\nSome providers were blocked; fix the key/credits and re-run to resume.")

    finalize(rows, prompts, ckpt_path, args.out_dir, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
