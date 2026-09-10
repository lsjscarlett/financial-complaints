"""Rubric-based quality rating of a stratified sample of replies, using two LLM judges.

Each judge sees the complaint (issue, sub-issue, narrative, exactly as the reply
model saw it) and one reply, with no indication of which model or prompt produced
it, and returns JSON scores on a 1-5 rubric plus three boolean flags. Two judges
(gpt-4o-mini and mistral-small-latest) rate every sampled reply so that
inter-judge agreement and judge self-preference can be reported.

Sample: JUDGE_COMPLAINTS complaints (default 300) drawn proportionally by product
family with a fixed seed, all 6 replies each -> 1,800 replies x 2 judges.

Ratings are appended to analysis/tables/judge_ratings.csv as they arrive and the
script resumes from it, like generate_llm_responses.py.

Environment: OPENAI_API_KEY, MISTRAL_API_KEY; JUDGE_COMPLAINTS, JUDGE_CONCURRENCY (4).
"""

import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from mistralai.client import Mistral

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(HERE, "tables")
OUT = os.path.join(TAB, "judge_ratings.csv")

N_COMPLAINTS = int(os.getenv("JUDGE_COMPLAINTS") or "300")
CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY") or "4")
NARRATIVE_CHARS = 1500  # same truncation the reply models saw
SEED = 42

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=90)
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

RUBRIC = """You are auditing draft replies that a financial institution's customer-service team might send to a consumer complaint. Rate the reply below on the complaint below. Be strict and consistent.

Score each criterion from 1 (poor) to 5 (excellent):
- acknowledgement: does the reply correctly and specifically reflect what this customer's complaint is about? (5 = names the actual issue accurately; 1 = generic or wrong)
- concreteness: does the reply give a specific, actionable next step? (5 = specific action, owner, and timeframe; 1 = vague reassurance only)
- tone: is the tone appropriate for a regulated financial institution: professional, respectful, empathetic without being saccharine or defensive? (5 = ideal; 1 = inappropriate)
- grounding: does the reply avoid inventing facts, figures, dates, names, or promises that are not supported by the complaint? (5 = nothing invented; 1 = invents specifics)
- overall: how good is this reply as a first response to send to this customer? (5 = send as is; 1 = unusable)

Also report:
- has_placeholder: true if the reply contains an unfilled template slot such as [Customer's Name] or [date]
- promises_outcome: true if the reply promises a specific outcome (refund, removal, reversal, waiver) rather than a review or investigation
- admits_liability: true if the reply concedes that the institution did something wrong or unlawful

Reply with a single JSON object with exactly these keys: acknowledgement, concreteness, tone, grounding, overall, has_placeholder, promises_outcome, admits_liability, and a one-sentence "rationale". No other text.

COMPLAINT
{complaint}

REPLY
{reply}"""


def build_complaint(row):
    issue, sub = row["Issue"], row["Sub-issue"]
    if sub.lower() == issue.lower():
        sub = ""
    narrative = row["Consumer complaint narrative"]
    if len(narrative) > NARRATIVE_CHARS:
        cut = narrative[:NARRATIVE_CHARS]
        narrative = cut[: cut.rfind(" ")] + " [...]"
    lines = [f"Issue: {issue}"]
    if sub:
        lines.append(f"Sub-issue: {sub}")
    if narrative:
        lines.append(f"Complaint: {narrative}")
    return "\n".join(lines)


def judge_openai(prompt):
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def judge_mistral(prompt):
    r = mistral_client.chat.complete(
        model="mistral-small-latest", temperature=0, max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


JUDGES = {"gpt-4o-mini": judge_openai, "mistral-small": judge_mistral}
KEYS = ["acknowledgement", "concreteness", "tone", "grounding", "overall",
        "has_placeholder", "promises_outcome", "admits_liability", "rationale"]
COLUMNS = ["Row", "Model", "Prompt_Variant", "Judge"] + KEYS + ["raw", "error"]


def parse(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else {}
    rec = {}
    for k in KEYS[:5]:
        try:
            rec[k] = int(round(float(obj.get(k))))
        except Exception:
            rec[k] = ""
    for k in KEYS[5:8]:
        v = obj.get(k)
        rec[k] = "" if v is None else bool(v)
    rec["rationale"] = str(obj.get("rationale", ""))[:300]
    return rec


def call(fn, prompt):
    delay = 2.0
    for attempt in range(6):
        try:
            return fn(prompt), ""
        except Exception as e:
            err = str(e).lower()
            if any(m in err for m in ("insufficient_quota", "credit balance", "invalid_api_key",
                                      "requests per day", "authentication")):
                return "", str(e)
            if attempt == 5:
                return "", str(e)
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)


def main():
    long_path = os.path.join(ROOT, "dataset", "llm_responses_long.csv")
    if not os.path.exists(long_path):
        long_path += ".gz"
    replies = pd.read_csv(long_path, dtype=str, keep_default_na=False)
    replies = replies[(replies["Is_Error"].str.lower() != "true") & replies["Model"].isin(["ChatGPT", "Mistral"])]
    comp = pd.read_csv(os.path.join(ROOT, "dataset", "complaints_10k.csv"), dtype=str, keep_default_na=False)
    comp = comp.rename(columns={"row_id": "Row"})

    # stratified sample by product family (proportional), fixed seed
    def family(p):
        p = p.lower()
        for key, name in (("credit report", "Credit reporting"), ("credit card", "Credit card / prepaid"),
                          ("prepaid", "Credit card / prepaid"), ("payday", "Payday / personal loan"),
                          ("personal loan", "Payday / personal loan"), ("checking", "Checking / savings")):
            if key in p:
                return name
        return p
    comp["family"] = comp["Product"].map(family)
    rng = random.Random(SEED)
    picked = []
    shares = comp["family"].value_counts(normalize=True)
    for fam, share in shares.items():
        rows = sorted(comp[comp["family"] == fam]["Row"])
        rng.shuffle(rows)
        picked += rows[: max(1, round(share * N_COMPLAINTS))]
    picked = set(picked[:N_COMPLAINTS] if len(picked) > N_COMPLAINTS else picked)
    sample = replies[replies["Row"].isin(picked)]
    comp = comp[comp["Row"].isin(picked)].set_index("Row")
    print(f"{len(picked)} complaints, {len(sample):,} replies, {len(JUDGES)} judges")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, dtype=str, keep_default_na=False)
        ok = prev[prev["error"] == ""]
        done = set(zip(ok["Row"], ok["Model"], ok["Prompt_Variant"], ok["Judge"]))
        if len(ok) < len(prev):
            ok.to_csv(OUT, index=False)
        print(f"Resuming: {len(done):,} ratings recorded")

    jobs = []
    for _, r in sample.iterrows():
        complaint = build_complaint(comp.loc[r["Row"]])
        prompt = RUBRIC.format(complaint=complaint, reply=r["Response"])
        for judge in JUDGES:
            if (r["Row"], r["Model"], r["Prompt_Variant"], judge) in done:
                continue
            jobs.append((r["Row"], r["Model"], r["Prompt_Variant"], judge, prompt))
    print(f"{len(jobs):,} ratings to do")

    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    if new:
        w.writeheader()
    lock = threading.Lock()
    sems = {j: threading.Semaphore(CONCURRENCY) for j in JUDGES}
    progress = {"n": 0, "err": 0}
    t0 = time.monotonic()

    def run(job):
        row, model, variant, judge, prompt = job
        with sems[judge]:
            raw, err = call(JUDGES[judge], prompt)
        rec = parse(raw) if raw else {k: "" for k in KEYS}
        rec.update({"Row": row, "Model": model, "Prompt_Variant": variant, "Judge": judge,
                    "raw": raw.replace("\n", " ")[:600], "error": err[:300]})
        with lock:
            w.writerow(rec); f.flush()
            progress["n"] += 1; progress["err"] += bool(err)
            n = progress["n"]
            if n % 100 == 0 or n == len(jobs):
                rate = n / (time.monotonic() - t0)
                print(f"  {n:,}/{len(jobs):,} done, {progress['err']} errors, {rate:.1f}/s, "
                      f"~{(len(jobs) - n) / rate / 60:.0f} min left", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY * len(JUDGES)) as pool:
            for fut in as_completed([pool.submit(run, j) for j in jobs]):
                fut.result()
    finally:
        f.close()

    # summary
    d = pd.read_csv(OUT, dtype=str, keep_default_na=False)
    d = d[d["error"] == ""]
    for k in KEYS[:5]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    for k in KEYS[5:8]:
        d[k] = d[k].str.lower() == "true"
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print("\nMean rubric scores by judge, model, and prompt:")
    print(d.groupby(["Judge", "Prompt_Variant", "Model"])[KEYS[:8]].mean().to_string())


if __name__ == "__main__":
    main()
