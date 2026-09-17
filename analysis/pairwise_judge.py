"""Pairwise judging with order swap: position bias and head-to-head verdicts.

Each judge sees the complaint and two replies to it, labelled A and B, and says which is the
better first response overall, and which is more concrete and which has the better tone. Every
pair is judged twice, once in each order, so a judge that prefers position A (or B) regardless
of content is measured rather than assumed. Pairs are built for the first PAIR_COMPLAINTS
complaints of the stratified judge sample (default 150):

  model pairs   ChatGPT vs Mistral in each of V1, V2/A00, V3, 000, A0C, ABC          (6 per complaint)
  cell pairs    within each model: V2 vs V1, V2 vs V3, A00 vs 000, A00 vs A0C,
                A0C vs ABC, 000 vs 00C                                            (12 per complaint)

That is 18 pairs x 2 orders = 36 calls per complaint per judge. Judges are the same three as
llm_judge.py (JUDGE_JUDGES selects a subset). Verdicts are appended to
analysis/tables/pairwise_judgments.csv and the script resumes from it.

Environment: PAIR_COMPLAINTS (150), PAIR_CONCURRENCY (4), JUDGE_JUDGES, JUDGE_OPENAI_LARGE.
"""

import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import llm_judge as lj                      # judges, clients, build_complaint, call  # noqa: E402
from human_rating_build import judge_sample_order  # noqa: E402

TAB = os.path.join(HERE, "tables")
OUT = os.path.join(TAB, "pairwise_judgments.csv")
N = int(os.getenv("PAIR_COMPLAINTS") or "150")
CONCURRENCY = int(os.getenv("PAIR_CONCURRENCY") or "4")

CELL = {"v1_terse": "V1", "v2_empathetic": "A00", "v3_structured": "V3", "f_000_base": "000",
        "f_00C_constraints": "00C", "f_A0C_empathetic_constraints": "A0C",
        "f_ABC_empathetic_format_constraints": "ABC"}
VARIANT = {v: k for k, v in CELL.items()}
MODEL_PAIR_CELLS = ["V1", "A00", "V3", "000", "A0C", "ABC"]
CELL_PAIRS = [("A00", "V1"), ("A00", "V3"), ("A00", "000"), ("A00", "A0C"), ("A0C", "ABC"), ("000", "00C")]

PROMPT = """You are auditing draft replies that a financial institution's customer-service team might send to a consumer complaint. Below are the complaint and two candidate replies, A and B. Compare them strictly.

Answer three questions, each with "A", "B", or "tie":
- overall: which is the better first response to send to this customer?
- concreteness: which gives the more specific, actionable next step?
- tone: which has the more appropriate tone for a regulated financial institution (professional, respectful, empathetic without being saccharine or defensive)?

Reply with a single JSON object with exactly the keys overall, concreteness, tone, and a one-sentence "rationale". No other text.

COMPLAINT
{complaint}

REPLY A
{a}

REPLY B
{b}"""

KEYS = ["overall", "concreteness", "tone", "rationale"]
COLUMNS = ["Row", "pair_type", "context", "left", "right", "order", "Judge"] + KEYS + ["raw", "error"]


def parse(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else {}
    rec = {}
    for k in KEYS[:3]:
        v = str(obj.get(k, "")).strip().upper()
        rec[k] = v if v in ("A", "B", "TIE") else ""
    rec["rationale"] = str(obj.get("rationale", ""))[:300]
    return rec


def main():
    order, comp = judge_sample_order()
    rows = order[:N]
    frames = []
    for name in ("llm_responses_long.csv", "llm_responses_study2_long.csv"):
        p = os.path.join(ROOT, "dataset", name)
        if not os.path.exists(p):
            p += ".gz"
        frames.append(pd.read_csv(p, dtype=str, keep_default_na=False))
    replies = pd.concat(frames, ignore_index=True)
    replies = replies[(replies["Is_Error"].str.lower() != "true") & replies["Model"].isin(["ChatGPT", "Mistral"])
                      & replies["Prompt_Variant"].isin(CELL)]
    replies = replies.set_index(["Row", "Model", "Prompt_Variant"])["Response"]

    def get(row, model, cell):
        try:
            return replies.loc[(row, model, VARIANT[cell])]
        except KeyError:
            return None

    pairs = []  # (Row, pair_type, context, left_id, right_id, left_text, right_text)
    for r in rows:
        for c in MODEL_PAIR_CELLS:
            a, b = get(r, "ChatGPT", c), get(r, "Mistral", c)
            if a and b:
                pairs.append((r, "model", c, f"ChatGPT|{c}", f"Mistral|{c}", a, b))
        for m in ("ChatGPT", "Mistral"):
            for c1, c2 in CELL_PAIRS:
                a, b = get(r, m, c1), get(r, m, c2)
                if a and b:
                    pairs.append((r, "cell", m, f"{m}|{c1}", f"{m}|{c2}", a, b))
    print(f"{len(rows)} complaints, {len(pairs):,} pairs, judges: {list(lj.JUDGES)}")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, dtype=str, keep_default_na=False)
        ok = prev[prev["error"] == ""]
        done = set(zip(ok["Row"], ok["left"], ok["right"], ok["order"], ok["Judge"]))
        if len(ok) < len(prev):
            ok.to_csv(OUT, index=False)
        print(f"Resuming: {len(done):,} verdicts recorded")

    jobs = []
    for r, ptype, ctx, lid, rid, lt, rt in pairs:
        complaint = lj.build_complaint(comp.loc[r])
        for order_, (x, y) in (("1", (lt, rt)), ("2", (rt, lt))):
            prompt = PROMPT.format(complaint=complaint, a=x, b=y)
            for judge in lj.JUDGES:
                if (r, lid, rid, order_, judge) in done:
                    continue
                jobs.append((r, ptype, ctx, lid, rid, order_, judge, prompt))
    print(f"{len(jobs):,} verdicts to do")
    if not jobs:
        return

    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    if new:
        w.writeheader()
    lock = threading.Lock()
    sems = {j: threading.Semaphore(CONCURRENCY) for j in lj.JUDGES}
    progress = {"n": 0, "err": 0}
    t0 = time.monotonic()

    def run(job):
        r, ptype, ctx, lid, rid, order_, judge, prompt = job
        with sems[judge]:
            raw, err = lj.call(lj.JUDGES[judge], prompt)
        rec = parse(raw) if raw else {k: "" for k in KEYS}
        rec.update({"Row": r, "pair_type": ptype, "context": ctx, "left": lid, "right": rid, "order": order_,
                    "Judge": judge, "raw": raw.replace("\n", " ")[:400], "error": err[:300]})
        with lock:
            w.writerow(rec); f.flush()
            progress["n"] += 1; progress["err"] += bool(err)
            n = progress["n"]
            if n % 200 == 0 or n == len(jobs):
                rate = n / (time.monotonic() - t0)
                print(f"  {n:,}/{len(jobs):,} done, {progress['err']} errors, {rate:.1f}/s, "
                      f"~{(len(jobs) - n) / rate / 60:.0f} min left", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY * len(lj.JUDGES)) as pool:
            for fut in as_completed([pool.submit(run, j) for j in jobs]):
                fut.result()
    finally:
        f.close()


if __name__ == "__main__":
    main()
