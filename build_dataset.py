"""Curate a high-quality, category-diverse subset of the CFPB complaint dataset.

Pipeline
--------
1. Load the raw CSV (Kaggle export or the current CFPB export; column names are
   matched ignoring case and separators).
2. Keep only rows that carry BOTH an issue and a sub-issue, where the sub-issue
   actually adds information (not blank, not "Other", not a repeat of the issue).
3. Engineer features used for ranking and for later analysis (narrative length,
   redaction ratio, sub-issue specificity, category keys, dates, ...).
4. Score every row for "context richness" and select N rows (default 10,000)
   with a coverage-first stratified sampler so that every
   product / issue / sub-issue category is represented, while the remaining
   budget is spread across categories in proportion to sqrt(availability).
5. Write the curated CSV plus a JSON coverage summary.

Usage
-----
    python build_dataset.py
    python build_dataset.py --input dataset/consumer_complaints.csv --target 10000
    python build_dataset.py --min-per-category 3 --seed 7

Environment variable CONSUMER_COMPLAINTS_CSV overrides the default input path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "dataset", "consumer_complaints.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "dataset", "curated_complaints.csv")

# Canonical name -> candidate names in the two known exports.
COLUMN_ALIASES = {
    "date_received": ("date_received", "Date received"),
    "product": ("product", "Product"),
    "sub_product": ("sub_product", "Sub-product"),
    "issue": ("issue", "Issue"),
    "sub_issue": ("sub_issue", "Sub-issue"),
    "narrative": ("consumer_complaint_narrative", "Consumer complaint narrative"),
    "company_public_response": ("company_public_response", "Company public response"),
    "company": ("company", "Company"),
    "state": ("state", "State"),
    "zipcode": ("zipcode", "ZIP code", "zip_code"),
    "tags": ("tags", "Tags"),
    "consumer_consent_provided": (
        "consumer_consent_provided",
        "Consumer consent provided?",
    ),
    "submitted_via": ("submitted_via", "Submitted via"),
    "date_sent_to_company": ("date_sent_to_company", "Date sent to company"),
    "company_response": (
        "company_response_to_consumer",
        "Company response to consumer",
    ),
    "timely_response": ("timely_response", "Timely response?"),
    "consumer_disputed": ("consumer_disputed?", "consumer_disputed", "Consumer disputed?"),
    "complaint_id": ("complaint_id", "Complaint ID"),
}

# Sub-issues that carry no extra information beyond the issue.
GENERIC_SUB_ISSUES = {
    "other",
    "others",
    "n/a",
    "na",
    "none",
    "unknown",
    "not applicable",
    "other (i.e. phone, health club, etc.)",
}

REDACTION_RE = re.compile(r"X{2,}|\bXX/XX/\d{2,4}\b")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names to the actual column names present in df."""
    present = {_norm(c): c for c in df.columns}
    mapping = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            if _norm(cand) in present:
                mapping[canonical] = present[_norm(cand)]
                break
    missing = [c for c in ("issue", "sub_issue", "product") if c not in mapping]
    if missing:
        raise SystemExit(
            f"Required column(s) {missing} not found. Columns present: {list(df.columns)}"
        )
    return mapping


def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints\n"
            "and place consumer_complaints.csv under dataset/, or pass --input."
        )
    df = pd.read_csv(path, low_memory=False, dtype=str, keep_default_na=True)
    mapping = resolve_columns(df)
    df = df.rename(columns={v: k for k, v in mapping.items()})
    keep = [c for c in COLUMN_ALIASES if c in df.columns]
    df = df[keep].copy()
    # Ensure every canonical column exists so downstream code is uniform.
    for canonical in COLUMN_ALIASES:
        if canonical not in df.columns:
            df[canonical] = np.nan
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    return df


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def filter_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep rows where issue and sub-issue are both present and informative."""
    stats = {"input_rows": int(len(df))}

    issue = clean_text(df["issue"])
    sub = clean_text(df["sub_issue"])
    product = clean_text(df["product"])

    has_issue = issue.str.len() > 0
    has_sub = sub.str.len() > 0
    has_product = product.str.len() > 0
    sub_lower = sub.str.lower()
    sub_generic = sub_lower.isin(GENERIC_SUB_ISSUES) | (sub.str.len() < 4)
    sub_repeats_issue = sub_lower == issue.str.lower()

    stats["rows_missing_issue"] = int((~has_issue).sum())
    stats["rows_missing_sub_issue"] = int((~has_sub).sum())
    stats["rows_generic_sub_issue"] = int((has_sub & sub_generic).sum())
    stats["rows_sub_issue_repeats_issue"] = int((has_sub & sub_repeats_issue).sum())

    mask = has_issue & has_sub & has_product & ~sub_generic & ~sub_repeats_issue
    out = df[mask].copy()
    out["issue"] = issue[mask]
    out["sub_issue"] = sub[mask]
    out["product"] = product[mask]
    stats["rows_after_filter"] = int(len(out))

    # Exact duplicate narratives are near-certain re-submissions; keep the first.
    narrative = clean_text(out["narrative"])
    dup = (narrative.str.len() > 0) & narrative.duplicated(keep="first")
    stats["rows_duplicate_narrative"] = int(dup.sum())
    out = out[~dup].copy()
    stats["rows_after_dedup"] = int(len(out))
    return out, stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    narrative = clean_text(df["narrative"])
    df["narrative"] = narrative
    df["has_narrative"] = narrative.str.len() > 0
    df["narrative_chars"] = narrative.str.len().astype(int)
    df["narrative_words"] = narrative.map(lambda s: len(WORD_RE.findall(s)) if s else 0)
    df["narrative_redactions"] = narrative.map(
        lambda s: len(REDACTION_RE.findall(s)) if s else 0
    )
    df["narrative_redaction_ratio"] = np.where(
        df["narrative_words"] > 0,
        df["narrative_redactions"] / df["narrative_words"].clip(lower=1),
        0.0,
    ).round(4)

    df["sub_product"] = clean_text(df["sub_product"])
    df["has_sub_product"] = df["sub_product"].str.len() > 0
    df["issue_words"] = df["issue"].map(lambda s: len(WORD_RE.findall(s)))
    df["sub_issue_words"] = df["sub_issue"].map(lambda s: len(WORD_RE.findall(s)))
    # Sub-issue specificity: how much text the sub-issue adds beyond the issue.
    df["sub_issue_specificity"] = (
        df["sub_issue_words"] + df["issue_words"] * 0.25
    ).round(2)

    df["category"] = df["product"] + " | " + df["issue"] + " | " + df["sub_issue"]
    df["issue_pair"] = df["issue"] + " | " + df["sub_issue"]

    dates = pd.to_datetime(df["date_received"], errors="coerce")
    df["date_received"] = dates.dt.strftime("%Y-%m-%d")
    df["year"] = dates.dt.year.astype("Int64")
    df["month"] = dates.dt.month.astype("Int64")

    df["consumer_disputed_flag"] = (
        clean_text(df["consumer_disputed"]).str.lower() == "yes"
    )
    df["timely_response_flag"] = (
        clean_text(df["timely_response"]).str.lower() == "yes"
    )
    df["company_response"] = clean_text(df["company_response"])
    df["has_company_response"] = df["company_response"].str.len() > 0

    df["quality_score"] = quality_score(df)
    return df


def quality_score(df: pd.DataFrame) -> pd.Series:
    """Higher = more context for an LLM prompt. Roughly 0-10."""
    score = pd.Series(0.0, index=df.index)

    # A consumer narrative is by far the richest context: strong preference.
    score += np.where(df["has_narrative"], 4.0, 0.0)

    # Narrative length in a useful band. Too short is uninformative; very long
    # narratives get truncated in the prompt anyway.
    chars = df["narrative_chars"]
    length_pts = np.select(
        [chars == 0, chars < 200, chars < 500, chars <= 3000, chars <= 6000],
        [0.0, 0.5, 1.5, 2.5, 1.5],
        default=1.0,
    )
    score += length_pts

    # Heavy redaction ("XXXX ... XXXX") drains information from a narrative.
    score -= (df["narrative_redaction_ratio"].clip(upper=0.5) * 3.0)

    # More specific sub-issue wording = more context.
    score += df["sub_issue_specificity"].clip(upper=8) * 0.25

    score += np.where(df["has_sub_product"], 0.5, 0.0)
    score += np.where(df["has_company_response"], 0.25, 0.0)
    return score.round(3)


def allocate(counts: dict[str, int], target: int, min_per: int) -> dict[str, int]:
    """Coverage-first allocation across categories.

    Phase 1 gives every category min(min_per, available).
    Phase 2 spreads the remaining budget proportionally to sqrt(available),
    which keeps frequent categories prominent without letting them swamp the
    long tail. Water-filling respects each category's availability.
    """
    alloc = {k: min(min_per, n) for k, n in counts.items()}
    remaining = target - sum(alloc.values())
    if remaining <= 0:
        # Even the minimum overshoots: trim proportionally, largest first.
        while sum(alloc.values()) > target:
            k = max(alloc, key=alloc.get)
            alloc[k] -= 1
        return alloc

    open_cats = {k for k, n in counts.items() if n > alloc[k]}
    while remaining > 0 and open_cats:
        weights = {k: math.sqrt(counts[k] - alloc[k]) for k in open_cats}
        total_w = sum(weights.values())
        gave = 0
        for k in sorted(open_cats, key=lambda c: -weights[c]):
            share = max(1, int(round(remaining * weights[k] / total_w)))
            room = counts[k] - alloc[k]
            add = min(share, room, remaining - gave)
            alloc[k] += add
            gave += add
            if alloc[k] >= counts[k]:
                open_cats.discard(k)
            if gave >= remaining:
                break
        remaining -= gave
        if gave == 0:
            break
    return alloc


def select_rows(df: pd.DataFrame, target: int, min_per: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_tiebreak"] = rng.random(len(df))
    df = df.sort_values(["category", "quality_score", "_tiebreak"], ascending=[True, False, False])

    counts = df["category"].value_counts().to_dict()
    alloc = allocate(counts, target, min_per)

    picked = []
    for cat, n in alloc.items():
        if n <= 0:
            continue
        picked.append(df[df["category"] == cat].head(n))
    out = pd.concat(picked) if picked else df.head(0)
    out = out.drop(columns=["_tiebreak"])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def summarize(raw_stats: dict, pool: pd.DataFrame, chosen: pd.DataFrame) -> dict:
    def top(series: pd.Series, n: int = 15) -> list[dict]:
        return [{"value": k, "count": int(v)} for k, v in series.value_counts().head(n).items()]

    return {
        "filtering": raw_stats,
        "pool": {
            "rows": int(len(pool)),
            "rows_with_narrative": int(pool["has_narrative"].sum()),
            "categories": int(pool["category"].nunique()),
            "products": int(pool["product"].nunique()),
            "issues": int(pool["issue"].nunique()),
            "sub_issues": int(pool["sub_issue"].nunique()),
        },
        "selected": {
            "rows": int(len(chosen)),
            "rows_with_narrative": int(chosen["has_narrative"].sum()),
            "narrative_share": round(float(chosen["has_narrative"].mean()), 4) if len(chosen) else 0,
            "categories": int(chosen["category"].nunique()),
            "products": int(chosen["product"].nunique()),
            "issues": int(chosen["issue"].nunique()),
            "sub_issues": int(chosen["sub_issue"].nunique()),
            "issue_sub_issue_pairs": int(chosen["issue_pair"].nunique()),
            "unique_prompts_possible": int(
                (chosen["category"] + "||" + chosen["narrative"]).nunique()
            ),
            "mean_quality_score": round(float(chosen["quality_score"].mean()), 3) if len(chosen) else 0,
            "mean_narrative_chars": round(float(chosen["narrative_chars"].mean()), 1) if len(chosen) else 0,
            "year_range": [
                int(chosen["year"].min()) if chosen["year"].notna().any() else None,
                int(chosen["year"].max()) if chosen["year"].notna().any() else None,
            ],
            "by_product": top(chosen["product"], 25),
            "top_issues": top(chosen["issue"], 25),
            "top_sub_issues": top(chosen["sub_issue"], 25),
            "smallest_categories": [
                {"value": k, "count": int(v)}
                for k, v in chosen["category"].value_counts().tail(10).items()
            ],
        },
    }


OUTPUT_COLUMNS = [
    "complaint_id",
    "date_received",
    "year",
    "month",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "category",
    "issue_pair",
    "narrative",
    "has_narrative",
    "narrative_chars",
    "narrative_words",
    "narrative_redactions",
    "narrative_redaction_ratio",
    "issue_words",
    "sub_issue_words",
    "sub_issue_specificity",
    "company",
    "company_public_response",
    "company_response",
    "has_company_response",
    "state",
    "zipcode",
    "tags",
    "consumer_consent_provided",
    "submitted_via",
    "timely_response_flag",
    "consumer_disputed_flag",
    "quality_score",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.getenv("CONSUMER_COMPLAINTS_CSV") or DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--target", type=int, default=10_000, help="rows to select")
    ap.add_argument(
        "--min-per-category",
        type=int,
        default=5,
        help="guaranteed rows per product|issue|sub-issue category (if available)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print(f"Loading {args.input} ...")
    raw = load_raw(args.input)
    print(f"  {len(raw):,} rows, {len(raw.columns)} columns")

    pool, stats = filter_rows(raw)
    print(
        f"Filtered to {len(pool):,} rows with informative issue + sub-issue "
        f"(dropped {stats['input_rows'] - stats['rows_after_dedup']:,})"
    )

    pool = engineer_features(pool)
    print(
        f"  pool: {pool['category'].nunique():,} categories, "
        f"{int(pool['has_narrative'].sum()):,} rows with a narrative"
    )

    chosen = select_rows(pool, args.target, args.min_per_category, args.seed)
    if len(chosen) < args.target:
        print(
            f"WARNING: only {len(chosen):,} rows available after filtering "
            f"(target {args.target:,})",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    chosen = chosen.reindex(columns=OUTPUT_COLUMNS)
    if chosen["complaint_id"].isna().any():
        # Fall back to a stable synthetic id so downstream joins always work.
        chosen["complaint_id"] = chosen["complaint_id"].fillna(
            pd.Series([f"row{i}" for i in range(len(chosen))], index=chosen.index)
        )
    chosen.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = summarize(stats, pool, chosen)
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    sel = summary["selected"]
    print(
        f"\nSelected {sel['rows']:,} rows -> {args.output}\n"
        f"  categories (product|issue|sub-issue): {sel['categories']:,}\n"
        f"  products: {sel['products']}  issues: {sel['issues']}  sub-issues: {sel['sub_issues']}\n"
        f"  rows with narrative: {sel['rows_with_narrative']:,} ({sel['narrative_share']:.0%})\n"
        f"  distinct complaint texts: {sel['unique_prompts_possible']:,}\n"
        f"  mean quality score: {sel['mean_quality_score']}\n"
        f"  summary: {summary_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
