"""Build a high-quality, category-diverse subset of the CFPB complaint database.

Reads the raw CFPB export (dataset/complaints.csv, ~9 GB) in chunks so it never
has to fit in memory, keeps only complaints that carry real context, adds a set
of engineered features, and picks a target number of rows (default 10,000)
spread across as many product / issue / sub-issue categories as possible.

Filters (in order):
  1. Issue and Sub-issue both present, and Sub-issue is not a copy of Issue.
  2. Consumer narrative present, at least MIN_NARRATIVE_CHARS long, and not
     mostly "XXXX" redaction.
  3. Exact-duplicate narratives dropped (the CFPB export contains many).

Selection: every (Product, Sub-product, Issue, Sub-issue) combination gets up to
CAP rows, where CAP is the smallest value that reaches the target. Within a
combination the highest quality_score rows win, so rare categories are always
represented and common ones cannot crowd them out.

Usage:
    python feature_engineering.py                 # defaults
    TARGET_ROWS=10000 SEED=42 python feature_engineering.py

Outputs (next to the input file):
    complaints_10k.csv          the selected rows plus engineered features
    complaints_10k_summary.md   coverage statistics for the selection
"""

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.getenv("CFPB_COMPLAINTS_CSV") or os.path.join(HERE, "dataset", "complaints.csv")
OUT_DIR = os.path.dirname(INPUT)
TARGET_ROWS = int(os.getenv("TARGET_ROWS") or "10000")
SEED = int(os.getenv("SEED") or "42")
CHUNK_ROWS = 250_000

# Quality thresholds
MIN_NARRATIVE_CHARS = 200      # shorter than this rarely describes the problem
MAX_NARRATIVE_CHARS = 6000     # very long narratives are kept but truncated downstream
MAX_REDACTION_RATIO = 0.25     # share of characters inside XXXX-style redactions

# Column names in the CFPB export
C_DATE = "Date received"
C_PRODUCT = "Product"
C_SUBPRODUCT = "Sub-product"
C_ISSUE = "Issue"
C_SUBISSUE = "Sub-issue"
C_NARR = "Consumer complaint narrative"
C_PUBRESP = "Company public response"
C_COMPANY = "Company"
C_STATE = "State"
C_ZIP = "ZIP code"
C_TAGS = "Tags"
C_VIA = "Submitted via"
C_SENT = "Date sent to company"
C_RESP = "Company response to consumer"
C_TIMELY = "Timely response?"
C_ID = "Complaint ID"

USECOLS = [C_DATE, C_PRODUCT, C_SUBPRODUCT, C_ISSUE, C_SUBISSUE, C_NARR, C_PUBRESP,
           C_COMPANY, C_STATE, C_ZIP, C_TAGS, C_VIA, C_SENT, C_RESP, C_TIMELY, C_ID]

REDACTION_RE = re.compile(r"X{2,}")
WORD_RE = re.compile(r"[A-Za-z']+")


def clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply the row filters to one chunk. Returns the surviving rows."""
    for c in (C_ISSUE, C_SUBISSUE, C_NARR, C_PRODUCT, C_SUBPRODUCT):
        chunk[c] = clean_str(chunk[c])

    has_both = (chunk[C_ISSUE] != "") & (chunk[C_SUBISSUE] != "")
    distinct = chunk[C_SUBISSUE].str.lower() != chunk[C_ISSUE].str.lower()
    long_enough = chunk[C_NARR].str.len() >= MIN_NARRATIVE_CHARS
    keep = chunk[has_both & distinct & long_enough].copy()
    if keep.empty:
        return keep

    # Redaction ratio: characters inside XXXX runs / total characters
    redacted_chars = keep[C_NARR].map(lambda t: sum(len(m) for m in REDACTION_RE.findall(t)))
    keep["redaction_ratio"] = (redacted_chars / keep[C_NARR].str.len()).round(4)
    return keep[keep["redaction_ratio"] <= MAX_REDACTION_RATIO]


def load_pool(path: str) -> pd.DataFrame:
    parts = []
    seen_rows = 0
    for chunk in pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=CHUNK_ROWS,
                             low_memory=False):
        seen_rows += len(chunk)
        kept = filter_chunk(chunk)
        parts.append(kept)
        print(f"  read {seen_rows:>10,} rows, kept {sum(len(p) for p in parts):>9,}",
              flush=True)
    pool = pd.concat(parts, ignore_index=True)
    print(f"Filtered pool: {len(pool):,} of {seen_rows:,} rows")
    return pool


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["narrative_chars"] = df[C_NARR].str.len()
    df["narrative_words"] = df[C_NARR].map(lambda t: len(WORD_RE.findall(t)))
    df["issue_words"] = df[C_ISSUE].map(lambda t: len(WORD_RE.findall(t)))
    df["sub_issue_words"] = df[C_SUBISSUE].map(lambda t: len(WORD_RE.findall(t)))
    # How specific the category labels are: longer labels carry more detail
    df["detail_score"] = df["issue_words"] + df["sub_issue_words"]

    received = pd.to_datetime(df[C_DATE], errors="coerce")
    sent = pd.to_datetime(df[C_SENT], errors="coerce")
    df["year"] = received.dt.year
    df["month"] = received.dt.month
    df["days_to_company"] = (sent - received).dt.days

    df["has_tags"] = df[C_TAGS].fillna("").str.strip() != ""
    df["has_public_response"] = df[C_PUBRESP].fillna("").str.strip() != ""
    df["timely_response"] = df[C_TIMELY].fillna("").str.strip().str.lower() == "yes"
    df["response_category"] = df[C_RESP].fillna("").str.strip()
    df["is_disputed_or_relief"] = df["response_category"].str.contains(
        "relief", case=False, regex=False)

    # Quality score used to rank rows inside a category:
    #   - narratives in the 300-2500 char sweet spot score highest
    #   - detailed labels score higher
    #   - redaction is penalised
    length = df["narrative_chars"].clip(upper=MAX_NARRATIVE_CHARS)
    length_score = np.where(length < 300, length / 300,
                            np.where(length <= 2500, 1.0, 1.0 - (length - 2500) / 7000))
    detail_norm = df["detail_score"] / df["detail_score"].max()
    df["quality_score"] = (0.5 * length_score + 0.3 * detail_norm
                           + 0.2 * (1 - df["redaction_ratio"])).round(4)
    return df


def select_diverse(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Take up to CAP rows per category combination, CAP chosen to hit target."""
    strata = [C_PRODUCT, C_SUBPRODUCT, C_ISSUE, C_SUBISSUE]
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_jitter"] = rng.random(len(df))
    df = df.sort_values(["quality_score", "_jitter"], ascending=[False, False])
    df["_rank"] = df.groupby(strata).cumcount() + 1

    counts = df.groupby(strata).size()
    if counts.sum() <= target:
        print(f"Pool ({counts.sum():,}) is smaller than target; keeping everything")
        return df.drop(columns=["_jitter", "_rank"])

    # Smallest cap whose per-stratum take reaches the target
    lo, hi = 1, int(counts.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if np.minimum(counts, mid).sum() >= target:
            hi = mid
        else:
            lo = mid + 1
    cap = lo
    chosen = df[df["_rank"] <= cap]

    # The cap overshoots a little; drop the lowest-quality rows from the
    # strata that hit the cap so no category loses its only representative.
    excess = len(chosen) - target
    if excess > 0:
        at_cap = chosen[chosen["_rank"] == cap]
        drop_idx = at_cap.sort_values("quality_score").index[:excess]
        chosen = chosen.drop(index=drop_idx)
    print(f"Selection: cap={cap} rows per category, {counts.size:,} categories in pool, "
          f"{len(chosen):,} rows chosen")
    return chosen.drop(columns=["_jitter", "_rank"])


def write_summary(pool: pd.DataFrame, chosen: pd.DataFrame, path: str, cap_info: str = ""):
    def nunique(d, cols):
        return d[cols].drop_duplicates().shape[0]

    lines = [
        "# Selection summary", "",
        f"- Filtered pool: {len(pool):,} rows",
        f"- Selected: {len(chosen):,} rows",
        f"- Products: {chosen[C_PRODUCT].nunique()} of {pool[C_PRODUCT].nunique()}",
        f"- Sub-products: {chosen[C_SUBPRODUCT].nunique()} of {pool[C_SUBPRODUCT].nunique()}",
        f"- Issues: {chosen[C_ISSUE].nunique()} of {pool[C_ISSUE].nunique()}",
        f"- Sub-issues: {chosen[C_SUBISSUE].nunique()} of {pool[C_SUBISSUE].nunique()}",
        f"- Issue/Sub-issue pairs: {nunique(chosen, [C_ISSUE, C_SUBISSUE])} "
        f"of {nunique(pool, [C_ISSUE, C_SUBISSUE])}",
        f"- Product/Sub-product/Issue/Sub-issue combos: "
        f"{nunique(chosen, [C_PRODUCT, C_SUBPRODUCT, C_ISSUE, C_SUBISSUE])} "
        f"of {nunique(pool, [C_PRODUCT, C_SUBPRODUCT, C_ISSUE, C_SUBISSUE])}",
        f"- Companies: {chosen[C_COMPANY].nunique()}",
        f"- Years: {int(chosen['year'].min())}-{int(chosen['year'].max())}",
        f"- Narrative length: median {int(chosen['narrative_chars'].median())} chars, "
        f"mean {int(chosen['narrative_chars'].mean())}",
        f"- Mean redaction ratio: {chosen['redaction_ratio'].mean():.3f}",
        "",
        "## Rows per product", "",
        chosen[C_PRODUCT].value_counts().to_frame("rows").to_markdown(),
        "",
        "## Rows per year", "",
        chosen["year"].value_counts().sort_index().to_frame("rows").to_markdown(),
        "",
        "## Top 25 issues", "",
        chosen[C_ISSUE].value_counts().head(25).to_frame("rows").to_markdown(),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    if not os.path.exists(INPUT):
        sys.exit(f"Input not found: {INPUT}")
    print(f"Reading {INPUT}")
    pool = load_pool(INPUT)

    before = len(pool)
    pool = pool.drop_duplicates(subset=[C_NARR], keep="first")
    print(f"Dropped {before - len(pool):,} duplicate narratives -> {len(pool):,} rows")

    pool = add_features(pool)
    chosen = select_diverse(pool, TARGET_ROWS, SEED)
    chosen = chosen.sort_values([C_PRODUCT, C_ISSUE, C_SUBISSUE, C_DATE]).reset_index(drop=True)
    chosen.insert(0, "row_id", range(1, len(chosen) + 1))

    out_csv = os.path.join(OUT_DIR, f"complaints_{TARGET_ROWS // 1000}k.csv")
    out_md = os.path.join(OUT_DIR, f"complaints_{TARGET_ROWS // 1000}k_summary.md")
    chosen.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_summary(pool, chosen, out_md)
    print(f"\nWrote {len(chosen):,} rows -> {out_csv}")
    print(f"Summary -> {out_md}\n")
    with open(out_md, encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
