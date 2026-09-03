"""End-to-end checks that run without the real dataset or any API key.

    python -m pytest tests/ -q        # or simply: python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_dataset  # noqa: E402
import generate_llm_responses as gen  # noqa: E402
from tests.make_fixture import make_rows, COLUMNS  # noqa: E402


def _fixture(tmp: str, rows: int = 6000) -> str:
    path = os.path.join(tmp, "raw.csv")
    pd.DataFrame(make_rows(rows, seed=3), columns=COLUMNS).to_csv(path, index=False)
    return path


def test_filter_keeps_only_informative_sub_issues():
    with tempfile.TemporaryDirectory() as tmp:
        raw = build_dataset.load_raw(_fixture(tmp))
        pool, stats = build_dataset.filter_rows(raw)
    assert stats["rows_after_dedup"] == len(pool)
    assert (pool["sub_issue"].str.len() > 0).all()
    assert (pool["sub_issue"].str.lower() != pool["issue"].str.lower()).all()
    assert not pool["sub_issue"].str.lower().isin(build_dataset.GENERIC_SUB_ISSUES).any()
    narr = pool["narrative"].fillna("").astype(str)
    assert not narr[narr.str.len() > 0].duplicated().any()


def test_allocation_covers_every_category_and_hits_target():
    counts = {"a": 500, "b": 120, "c": 40, "d": 3, "e": 1}
    alloc = build_dataset.allocate(counts, target=200, min_per=5)
    assert sum(alloc.values()) == 200
    assert alloc["d"] == 3 and alloc["e"] == 1          # everything available
    assert alloc["c"] >= 5                               # guaranteed minimum
    assert alloc["a"] > alloc["b"] > alloc["c"]          # still reflects prevalence
    assert alloc["a"] < 500 * 200 / sum(counts.values()) * 2  # but dampened

    # Target smaller than the sum of minimums still lands exactly on target.
    alloc = build_dataset.allocate({f"k{i}": 10 for i in range(50)}, target=30, min_per=5)
    assert sum(alloc.values()) == 30


def test_build_dataset_end_to_end_prefers_narratives_and_covers_categories():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = _fixture(tmp, rows=12000)
        out = os.path.join(tmp, "curated.csv")
        rc = build_dataset.main(["--input", raw_path, "--output", out, "--target", "800", "--seed", "1"])
        assert rc == 0
        chosen = pd.read_csv(out, dtype=str, keep_default_na=False)
        summary = json.load(open(os.path.splitext(out)[0] + "_summary.json"))

    assert len(chosen) == 800
    assert chosen["complaint_id"].is_unique
    assert set(build_dataset.OUTPUT_COLUMNS) <= set(chosen.columns)
    # Every category that survived filtering is represented.
    assert summary["selected"]["categories"] == summary["pool"]["categories"]
    # Narrative rows are ranked first within each category.
    assert summary["selected"]["narrative_share"] > 0.5
    qs = chosen["quality_score"].astype(float)
    assert qs.mean() > 3.0


def test_prompt_rendering_includes_context_and_truncates():
    row = pd.Series({"product": "Mortgage", "sub_product": "FHA mortgage",
                     "issue": "Loan servicing", "sub_issue": "Escrow",
                     "narrative": "word " * 2000})
    complaint = gen.build_complaint(row)
    assert "Product: Mortgage" in complaint and "Sub-issue: Escrow" in complaint
    assert "[...]" in complaint and len(complaint) < gen.MAX_NARRATIVE_CHARS + 300

    row["sub_issue"] = "loan servicing"           # repeats the issue -> omitted
    row["narrative"] = ""
    complaint = gen.build_complaint(row)
    assert "Sub-issue" not in complaint and "Customer's description" not in complaint
    for v in gen.PROMPT_VARIANTS:
        assert complaint in gen.render_prompt(v, complaint)


def test_generator_dry_run_resumes_and_retries_errors():
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = _fixture(tmp, rows=4000)
        curated = os.path.join(tmp, "curated.csv")
        build_dataset.main(["--input", raw_path, "--output", curated, "--target", "12"])
        base = [sys.executable, os.path.join(ROOT, "generate_llm_responses.py"), "--dry-run",
                "--rows-file", curated, "--out-dir", tmp, "--base-backoff", "0.001"]

        first = subprocess.run(base + ["--max-attempts", "1"], capture_output=True, text=True, check=True)
        assert "108 to run" in first.stdout
        ckpt = os.path.join(tmp, "llm_responses_raw.jsonl")
        n_first = sum(1 for _ in open(ckpt, encoding="utf-8"))
        assert n_first == 108
        long_df = pd.read_csv(os.path.join(tmp, "llm_responses_long.csv"))
        assert len(long_df) == 108 and long_df["Is_Error"].sum() > 0
        wide = pd.read_csv(os.path.join(tmp, "llm_responses_wide.csv"))
        assert len(wide) == 12 and wide.shape[1] == 4 + 3 + 9

        second = subprocess.run(base, capture_output=True, text=True, check=True)
        assert "108 already done, 0 to run" in second.stdout

        third = subprocess.run(base + ["--retry-errors", "--max-attempts", "5"],
                               capture_output=True, text=True, check=True)
        assert "already done" in third.stdout
        long_df = pd.read_csv(os.path.join(tmp, "llm_responses_long.csv"))
        assert len(long_df) == 108 and long_df["Is_Error"].sum() == 0
        assert (long_df.groupby(["Prompt_Variant", "Model"]).size() == 12).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
