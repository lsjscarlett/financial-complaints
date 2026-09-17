"""Within-model sampling variance: how much of a model gap is randomness in a single draw?

Reads dataset/llm_responses_samples_long.csv(.gz): the 299 judge-sample complaints x 3 Study 1
prompts x 2 models x 5 independent draws at the generation temperature. For each feature it
reports, per prompt:

  - the sample-to-sample SD within a (complaint, model) cell, pooled over complaints;
  - the ChatGPT - Mistral gap estimated from cell means over the 5 draws, with a 95% CI from a
    cluster bootstrap over complaints;
  - the same gap re-estimated from single random draws (one per cell, as the main study did),
    repeated 200 times, to show how far a one-draw estimate can wander;
  - a variance decomposition of each feature into complaint, model, model x complaint, and
    draw (residual) shares, within prompt;
  - for binary features, the share of (complaint, model) cells whose 5 draws disagree.

Writes analysis/tables/sampling_variance.csv, sampling_gap_stability.csv, sampling_decomposition.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "dataset")
TAB = os.path.join(HERE, "tables")
sys.path.insert(0, HERE)
import eda_sentiment as eda  # noqa: E402

FEATURES = [("Response_Chars", "Length (chars)"), ("compound", "VADER compound"), ("fk_grade", "FK grade"),
            ("apology", "Apology"), ("ownership", "Ownership"), ("gratitude", "Thanks the customer"),
            ("time_bound", "Time-bound commitment"), ("escalation", "Escalation"),
            ("placeholder", "Leaves a [placeholder]"), ("promise_outcome", "Promises an outcome")]
BINARY = {"apology", "ownership", "gratitude", "time_bound", "escalation", "placeholder", "promise_outcome"}
VLABEL = {"v1_terse": "V1", "v2_empathetic": "V2", "v3_structured": "V3"}
B = 1000
SEED = 7


def load():
    p = os.path.join(DATA, "llm_responses_samples_long.csv")
    if not os.path.exists(p):
        p += ".gz"
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d = d[(d["Is_Error"].str.lower() != "true")].copy()
    for c in ("Response_Chars",):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    comp = pd.read_csv(os.path.join(DATA, "complaints_10k.csv"), dtype=str, keep_default_na=False)
    comp = comp.rename(columns={"row_id": "Row"})
    comp = comp[["Row", "Consumer complaint narrative", "narrative_chars", "redaction_ratio"]].copy()
    for c in ("narrative_chars", "redaction_ratio"):
        comp[c] = pd.to_numeric(comp[c], errors="coerce")
    d, _ = eda.add_features(d, comp)
    for f, _ in FEATURES:
        d[f] = pd.to_numeric(d[f], errors="coerce").astype(float)
    return d


def main():
    d = load()
    rng = np.random.default_rng(SEED)
    print(f"{len(d):,} replies, {d['Row'].nunique()} complaints, draws per cell: "
          f"{d.groupby(['Row', 'Prompt_Variant', 'Model']).size().describe()[['min', 'max']].to_dict()}")
    rows_var, rows_gap, rows_dec = [], [], []
    for v, g in d.groupby("Prompt_Variant"):
        for f, label in FEATURES:
            cells = g.groupby(["Row", "Model"])[f]
            cell_mean = cells.mean().unstack("Model")            # Row x Model, mean over draws
            cell_sd = cells.std(ddof=1).unstack("Model")
            within_sd = float(np.sqrt(np.nanmean(cell_sd.values ** 2)))
            rows_ = cell_mean.dropna()
            gap = rows_["ChatGPT"] - rows_["Mistral"]
            n = len(gap)
            boots = np.array([gap.values[rng.integers(0, n, n)].mean() for _ in range(B)])
            # single-draw re-estimates: pick one draw per cell, difference the cell values, average
            draws = g.groupby(["Row", "Model"])[f].apply(list)
            single = []
            for _ in range(200):
                est = []
                for r in rows_.index:
                    a, b = draws[(r, "ChatGPT")], draws[(r, "Mistral")]
                    est.append(a[rng.integers(len(a))] - b[rng.integers(len(b))])
                single.append(np.mean(est))
            single = np.array(single)
            rec = {"prompt": VLABEL[v], "feature": label, "n_complaints": n,
                   "mean_ChatGPT": rows_["ChatGPT"].mean(), "mean_Mistral": rows_["Mistral"].mean(),
                   "within_cell_sd": within_sd, "gap_5draw_mean": gap.mean(),
                   "gap_ci_lo": np.percentile(boots, 2.5), "gap_ci_hi": np.percentile(boots, 97.5),
                   "gap_d_vs_within": gap.mean() / within_sd if within_sd else np.nan,
                   "single_draw_gap_sd": single.std(ddof=1), "single_draw_gap_min": single.min(),
                   "single_draw_gap_max": single.max(),
                   "single_draw_sign_agree": float(np.mean(np.sign(single) == np.sign(gap.mean())))}
            if f in BINARY:
                flips = cells.agg(lambda s: s.nunique() > 1)
                rec["cells_with_disagreeing_draws"] = float(flips.mean())
            rows_var.append(rec)
            # variance decomposition (balanced: 5 draws per cell)
            y = g[[f, "Row", "Model"]].dropna()
            grand = y[f].mean()
            ss_tot = ((y[f] - grand) ** 2).sum()
            m_row = y.groupby("Row")[f].transform("mean")
            m_mod = y.groupby("Model")[f].transform("mean")
            m_cell = y.groupby(["Row", "Model"])[f].transform("mean")
            ss_row = ((m_row - grand) ** 2).sum()
            ss_mod = ((m_mod - grand) ** 2).sum()
            ss_int = ((m_cell - m_row - m_mod + grand) ** 2).sum()
            ss_res = ((y[f] - m_cell) ** 2).sum()
            rows_dec.append({"prompt": VLABEL[v], "feature": label, "complaint": ss_row / ss_tot,
                             "model": ss_mod / ss_tot, "model_x_complaint": ss_int / ss_tot,
                             "draw": ss_res / ss_tot})
    var = pd.DataFrame(rows_var)
    dec = pd.DataFrame(rows_dec)
    var.to_csv(os.path.join(TAB, "sampling_variance.csv"), index=False)
    dec.to_csv(os.path.join(TAB, "sampling_decomposition.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print("\n=== Model gap (ChatGPT - Mistral) from 5 draws vs single-draw wander ===")
    print(var[["prompt", "feature", "within_cell_sd", "gap_5draw_mean", "gap_ci_lo", "gap_ci_hi",
               "gap_d_vs_within", "single_draw_gap_sd", "single_draw_sign_agree"]].to_string(index=False))
    print("\n=== Binary features: share of (complaint, model) cells whose 5 draws disagree ===")
    print(var.dropna(subset=["cells_with_disagreeing_draws"])[["prompt", "feature", "cells_with_disagreeing_draws"]]
          .pivot(index="feature", columns="prompt", values="cells_with_disagreeing_draws").to_string())
    print("\n=== Variance shares within prompt: complaint / model / model x complaint / draw ===")
    print(dec.to_string(index=False))


if __name__ == "__main__":
    main()
