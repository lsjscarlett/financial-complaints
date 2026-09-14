"""Analyse the returned human rating sheets (see paper/human_rating_protocol.md, Section 6).

Usage
  python3 analysis/human_rating_analysis.py                 # main pass: rater_A.xlsx + rater_B.xlsx
  python3 analysis/human_rating_analysis.py --calibration   # calibration.xlsx returned by both raters
                                                            # as calibration_A.xlsx / calibration_B.xlsx

Outputs analysis/tables/human_agreement.csv, human_vs_judges.csv, human_instruments.csv,
human_cell_means.csv, and prints them.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
HR = os.path.join(HERE, "human_rating")
TAB = os.path.join(HERE, "tables")
SCORES = ["acknowledgement", "concreteness", "tone", "grounding", "overall"]
FLAGS = ["has_placeholder", "promises_outcome", "admits_liability", "would_send"]
JUDGE_FLAGS = ["has_placeholder", "promises_outcome", "admits_liability"]


def read_sheet(path, rater):
    d = pd.read_excel(path, sheet_name="ratings", dtype=str).fillna("")
    for c in SCORES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in FLAGS:
        d[c] = d[c].str.strip().str.lower().map({"yes": 1.0, "no": 0.0})
    d["rater"] = rater
    return d[["item_id", "rater"] + SCORES + FLAGS + ["note"]]


def krippendorff_alpha_ordinal(a, b):
    """Krippendorff's alpha for two raters on an ordinal 1-5 scale (interval distance on the ranks
    of the observed values), computed from the coincidence matrix."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok].astype(int), b[ok].astype(int)
    vals = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1; o[idx[y], idx[x]] += 1
    n = o.sum()
    nc = o.sum(axis=1)
    # ordinal metric: distance between categories = (sum of n_g between them, inclusive, minus half ends)^2
    cum = np.cumsum(nc)
    def delta(i, j):
        if i == j:
            return 0.0
        lo, hi = min(i, j), max(i, j)
        return (cum[hi] - (cum[lo] - nc[lo]) - (nc[lo] + nc[hi]) / 2) ** 2
    D = np.array([[delta(i, j) for j in range(k)] for i in range(k)])
    Do = (o * D).sum() / n
    De = (np.outer(nc, nc) * D).sum() / (n * (n - 1))
    return 1 - Do / De if De else np.nan


def weighted_kappa(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok].astype(int), b[ok].astype(int)
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b, weights="quadratic")


def kappa(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a[ok].astype(int), b[ok].astype(int))


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def agreement_table(A, B, label):
    rows = []
    for c in SCORES:
        a, b = A[c].values, B[c].values
        ok = ~(np.isnan(a) | np.isnan(b))
        rows.append({"set": label, "criterion": c, "n": int(ok.sum()),
                     "krippendorff_alpha_ordinal": krippendorff_alpha_ordinal(a, b),
                     "weighted_kappa": weighted_kappa(a, b),
                     "exact": (a[ok] == b[ok]).mean(), "within_1": (np.abs(a[ok] - b[ok]) <= 1).mean(),
                     "mean_A": np.nanmean(a), "mean_B": np.nanmean(b)})
    for c in FLAGS:
        a, b = A[c].values, B[c].values
        ok = ~(np.isnan(a) | np.isnan(b))
        rows.append({"set": label, "criterion": c, "n": int(ok.sum()), "cohen_kappa": kappa(a, b),
                     "exact": (a[ok] == b[ok]).mean(), "rate_A": np.nanmean(a), "rate_B": np.nanmean(b)})
    return pd.DataFrame(rows)


def main():
    calibration = "--calibration" in sys.argv
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    if calibration:
        A = read_sheet(os.path.join(HR, "calibration_A.xlsx"), "A")
        B = read_sheet(os.path.join(HR, "calibration_B.xlsx"), "B")
        A, B = A.set_index("item_id"), B.set_index("item_id").reindex(A.index)
        agr = agreement_table(A, B, "calibration")
        agr.to_csv(os.path.join(TAB, "human_calibration_agreement.csv"), index=False)
        print("=== Calibration agreement (before discussion) ===")
        print(agr.to_string(index=False))
        both = A.join(B, lsuffix="_A", rsuffix="_B")
        far = both[(both["overall_A"] - both["overall_B"]).abs() >= 2]
        print(f"\nItems to discuss (overall differs by >= 2): {len(far)}")
        print(far[["overall_A", "overall_B"]].to_string())
        return

    A = read_sheet(os.path.join(HR, "rater_A.xlsx"), "A").set_index("item_id")
    B = read_sheet(os.path.join(HR, "rater_B.xlsx"), "B").set_index("item_id").reindex(A.index)
    key = pd.read_csv(os.path.join(HR, "key.csv"), dtype=str).set_index("item_id")
    agr = agreement_table(A, B, "main")
    agr.to_csv(os.path.join(TAB, "human_agreement.csv"), index=False)
    print("=== Inter-rater agreement ===")
    print(agr.to_string(index=False))

    # human mean scores per item
    H = pd.DataFrame({c: (A[c] + B[c]) / 2 for c in SCORES + FLAGS})
    H = H.join(key)

    # LLM judge ratings for the same items
    jr = pd.concat([pd.read_csv(os.path.join(TAB, f), dtype=str, keep_default_na=False)
                    for f in ("judge_ratings.csv", "judge_ratings_study2.csv")], ignore_index=True)
    jr = jr[jr["error"] == ""]
    for c in SCORES:
        jr[c] = pd.to_numeric(jr[c], errors="coerce")
    for c in JUDGE_FLAGS:
        jr[c] = (jr[c].str.lower() == "true").astype(float)
    feats = pd.read_csv(os.path.join(TAB, "reply_features.csv.gz"), dtype={"Row": str}, keep_default_na=False,
                        usecols=["Row", "Model", "Prompt_Variant", "compound", "Response_Chars", "placeholder", "promise_outcome"])
    # study 2 features are not in reply_features; compute compound/length on the fly if missing
    H = H.reset_index().merge(feats, on=["Row", "Model", "Prompt_Variant"], how="left")

    rows = []
    for judge, g in jr.groupby("Judge"):
        g = g.set_index(["Row", "Model", "Prompt_Variant"])
        for c in SCORES:
            j = g[c].reindex(list(zip(H["Row"], H["Model"], H["Prompt_Variant"]))).values
            h = H[c].values
            ok = ~(np.isnan(j) | np.isnan(h))
            rho = stats.spearmanr(h[ok], j[ok])[0] if ok.sum() > 2 else np.nan
            rows.append({"judge": judge, "criterion": c, "n": int(ok.sum()), "spearman_vs_human": rho,
                         "within_1": (np.abs(h[ok] - j[ok]) <= 1).mean(), "human_mean": np.nanmean(h),
                         "judge_mean": np.nanmean(j)})
        for c in JUDGE_FLAGS:
            j = g[c].reindex(list(zip(H["Row"], H["Model"], H["Prompt_Variant"]))).values
            h = (H[c].values >= 0.5).astype(float)
            ok = ~np.isnan(j)
            rows.append({"judge": judge, "criterion": c, "n": int(ok.sum()), "kappa_vs_human": kappa(h[ok], j[ok]),
                         "human_rate": np.nanmean(h), "judge_rate": np.nanmean(j)})
    vj = pd.DataFrame(rows)
    vj.to_csv(os.path.join(TAB, "human_vs_judges.csv"), index=False)
    print("\n=== Human vs LLM judges ===")
    print(vj.to_string(index=False))

    # instruments predicting human overall
    inst = {}
    for judge, g in jr.groupby("Judge"):
        g = g.set_index(["Row", "Model", "Prompt_Variant"])
        inst[f"judge_{judge}"] = g["overall"].reindex(list(zip(H["Row"], H["Model"], H["Prompt_Variant"]))).values
    inst["vader"] = H["compound"].values
    inst["length"] = H["Response_Chars"].values
    rows = []
    for k, v in inst.items():
        v = np.asarray(v, float); ok = ~(np.isnan(v) | np.isnan(H["overall"].values))
        rows.append({"instrument": k, "spearman_vs_human_overall": stats.spearmanr(H["overall"].values[ok], v[ok])[0], "n": int(ok.sum())})
    iv = pd.DataFrame(rows)
    iv.to_csv(os.path.join(TAB, "human_instruments.csv"), index=False)
    print("\n=== Instruments vs human overall ===")
    print(iv.to_string(index=False))

    # cell means under human rating
    cm = H.groupby(["Model", "cell"])[SCORES + FLAGS].agg(["mean"]).round(3)
    cm.columns = [c for c, _ in cm.columns]
    cm["n"] = H.groupby(["Model", "cell"]).size()
    cm.to_csv(os.path.join(TAB, "human_cell_means.csv"))
    print("\n=== Human cell means ===")
    print(cm.to_string())
    gap = H.pivot_table(index=["Row", "cell"], columns="Model", values="overall")
    gap = (gap["ChatGPT"] - gap["Mistral"]).groupby("cell").agg(["mean", "count"])
    print("\n=== Human ChatGPT - Mistral gap on overall, by cell ===")
    print(gap.to_string())


if __name__ == "__main__":
    main()
