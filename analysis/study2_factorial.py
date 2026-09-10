"""Study 2: factorial analysis of the prompt components, hygiene mitigation, and paraphrase noise.

Cells (all with a 3-4 sentence target; v2_empathetic from Study 1 is cell A00):
  A = empathetic framing (v2's persona + directives)
  B = three-field format (Acknowledgement / Next step / What we need from you)
  C = compliance constraints (no promised outcome, no admitted liability, ...)

Inputs
  dataset/llm_responses_study2_long.csv(.gz)  the seven new factorial cells + hygiene + paraphrases
  dataset/llm_responses_long.csv(.gz)         Study 1, for v2_empathetic (A00) and v3_structured
  dataset/study2_rows.txt                     the 2,999-complaint subset

Outputs (analysis/tables/ and analysis/figures/)
  study2_cell_means.csv        feature means per cell x model
  study2_factorial_effects.csv main effects and interactions of A, B, C per feature and model,
                               from OLS with complaint fixed effects (within-complaint contrasts)
  study2_hygiene.csv           hygiene cell vs v2 on placeholders, salutations, markdown, and tone
  study2_paraphrase.csv        paraphrase cells vs v2: how much does rewording move each feature?
  fig11_factorial.png          cube plot of key features by cell
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "dataset")
TAB = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)
import eda_sentiment as eda  # noqa: E402  (feature functions)

MODELS = ["ChatGPT", "Mistral"]
CELLS = {  # variant name -> (A, B, C)
    "f_000_base": (0, 0, 0), "v2_empathetic": (1, 0, 0), "f_0B0_format": (0, 1, 0),
    "f_00C_constraints": (0, 0, 1), "f_AB0_empathetic_format": (1, 1, 0),
    "f_A0C_empathetic_constraints": (1, 0, 1), "f_0BC_format_constraints": (0, 1, 1),
    "f_ABC_empathetic_format_constraints": (1, 1, 1),
}
CELL_LABEL = {v: "".join(("A" if a else "0", "B" if b else "0", "C" if c else "0")) for v, (a, b, c) in CELLS.items()}
EXTRA = ["h_A00_hygiene", "p_A00_para1", "p_A00_para2", "v3_structured"]
FEATURES = [
    ("Response_Chars", "Length (chars)"), ("sentences", "Sentences"), ("compound", "VADER compound"),
    ("fk_grade", "FK grade"), ("apology", "Apology"), ("ownership", "Ownership"),
    ("time_bound", "Time-bound commitment"), ("escalation", "Escalation"),
    ("asks_info", "Asks for information"), ("gratitude", "Thanks the customer"),
    ("placeholder", "Leaves a [placeholder]"), ("markdown", "Markdown"), ("hedging", "Hedging"),
    ("promise_outcome", "Promises an outcome"), ("v3_all_labels", "All three labels present"),
]
COLOR = {"ChatGPT": "#2a78d6", "Mistral": "#eb6834"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def read_long(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        p += ".gz"
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d = d[(d["Is_Error"].str.lower() != "true") & d["Model"].isin(MODELS)].copy()
    for c in ("Response_Chars", "Prompt_Tokens", "Completion_Tokens", "Latency_s"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load():
    rows = {ln.strip() for ln in open(os.path.join(DATA, "study2_rows.txt")) if ln.strip()}
    s2 = read_long("llm_responses_study2_long.csv")
    s1 = read_long("llm_responses_long.csv")
    s1 = s1[s1["Row"].isin(rows) & s1["Prompt_Variant"].isin(["v2_empathetic", "v3_structured"])]
    df = pd.concat([s2, s1], ignore_index=True)
    df = df[df["Row"].isin(rows)]
    comp = pd.read_csv(os.path.join(DATA, "complaints_10k.csv"), dtype=str, keep_default_na=False)
    comp = comp.rename(columns={"row_id": "Row"})
    comp = comp[["Row", "Consumer complaint narrative", "narrative_chars", "redaction_ratio"]].copy()
    for c in ("narrative_chars", "redaction_ratio"):
        comp[c] = pd.to_numeric(comp[c], errors="coerce")
    df, comp = eda.add_features(df, comp)
    df["cell"] = df["Prompt_Variant"].map(CELL_LABEL)
    for i, f in enumerate("ABC"):
        df[f] = df["Prompt_Variant"].map(lambda v: CELLS.get(v, (np.nan,) * 3)[i])
    return df


def cell_means(df):
    d = df[df["Prompt_Variant"].isin(CELLS)]
    rows = []
    for (m, v), g in d.groupby(["Model", "Prompt_Variant"]):
        r = {"Model": m, "Cell": CELL_LABEL[v], "Variant": v, "n": len(g)}
        for col, label in FEATURES:
            r[label] = g[col].astype(float).mean()
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["Model", "Cell"])
    out.to_csv(os.path.join(TAB, "study2_cell_means.csv"), index=False)
    return out


def factorial_effects(df):
    """Per model and feature: OLS y ~ A*B*C with complaint fixed effects (demeaned within
    complaint), standard errors clustered by complaint. Coefficients are in the feature's
    units (share for binary features)."""
    import statsmodels.formula.api as smf
    d = df[df["Prompt_Variant"].isin(CELLS)].copy()
    # keep only complaints with all 8 cells for the model so the design stays balanced
    rows = []
    for m in MODELS:
        dm = d[d["Model"] == m]
        full = dm.groupby("Row")["Prompt_Variant"].nunique()
        dm = dm[dm["Row"].isin(full[full == 8].index)].copy()
        if dm.empty:
            continue
        for col, label in FEATURES:
            y = dm[col].astype(float)
            dm["_y"] = y - y.groupby(dm["Row"]).transform("mean")  # complaint fixed effects
            try:
                fit = smf.ols("_y ~ A * B * C", data=dm).fit(cov_type="cluster", cov_kwds={"groups": dm["Row"]})
            except Exception as e:
                print("skip", m, label, e)
                continue
            base = dm[dm["cell"] == "000"][col].astype(float).mean()
            for term in ["A", "B", "C", "A:B", "A:C", "B:C", "A:B:C"]:
                rows.append({"Model": m, "Feature": label, "Term": term, "Estimate": fit.params[term],
                             "SE": fit.bse[term], "p": fit.pvalues[term], "Baseline (000) mean": base,
                             "n complaints": dm["Row"].nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "study2_factorial_effects.csv"), index=False)
    return out


def paired_contrast(df, model, variant, reference, cols):
    a = df[(df["Model"] == model) & (df["Prompt_Variant"] == variant)].set_index("Row")
    b = df[(df["Model"] == model) & (df["Prompt_Variant"] == reference)].set_index("Row")
    idx = a.index.intersection(b.index)
    rows = []
    for col, label in cols:
        x, y = a.loc[idx, col].astype(float), b.loc[idx, col].astype(float)
        diff = x - y
        if set(np.unique(np.concatenate([x, y]))) <= {0.0, 1.0}:
            n01 = int(((x == 1) & (y == 0)).sum()); n10 = int(((x == 0) & (y == 1)).sum())
            p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            d = np.nan
        else:
            p = stats.wilcoxon(diff, zero_method="zsplit").pvalue if (diff != 0).any() else 1.0
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0.0
        rows.append({"Model": model, "Variant": variant, "Reference": reference, "Feature": label,
                     "Variant mean": x.mean(), "Reference mean": y.mean(), "Diff": diff.mean(),
                     "Paired d": d, "p": p, "n": len(idx)})
    return rows


def hygiene(df):
    cols = [("placeholder", "Leaves a [placeholder]"), ("markdown", "Markdown"),
            ("gratitude", "Thanks the customer"), ("apology", "Apology"), ("compound", "VADER compound"),
            ("Response_Chars", "Length (chars)"), ("sentences", "Sentences"), ("ownership", "Ownership"),
            ("time_bound", "Time-bound commitment"), ("echo_redaction", "Echoes XXXX")]
    # salutation / sign-off presence
    df["salutation"] = df["Response"].str.contains(r"^\s*(dear|hi|hello)\b", case=False, regex=True)
    df["signoff"] = df["Response"].str.contains(r"\b(sincerely|best regards|kind regards|warm regards|regards,)\b", case=False, regex=True)
    cols = [("salutation", "Salutation"), ("signoff", "Sign-off")] + cols
    rows = []
    for m in MODELS:
        rows += paired_contrast(df, m, "h_A00_hygiene", "v2_empathetic", cols)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "study2_hygiene.csv"), index=False)
    return out


def paraphrase(df):
    rows = []
    for m in MODELS:
        for v in ("p_A00_para1", "p_A00_para2"):
            rows += paired_contrast(df, m, v, "v2_empathetic", FEATURES)
        # and the two paraphrases against each other
        rows += paired_contrast(df, m, "p_A00_para2", "p_A00_para1", FEATURES)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "study2_paraphrase.csv"), index=False)
    return out


def fig_cube(means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.titlecolor": TEXT, "axes.titleweight": "bold",
                         "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    keys = [("Apology", True), ("Time-bound commitment", True), ("Thanks the customer", True),
            ("Leaves a [placeholder]", True), ("Promises an outcome", True), ("VADER compound", False),
            ("Length (chars)", False), ("FK grade", False)]
    order = ["000", "A00", "0B0", "00C", "AB0", "A0C", "0BC", "ABC"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    x = np.arange(len(order))
    for ax, (label, is_share) in zip(axes.flat, keys):
        for i, m in enumerate(MODELS):
            mm = means[means["Model"] == m].set_index("Cell").reindex(order)
            vals = mm[label].values * (100 if is_share else 1)
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.34, color=COLOR[m], label=m)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5)
        ax.set_title(label + (" (%)" if is_share else ""), fontsize=9); ax.grid(axis="x", visible=False)
        if is_share:
            ax.set_ylim(0, 105)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Study 2: reply features across the 2x2x2 prompt factorial (A empathetic framing, B format, C constraints)",
                 fontsize=10, color=TEXT, fontweight="bold", y=1.06)
    p = os.path.join(FIG, "fig11_factorial.png")
    plt.savefig(p); plt.close(); print("  wrote", p)


def main():
    df = load()
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
    print("Replies per model x variant:")
    print(df.groupby(["Model", "Prompt_Variant"]).size().unstack(0).to_string())

    means = cell_means(df)
    print("\n=== Cell means ===")
    print(means.drop(columns=["Variant"]).to_string(index=False))
    fig_cube(means)

    eff = factorial_effects(df)
    if not eff.empty:
        print("\n=== Factorial effects (complaint fixed effects, clustered SE) ===")
        key = eff[eff["Feature"].isin(["Apology", "Time-bound commitment", "Thanks the customer", "Leaves a [placeholder]",
                                       "Promises an outcome", "VADER compound", "Length (chars)", "Asks for information"])]
        print(key.pivot_table(index=["Feature", "Term"], columns="Model", values="Estimate").round(3).to_string())

    hy = hygiene(df)
    if not hy.empty:
        print("\n=== Hygiene cell vs v2 ===")
        print(hy[["Model", "Feature", "Reference mean", "Variant mean", "Diff", "p"]].to_string(index=False))

    pa = paraphrase(df)
    if not pa.empty:
        print("\n=== Paraphrase noise (vs v2 and vs each other) ===")
        print(pa[pa["Feature"].isin(["Length (chars)", "VADER compound", "Thanks the customer", "Time-bound commitment",
                                     "Leaves a [placeholder]", "Escalation"])]
              [["Model", "Variant", "Reference", "Feature", "Diff", "Paired d", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
