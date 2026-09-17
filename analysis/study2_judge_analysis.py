"""Judge ratings over the factorial cells: which cell do the judges prefer, and do the
compliance flags track the factors?

Inputs: analysis/tables/judge_ratings_study2.csv (seven new cells) and judge_ratings.csv
(v2_empathetic = cell A00, from the realistic-prompt pass), restricted to complaints rated
in both. Outputs: study2_judge_means.csv, study2_judge_effects.csv (A, B, C main effects and
two-way interactions on each criterion, per judge and model, complaint fixed effects),
study2_judge_vs_v2.csv (each cell vs A00, paired), and fig12_judge_factorial.png.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
MODELS = ["ChatGPT", "Mistral"]
JUDGES = ["gpt-4o-mini", "mistral-small", "gpt-4.1"]  # trimmed to those present in the data
STYLE = {"gpt-4o-mini": ("o", "-", 1.0), "mistral-small": ("s", "--", 0.75), "gpt-4.1": ("^", ":", 0.75)}
SCORES = ["acknowledgement", "concreteness", "tone", "grounding", "overall"]
FLAGS = ["has_placeholder", "promises_outcome", "admits_liability"]
CELLS = {"f_000_base": "000", "v2_empathetic": "A00", "f_0B0_format": "0B0", "f_00C_constraints": "00C",
         "f_AB0_empathetic_format": "AB0", "f_A0C_empathetic_constraints": "A0C",
         "f_0BC_format_constraints": "0BC", "f_ABC_empathetic_format_constraints": "ABC"}
ORDER = ["000", "A00", "0B0", "00C", "AB0", "A0C", "0BC", "ABC"]
COLOR = {"ChatGPT": "#2a78d6", "Mistral": "#eb6834"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load():
    frames = []
    for name in ("judge_ratings_study2.csv", "judge_ratings.csv"):
        d = pd.read_csv(os.path.join(TAB, name), dtype=str, keep_default_na=False)
        frames.append(d[d["error"] == ""])
    d = pd.concat(frames, ignore_index=True)
    d = d[d["Prompt_Variant"].isin(CELLS)].copy()
    for k in SCORES:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    for k in FLAGS:
        d[k] = (d[k].str.lower() == "true").astype(float)
    d = d.dropna(subset=SCORES)
    d["cell"] = d["Prompt_Variant"].map(CELLS)
    d["A"] = d["cell"].str[0].eq("A").astype(int)
    d["B"] = d["cell"].str[1].eq("B").astype(int)
    d["C"] = d["cell"].str[2].eq("C").astype(int)
    # keep complaints that have all eight cells for a given judge x model
    keep = []
    for (j, m), g in d.groupby(["Judge", "Model"]):
        full = g.groupby("Row")["cell"].nunique()
        keep.append(g[g["Row"].isin(full[full == 8].index)])
    return pd.concat(keep, ignore_index=True)


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main():
    global JUDGES
    d = load()
    JUDGES = [j for j in JUDGES if j in set(d["Judge"])] + sorted(set(d["Judge"]) - set(JUDGES))
    print(f"{len(d):,} ratings; complaints per judge x model:")
    print(d.groupby(["Judge", "Model"])["Row"].nunique().to_string())

    # means
    rows = []
    for (j, m, c), g in d.groupby(["Judge", "Model", "cell"]):
        r = {"Judge": j, "Model": m, "Cell": c, "n": len(g)}
        for k in SCORES + FLAGS:
            r[k] = g[k].mean(); r[k + "_ci"] = ci(g[k])
        rows.append(r)
    means = pd.DataFrame(rows)
    means["Cell"] = pd.Categorical(means["Cell"], ORDER, ordered=True)
    means = means.sort_values(["Judge", "Model", "Cell"])
    means.to_csv(os.path.join(TAB, "study2_judge_means.csv"), index=False)

    # factorial effects with complaint fixed effects
    import statsmodels.formula.api as smf
    eff = []
    for (j, m), g in d.groupby(["Judge", "Model"]):
        g = g.copy()
        for k in SCORES + FLAGS:
            g["_y"] = g[k] - g.groupby("Row")[k].transform("mean")
            fit = smf.ols("_y ~ A * B * C", data=g).fit(cov_type="cluster", cov_kwds={"groups": g["Row"]})
            for term in ["A", "B", "C", "A:B", "A:C", "B:C", "A:B:C"]:
                eff.append({"Judge": j, "Model": m, "Criterion": k, "Term": term, "Estimate": fit.params[term],
                            "SE": fit.bse[term], "p": fit.pvalues[term]})
    eff = pd.DataFrame(eff)
    eff.to_csv(os.path.join(TAB, "study2_judge_effects.csv"), index=False)

    # each cell vs A00 (V2), paired
    vs = []
    for (j, m), g in d.groupby(["Judge", "Model"]):
        w = g.pivot_table(index="Row", columns="cell", values=SCORES + FLAGS)
        for c in ORDER:
            if c == "A00":
                continue
            for k in SCORES + FLAGS:
                diff = (w[(k, c)] - w[(k, "A00")]).dropna()
                p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                vs.append({"Judge": j, "Model": m, "Cell": c, "Criterion": k, "Cell mean": w[(k, c)].mean(),
                           "A00 mean": w[(k, "A00")].mean(), "Diff": diff.mean(), "ci95": ci(diff), "p": p})
    vs = pd.DataFrame(vs)
    vs.to_csv(os.path.join(TAB, "study2_judge_vs_v2.csv"), index=False)

    # figure: overall, concreteness, admits_liability, promises_outcome by cell, per judge
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.titlecolor": TEXT, "axes.titleweight": "bold",
                         "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    panels = [("overall", "Overall (1-5)", False), ("concreteness", "Concreteness (1-5)", False),
              ("acknowledgement", "Acknowledgement (1-5)", False), ("tone", "Tone (1-5)", False),
              ("admits_liability", "Admits liability (%)", True), ("promises_outcome", "Promises outcome (%)", True),
              ("has_placeholder", "Placeholder flagged (%)", True), ("grounding", "Grounding (1-5)", False)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    x = np.arange(len(ORDER))
    for ax, (k, title, is_share) in zip(axes.flat, panels):
        for j in JUDGES:
            marker, ls, alpha = STYLE.get(j, ("d", "-.", 0.75))
            for m in MODELS:
                mm = means[(means["Judge"] == j) & (means["Model"] == m)].set_index("Cell").reindex(ORDER)
                y = mm[k].values * (100 if is_share else 1)
                e = mm[k + "_ci"].values * (100 if is_share else 1)
                off = -0.08 if m == "ChatGPT" else 0.08
                ax.errorbar(x + off, y, yerr=e, fmt=marker, linestyle=ls, color=COLOR[m], markersize=4.5,
                            capsize=2, linewidth=1, markeredgecolor="white", alpha=alpha,
                            label=f"{m}, judged by {j}")
        ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=7.5); ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 105) if is_share else ax.set_ylim(1, 5.1)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.suptitle("Judge ratings across the prompt factorial (circles / solid: gpt-4o-mini; squares / dashed: mistral-small; triangles / dotted: gpt-4.1)",
                 fontsize=9.5, color=TEXT, fontweight="bold", y=1.11)
    p = os.path.join(FIG, "fig12_judge_factorial.png")
    plt.savefig(p); plt.close(); print("  wrote", p)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    print("\n=== Means by judge x model x cell ===")
    print(means[["Judge", "Model", "Cell", "n"] + SCORES + FLAGS].to_string(index=False))
    print("\n=== Main effects (A, B, C) and two-way interactions on judged criteria ===")
    key = eff[eff["Term"].isin(["A", "B", "C", "A:B", "A:C", "B:C"])]
    print(key.pivot_table(index=["Criterion", "Term"], columns=["Judge", "Model"], values="Estimate").round(2).to_string())
    print("\n=== Each cell vs V2 (A00), overall score ===")
    print(vs[vs["Criterion"] == "overall"].pivot_table(index="Cell", columns=["Judge", "Model"], values="Diff").round(2).reindex([c for c in ORDER if c != "A00"]).to_string())


if __name__ == "__main__":
    main()
