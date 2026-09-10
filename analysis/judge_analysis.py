"""Analyse the two-judge rubric ratings written by llm_judge.py.

Outputs (analysis/tables/):
  judge_means.csv          mean rubric scores by judge x prompt x model, with 95% CIs
  judge_agreement.csv      inter-judge Spearman per criterion, Cohen's kappa on the flags,
                           and agreement of the judges' placeholder flag with the regex flag
  judge_self_preference.csv  paired ChatGPT - Mistral difference on each criterion, per judge,
                           and the judge x reply-model interaction (self-preference test)
  judge_prompt_effects.csv paired V2/V3 - V1 differences per judge and model
and analysis/figures/fig10_judge_overall.png
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
MODELS = ["ChatGPT", "Mistral"]
VARIANTS = ["v1_terse", "v2_empathetic", "v3_structured"]
VARIANT_LABEL = {"v1_terse": "V1 terse", "v2_empathetic": "V2 empathetic", "v3_structured": "V3 structured"}
JUDGES = ["gpt-4o-mini", "mistral-small"]
SCORES = ["acknowledgement", "concreteness", "tone", "grounding", "overall"]
FLAGS = ["has_placeholder", "promises_outcome", "admits_liability"]
COLOR = {"ChatGPT": "#2a78d6", "Mistral": "#eb6834"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load():
    d = pd.read_csv(os.path.join(TAB, "judge_ratings.csv"), dtype=str, keep_default_na=False)
    d = d[d["error"] == ""].copy()
    for k in SCORES:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    for k in FLAGS:
        d[k] = d[k].str.lower() == "true"
    d = d.dropna(subset=SCORES)
    feats = pd.read_csv(os.path.join(TAB, "reply_features.csv.gz"), dtype={"Row": str}, keep_default_na=False,
                        usecols=["Row", "Model", "Prompt_Variant", "placeholder", "promise_outcome",
                                 "compound", "Response_Chars"])
    d = d.merge(feats, on=["Row", "Model", "Prompt_Variant"], how="left")
    return d


def ci(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return 1.96 * se


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main():
    d = load()
    print(f"{len(d):,} ratings, {d['Row'].nunique()} complaints, judges: {sorted(d['Judge'].unique())}")

    # means with CIs
    rows = []
    for (j, v, m), g in d.groupby(["Judge", "Prompt_Variant", "Model"]):
        r = {"Judge": j, "Variant": VARIANT_LABEL[v], "Model": m, "n": len(g)}
        for k in SCORES:
            r[k] = g[k].mean(); r[k + "_ci"] = ci(g[k])
        for k in FLAGS:
            r[k] = g[k].mean()
        rows.append(r)
    means = pd.DataFrame(rows)
    means.to_csv(os.path.join(TAB, "judge_means.csv"), index=False)

    # inter-judge agreement, paired on (Row, Model, Variant)
    w = d.pivot_table(index=["Row", "Model", "Prompt_Variant"], columns="Judge", values=SCORES + FLAGS + ["placeholder", "promise_outcome"])
    agree = []
    for k in SCORES:
        a, b = w[(k, JUDGES[0])], w[(k, JUDGES[1])]
        ok = a.notna() & b.notna()
        rho, p = stats.spearmanr(a[ok], b[ok])
        agree.append({"criterion": k, "type": "score", "spearman_rho": rho, "p": p,
                      "exact_agreement": (a[ok] == b[ok]).mean(), "within_1": ((a[ok] - b[ok]).abs() <= 1).mean(),
                      "n": int(ok.sum())})
    for k in FLAGS:
        a, b = w[(k, JUDGES[0])].astype(bool), w[(k, JUDGES[1])].astype(bool)
        agree.append({"criterion": k, "type": "flag", "kappa": kappa(a, b), "exact_agreement": (a == b).mean(),
                      f"rate_{JUDGES[0]}": a.mean(), f"rate_{JUDGES[1]}": b.mean(), "n": len(a)})
    # judges vs regex placeholder flag
    regex = w[("placeholder", JUDGES[0])].astype(bool)
    for j in JUDGES:
        a = w[("has_placeholder", j)].astype(bool)
        agree.append({"criterion": f"has_placeholder vs regex ({j})", "type": "validation", "kappa": kappa(a, regex),
                      "exact_agreement": (a == regex).mean(), "judge_rate": a.mean(), "regex_rate": regex.mean(),
                      "n": len(a)})
    agree = pd.DataFrame(agree)
    agree.to_csv(os.path.join(TAB, "judge_agreement.csv"), index=False)

    # self-preference: per judge, paired ChatGPT - Mistral within (Row, Variant)
    sp = []
    for j in JUDGES:
        dj = d[d["Judge"] == j].pivot_table(index=["Row", "Prompt_Variant"], columns="Model", values=SCORES)
        for k in SCORES:
            diff = (dj[(k, "ChatGPT")] - dj[(k, "Mistral")]).dropna()
            sp.append({"Judge": j, "criterion": k, "ChatGPT - Mistral": diff.mean(), "ci95": ci(diff),
                       "paired_d": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0,
                       "wilcoxon_p": stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0, "n": len(diff)})
    sp = pd.DataFrame(sp)
    # interaction: does the gpt judge favour ChatGPT more than the mistral judge does?
    inter = []
    for k in SCORES:
        a = d[d["Judge"] == JUDGES[0]].pivot_table(index=["Row", "Prompt_Variant"], columns="Model", values=k)
        b = d[d["Judge"] == JUDGES[1]].pivot_table(index=["Row", "Prompt_Variant"], columns="Model", values=k)
        da = (a["ChatGPT"] - a["Mistral"]); db = (b["ChatGPT"] - b["Mistral"])
        both = pd.concat([da, db], axis=1, keys=["gpt", "mis"]).dropna()
        dd = both["gpt"] - both["mis"]
        inter.append({"Judge": "interaction (gpt judge - mistral judge)", "criterion": k,
                      "ChatGPT - Mistral": dd.mean(), "ci95": ci(dd), "paired_d": dd.mean() / dd.std(ddof=1),
                      "wilcoxon_p": stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0, "n": len(dd)})
    sp = pd.concat([sp, pd.DataFrame(inter)])
    sp.to_csv(os.path.join(TAB, "judge_self_preference.csv"), index=False)

    # prompt effects per judge and model
    pe = []
    for j in JUDGES:
        for m in MODELS:
            dm = d[(d["Judge"] == j) & (d["Model"] == m)].pivot_table(index="Row", columns="Prompt_Variant", values=SCORES)
            for v in ("v2_empathetic", "v3_structured"):
                for k in SCORES:
                    diff = (dm[(k, v)] - dm[(k, "v1_terse")]).dropna()
                    pe.append({"Judge": j, "Model": m, "comparison": f"{VARIANT_LABEL[v]} - V1", "criterion": k,
                               "diff": diff.mean(), "ci95": ci(diff), "paired_d": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0,
                               "wilcoxon_p": stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0})
    pe = pd.DataFrame(pe)
    pe.to_csv(os.path.join(TAB, "judge_prompt_effects.csv"), index=False)

    # sentiment vs judged quality
    corr = []
    for j in JUDGES:
        dj = d[d["Judge"] == j]
        for k in ("overall", "tone", "concreteness"):
            corr.append({"Judge": j, "criterion": k,
                         "spearman_vs_vader": stats.spearmanr(dj[k], dj["compound"])[0],
                         "spearman_vs_length": stats.spearmanr(dj[k], dj["Response_Chars"])[0]})
    corr = pd.DataFrame(corr)
    corr.to_csv(os.path.join(TAB, "judge_vs_sentiment.csv"), index=False)

    # figure: overall score by model x variant, one panel per judge
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.titlecolor": TEXT, "axes.titleweight": "bold",
                         "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    fig, axes = plt.subplots(1, len(SCORES), figsize=(2.6 * len(SCORES), 3.4), sharey=True)
    x = np.arange(len(VARIANTS))
    for ax, k in zip(axes, SCORES):
        for j, marker in zip(JUDGES, ("o", "s")):
            for m in MODELS:
                mm = means[(means["Judge"] == j) & (means["Model"] == m)].set_index("Variant").loc[[VARIANT_LABEL[v] for v in VARIANTS]]
                off = -0.08 if m == "ChatGPT" else 0.08
                ax.errorbar(x + off, mm[k], yerr=mm[k + "_ci"], fmt=marker, color=COLOR[m], markersize=5,
                            capsize=2, linewidth=1, markeredgecolor="white", label=f"{m} rated by {j}",
                            linestyle="-" if j == JUDGES[0] else "--", alpha=1 if j == JUDGES[0] else 0.75)
        ax.set_xticks(x); ax.set_xticklabels(["V1", "V2", "V3"]); ax.set_title(k, fontsize=9)
        ax.set_ylim(1, 5); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean score (1-5), 95% CI")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=8)
    fig.suptitle("Rubric ratings by two LLM judges (circles / solid: gpt-4o-mini; squares / dashed: mistral-small)",
                 fontsize=9, color=TEXT2, y=1.0)
    p = os.path.join(FIG, "fig10_judge_scores.png")
    plt.savefig(p); plt.close(); print("  wrote", p)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print("\n=== Means by judge x variant x model ===")
    print(means[["Judge", "Variant", "Model", "n"] + SCORES + FLAGS].to_string(index=False))
    print("\n=== Inter-judge agreement ===")
    print(agree.to_string(index=False))
    print("\n=== Self-preference (ChatGPT - Mistral, paired) ===")
    print(sp.to_string(index=False))
    print("\n=== Prompt effects vs V1 ===")
    print(pe.to_string(index=False))
    print("\n=== Judged quality vs VADER and length ===")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
