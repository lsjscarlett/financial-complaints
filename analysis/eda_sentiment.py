"""EDA and sentiment analysis of the LLM replies to CFPB complaints.

Reads dataset/llm_responses_long.csv (or the .gz) and dataset/complaints_10k.csv,
computes per-reply features, compares models and prompt variants with paired
tests, and writes figures + tables under analysis/.

Features per reply
  length          Response_Chars, words, sentences
  sentiment       VADER compound / pos / neg / neu (reply and complaint narrative)
  readability     Flesch reading ease, Flesch-Kincaid grade (textstat)
  markers         regex flags: apology, empathy, ownership, time-bound commitment,
                  hedging, asks for information, escalation, promises of outcome,
                  legal advice, markdown, echoes of the XXXX redaction token
  compliance      v1: exactly 2 sentences; v2: 3-4 sentences; v3: all three labels

Usage
  python analysis/eda_sentiment.py            # full run
  ANALYSIS_SAMPLE=2000 python analysis/...    # quick pass on a random subset of complaints
  ANALYSIS_FROM_CACHE=1 python analysis/...   # redo stats + figures from tables/reply_features.csv.gz
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "dataset")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

MODELS = ["ChatGPT", "Mistral"]
VARIANTS = ["v1_terse", "v2_empathetic", "v3_structured"]
VARIANT_LABEL = {"v1_terse": "V1 terse", "v2_empathetic": "V2 empathetic",
                 "v3_structured": "V3 structured"}
# Categorical palette (validated, see dataviz reference): slot 1 blue, slot 2 orange
COLOR = {"ChatGPT": "#2a78d6", "Mistral": "#eb6834"}
# Ordinal ramp for the three variants (blue, light -> dark)
VARIANT_COLOR = {"v1_terse": "#86b6ef", "v2_empathetic": "#2a78d6", "v3_structured": "#104281"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# --------------------------------------------------------------------------- load
def load():
    long_path = os.path.join(DATA, "llm_responses_long.csv")
    if not os.path.exists(long_path):
        long_path += ".gz"
    df = pd.read_csv(long_path, dtype=str, keep_default_na=False)
    df = df[df["Is_Error"].str.lower() != "true"].copy()
    df = df[df["Model"].isin(MODELS)]
    for c in ("Response_Chars", "Prompt_Tokens", "Completion_Tokens", "Latency_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    comp = pd.read_csv(os.path.join(DATA, "complaints_10k.csv"), dtype=str, keep_default_na=False)
    comp = comp.rename(columns={"row_id": "Row"})
    keep = ["Row", "Consumer complaint narrative", "narrative_chars", "narrative_words",
            "redaction_ratio", "detail_score", "quality_score", "year", "days_to_company",
            "timely_response", "response_category", "is_disputed_or_relief", "State",
            "Company", "Tags"]
    keep = [c for c in keep if c in comp.columns]
    comp = comp[keep].copy()
    for c in ("narrative_chars", "narrative_words", "redaction_ratio", "detail_score",
              "quality_score", "year", "days_to_company"):
        if c in comp:
            comp[c] = pd.to_numeric(comp[c], errors="coerce")

    sample = os.getenv("ANALYSIS_SAMPLE")
    if sample:
        rows = comp["Row"].sample(int(sample), random_state=42)
        comp = comp[comp["Row"].isin(rows)]
        df = df[df["Row"].isin(rows)]
    return df, comp


# ----------------------------------------------------------------------- features
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[*])")

MARKERS = {
    "apology": r"\b(sorry|apologi[sz]e|apologies|regret)\b",
    "empathy": r"\b(understand|frustrat\w*|appreciate|stressful|upsetting|concern\w*|difficult)\b",
    "ownership": r"\b(we will|we'll|i will|i'll|i am going to|we are going to|i have escalated|"
                 r"i've escalated|our team will|i will personally|i'll personally)\b",
    "time_bound": r"\b(within \d+|within (one|two|three|five|seven|ten|[a-z]+) (business |working )?"
                  r"(days?|hours?|weeks?)|by (the end of|end of) (the )?(day|week|month)|"
                  r"in the next \d+|\d+[- ]business[- ]days?|\d+[- ]days?|24 hours|48 hours)\b",
    "hedging": r"\b(may|might|could|possibly|cannot guarantee|can't guarantee|no guarantee|"
               r"subject to|if applicable|where appropriate)\b",
    "asks_info": r"\b(please (provide|send|share|reply|contact|call|submit|reach out)|"
                 r"we need (you to|the following)|send us|provide us|get in touch|contact (us|our)|"
                 r"call (us|our))\b",
    "escalation": r"\b(escalat\w*|supervisor|manager|specialist|dedicated team|resolution team|"
                  r"complaints? department|executive)\b",
    "promise_outcome": r"\b(will be (refunded|reimbursed|removed|corrected|reversed|credited|waived|"
                       r"deleted|closed))|(full|immediate) refund|we will (refund|reimburse|remove|"
                       r"reverse|waive|correct|delete)\b",
    "external_body": r"\b(cfpb|consumer financial protection bureau|attorney|lawyer|legal action|"
                     r"credit bureau|equifax|experian|transunion|ftc|regulator)\b",
    "markdown": r"(\*\*|^\s*[-*] |^\s*\d+\.\s)",
    "echo_redaction": r"XXXX",
    "gratitude": r"\b(thank you|thanks|appreciate you)\b",
    # unfilled template slots such as "[Your Name]" or "[specific date, e.g., Friday]"
    "placeholder": r"\[[^\]\n]{2,60}\]",
}
MARKER_LABEL = {
    "apology": "Apology", "empathy": "Empathy / acknowledgement", "ownership": "Ownership",
    "time_bound": "Time-bound commitment", "hedging": "Hedging", "asks_info": "Asks for information",
    "escalation": "Escalation", "promise_outcome": "Promises an outcome",
    "external_body": "Mentions regulator / bureau", "markdown": "Markdown formatting",
    "echo_redaction": "Echoes XXXX redaction", "gratitude": "Thanks the customer",
    "placeholder": "Leaves a [placeholder]",
}
V3_LABELS = [r"acknowledg?ement\s*:", r"next step\s*:", r"what we need from you\s*:"]
MAX_TOKENS = {"v1_terse": 150, "v2_empathetic": 300, "v3_structured": 350}  # from generate_llm_responses.py


def add_features(df, comp):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import textstat
    sia = SentimentIntensityAnalyzer()

    # Normalise typographic quotes so "I’ll" matches the same patterns as "I'll"
    text = (df["Response"].astype(str)
            .str.replace("\u2019", "'", regex=False).str.replace("\u2018", "'", regex=False)
            .str.replace("\u201c", '"', regex=False).str.replace("\u201d", '"', regex=False))
    df["words"] = text.str.split().str.len()
    df["sentences"] = text.apply(lambda t: len([s for s in SENT_SPLIT.split(t.strip()) if s.strip()]))

    vs = pd.DataFrame([sia.polarity_scores(t) for t in text], index=df.index)
    df["compound"] = vs["compound"]
    df["vader_pos"] = vs["pos"]
    df["vader_neg"] = vs["neg"]
    df["vader_neu"] = vs["neu"]

    df["flesch_ease"] = text.apply(textstat.flesch_reading_ease)
    df["fk_grade"] = text.apply(textstat.flesch_kincaid_grade)

    for k, pat in MARKERS.items():
        flags = re.I | re.M
        df[k] = text.str.contains(pat, flags=flags, regex=True)
    df["v3_all_labels"] = np.all([text.str.contains(p, flags=re.I, regex=True) for p in V3_LABELS], axis=0)
    df["v3_nothing_needed"] = text.str.contains(r"nothing at this time", flags=re.I, regex=True)
    df["sentence_compliant"] = np.select(
        [df["Prompt_Variant"] == "v1_terse", df["Prompt_Variant"] == "v2_empathetic",
         df["Prompt_Variant"] == "v3_structured"],
        [df["sentences"] == 2, df["sentences"].between(3, 4), df["v3_all_labels"]], False)
    cap = df["Prompt_Variant"].map(MAX_TOKENS)
    df["truncated"] = df["Completion_Tokens"] >= cap

    # complaint-side sentiment
    narr = comp["Consumer complaint narrative"].astype(str)
    cvs = pd.DataFrame([sia.polarity_scores(t) for t in narr], index=comp.index)
    comp["narr_compound"] = cvs["compound"]
    comp["narr_neg"] = cvs["neg"]
    df = df.merge(comp.drop(columns=["Consumer complaint narrative"]), on="Row", how="left")
    df["model_variant"] = df["Model"] + " / " + df["Prompt_Variant"].map(VARIANT_LABEL)
    df["Product_family"] = df["Product"].map(product_family)
    return df, comp


def product_family(p):
    """Collapse the CFPB's renamed product categories into stable families."""
    p = p.lower()
    if "credit report" in p:
        return "Credit reporting"
    if "credit card" in p or "prepaid" in p:
        return "Credit card / prepaid card"
    if "payday" in p or "personal loan" in p or "consumer loan" in p:
        return "Payday / personal loan"
    if "checking" in p or "bank account" in p:
        return "Checking / savings account"
    if "money transfer" in p or "virtual currency" in p:
        return "Money transfer / virtual currency"
    return p[0].upper() + p[1:]


# --------------------------------------------------------------------- statistics
def cohen_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean() / d.std(ddof=1) if d.std(ddof=1) else 0.0


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - 1.96 * se, x.mean() + 1.96 * se)


METRICS = [
    ("Response_Chars", "Length (chars)"), ("words", "Length (words)"), ("sentences", "Sentences"),
    ("compound", "VADER compound"), ("vader_pos", "VADER positive share"),
    ("vader_neg", "VADER negative share"), ("flesch_ease", "Flesch reading ease"),
    ("fk_grade", "Flesch-Kincaid grade"), ("Latency_s", "Latency (s)"),
    ("Completion_Tokens", "Completion tokens"),
]
FLAGS = list(MARKERS) + ["v3_all_labels", "v3_nothing_needed", "sentence_compliant", "truncated"]


def summary_tables(df):
    # means with CI per model x variant
    rows = []
    for (m, v), g in df.groupby(["Model", "Prompt_Variant"]):
        r = {"Model": m, "Variant": VARIANT_LABEL[v], "n": len(g)}
        for col, label in METRICS:
            lo, hi = ci95(g[col])
            r[label] = g[col].mean()
            r[label + " CI"] = f"[{lo:.3g}, {hi:.3g}]"
        for f in FLAGS:
            r[MARKER_LABEL.get(f, f)] = g[f].mean()
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(TAB, "summary_by_model_variant.csv"), index=False)

    # paired model comparison within each variant (same complaint, same prompt)
    rows = []
    for v in VARIANTS:
        w = df[df["Prompt_Variant"] == v].pivot_table(index="Row", columns="Model",
                                                       values=[c for c, _ in METRICS] + FLAGS,
                                                       aggfunc="first")
        for col, label in METRICS + [(f, MARKER_LABEL.get(f, f)) for f in FLAGS]:
            a, b = w[(col, "ChatGPT")].astype(float), w[(col, "Mistral")].astype(float)
            ok = a.notna() & b.notna()
            a, b = a[ok], b[ok]
            if col in FLAGS:
                # McNemar for paired binary flags
                n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
                p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                test = "McNemar (exact)"
                d = np.nan
            else:
                p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue if (a != b).any() else 1.0
                test = "Wilcoxon signed-rank"
                d = cohen_d_paired(a, b)
            rows.append({"Variant": VARIANT_LABEL[v], "Metric": label, "n pairs": int(ok.sum()),
                         "ChatGPT mean": a.mean(), "Mistral mean": b.mean(),
                         "Diff (ChatGPT - Mistral)": a.mean() - b.mean(), "Cohen d (paired)": d,
                         "Test": test, "p": p})
    mc = pd.DataFrame(rows)
    mc.to_csv(os.path.join(TAB, "model_comparison_paired.csv"), index=False)

    # paired variant comparison within each model (v2 vs v1, v3 vs v1)
    rows = []
    for m in MODELS:
        w = df[df["Model"] == m].pivot_table(index="Row", columns="Prompt_Variant",
                                             values=[c for c, _ in METRICS] + FLAGS, aggfunc="first")
        for v in ("v2_empathetic", "v3_structured"):
            for col, label in METRICS + [(f, MARKER_LABEL.get(f, f)) for f in FLAGS]:
                a, b = w[(col, v)].astype(float), w[(col, "v1_terse")].astype(float)
                ok = a.notna() & b.notna()
                a, b = a[ok], b[ok]
                if col in FLAGS:
                    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
                    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                    d = np.nan
                else:
                    p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue if (a != b).any() else 1.0
                    d = cohen_d_paired(a, b)
                rows.append({"Model": m, "Comparison": f"{VARIANT_LABEL[v]} vs V1 terse",
                             "Metric": label, "n pairs": int(ok.sum()), "Variant mean": a.mean(),
                             "V1 mean": b.mean(), "Diff": a.mean() - b.mean(),
                             "Cohen d (paired)": d, "p": p})
    vc = pd.DataFrame(rows)
    vc.to_csv(os.path.join(TAB, "variant_comparison_paired.csv"), index=False)
    return summ, mc, vc


def complaint_context_tables(df):
    """How reply sentiment tracks the complaint: narrative tone, product, outcome."""
    out = {}
    # correlation reply sentiment vs narrative sentiment
    rows = []
    for (m, v), g in df.groupby(["Model", "Prompt_Variant"]):
        r, p = stats.spearmanr(g["compound"], g["narr_compound"])
        r2, p2 = stats.spearmanr(g["Response_Chars"], g["narrative_chars"])
        rows.append({"Model": m, "Variant": VARIANT_LABEL[v],
                     "Spearman rho: reply vs narrative sentiment": r, "p": p,
                     "Spearman rho: reply length vs narrative length": r2, "p ": p2})
    out["mirroring"] = pd.DataFrame(rows)
    out["mirroring"].to_csv(os.path.join(TAB, "sentiment_mirroring.csv"), index=False)

    # by narrative sentiment quintile
    df["narr_quintile"] = pd.qcut(df["narr_compound"], 5, labels=["Q1 most negative", "Q2", "Q3", "Q4", "Q5 least negative"])
    q = df.groupby(["narr_quintile", "Model"], observed=True).agg(
        reply_compound=("compound", "mean"), apology=("apology", "mean"), empathy=("empathy", "mean"),
        escalation=("escalation", "mean"), n=("compound", "size")).reset_index()
    q.to_csv(os.path.join(TAB, "reply_by_narrative_sentiment_quintile.csv"), index=False)
    out["quintile"] = q

    # by product
    prod = df.groupby(["Product_family", "Model"]).agg(
        reply_compound=("compound", "mean"), chars=("Response_Chars", "mean"),
        apology=("apology", "mean"), time_bound=("time_bound", "mean"),
        promise_outcome=("promise_outcome", "mean"), n=("compound", "size")).reset_index()
    prod.to_csv(os.path.join(TAB, "reply_by_product.csv"), index=False)
    out["product"] = prod

    # by company outcome
    if "response_category" in df:
        oc = df.groupby(["response_category", "Model"]).agg(
            reply_compound=("compound", "mean"), promise_outcome=("promise_outcome", "mean"),
            apology=("apology", "mean"), n=("compound", "size")).reset_index()
        oc.to_csv(os.path.join(TAB, "reply_by_company_outcome.csv"), index=False)
        out["outcome"] = oc

    # OLS: does the complaint predict the reply sentiment beyond model and variant?
    try:
        import statsmodels.formula.api as smf
        d = df.dropna(subset=["compound", "narr_compound", "narrative_chars", "redaction_ratio"]).copy()
        d["log_narr_chars"] = np.log(d["narrative_chars"].clip(lower=1))
        model = smf.ols("compound ~ C(Model) * C(Prompt_Variant) + narr_compound + log_narr_chars + "
                        "redaction_ratio + C(Product_family)", data=d).fit(cov_type="cluster",
                                                                     cov_kwds={"groups": d["Row"]})
        with open(os.path.join(TAB, "ols_reply_sentiment.txt"), "w") as f:
            f.write(model.summary().as_text())
        out["ols"] = model
    except Exception as e:  # keep the pipeline going if statsmodels misbehaves
        print("OLS skipped:", e)
    return out


# ------------------------------------------------------------------------ figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 10,
        "legend.frameon": False,
    })
    return plt


def save(plt, name):
    p = os.path.join(FIG, name)
    plt.savefig(p)
    plt.close()
    print("  wrote", p)


def fig_distributions(df, plt, col, label, name, clip=None):
    """Boxplots per variant, ChatGPT vs Mistral side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        data = [df[(df.Prompt_Variant == v) & (df.Model == m)][col].dropna() for m in MODELS]
        if clip:
            data = [d.clip(*clip) for d in data]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=1.4),
                        whiskerprops=dict(color=TEXT2, linewidth=0.8),
                        capprops=dict(color=TEXT2, linewidth=0.8))
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLOR[m]); patch.set_edgecolor("none")
        for i, (d, m) in enumerate(zip(data, MODELS), start=1):
            ax.text(i, d.median(), f"{d.median():.0f}" if abs(d.median()) >= 10 else f"{d.median():.2f}", ha="center", va="bottom", fontsize=7.5,
                    color=TEXT, transform=ax.transData, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.8))
        ax.set_xticks([1, 2]); ax.set_xticklabels(MODELS)
        ax.set_title(VARIANT_LABEL[v]); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(label)
    fig.suptitle(f"{label} by model and prompt variant (median labelled, whiskers 1.5 IQR)",
                 fontsize=10, color=TEXT, fontweight="bold", y=1.02)
    save(plt, name)


def fig_markers(df, plt):
    """Dot plot: marker prevalence per model, one panel per variant."""
    keys = [k for k in MARKERS if k != "echo_redaction"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, v in zip(axes, VARIANTS):
        g = df[df.Prompt_Variant == v]
        vals = {m: [g[g.Model == m][k].mean() * 100 for k in keys] for m in MODELS}
        for yi, a, b in zip(y, vals["ChatGPT"], vals["Mistral"]):
            ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
        for m in MODELS:
            ax.scatter(vals[m], y, s=42, color=COLOR[m], zorder=2, label=m, edgecolor="white", linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels([MARKER_LABEL[k] for k in keys])
        ax.set_xlim(0, 100); ax.set_xlabel("% of replies"); ax.set_title(VARIANT_LABEL[v])
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Share of replies containing each rhetorical marker", fontsize=10, color=TEXT,
                 fontweight="bold", y=1.0)
    save(plt, "fig3_markers.png")


def fig_mirroring(df, plt):
    """Reply sentiment as a function of narrative sentiment (binned), per model and variant."""
    bins = np.linspace(-1, 1, 11)
    df = df.copy()
    df["nbin"] = pd.cut(df["narr_compound"], bins, include_lowest=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        for m in MODELS:
            g = df[(df.Prompt_Variant == v) & (df.Model == m)].groupby("nbin", observed=True)["compound"]
            mean = g.mean(); se = g.std() / np.sqrt(g.size().clip(lower=1))
            x = [iv.mid for iv in mean.index]
            ax.plot(x, mean.values, color=COLOR[m], linewidth=2, label=m)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=COLOR[m], alpha=0.15, linewidth=0)
        ax.set_title(VARIANT_LABEL[v]); ax.set_xlabel("Complaint narrative VADER compound")
    axes[0].set_ylabel("Reply VADER compound"); axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Do replies mirror the complaint's tone? Reply sentiment vs narrative sentiment (mean, 95% CI)",
                 fontsize=10, color=TEXT, fontweight="bold", y=1.02)
    save(plt, "fig4_mirroring.png")


def fig_product(df, plt):
    prod = df.groupby(["Product_family", "Model"])["compound"].mean().unstack("Model")
    n = df.groupby("Product_family")["Row"].nunique()
    prod = prod.loc[n.sort_values().index]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(prod) + 1.2))
    y = np.arange(len(prod))
    for yi, (_, r) in zip(y, prod.iterrows()):
        ax.plot([r["ChatGPT"], r["Mistral"]], [yi, yi], color=GRID, linewidth=2, zorder=1)
    for m in MODELS:
        ax.scatter(prod[m], y, s=42, color=COLOR[m], zorder=2, label=m, edgecolor="white", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{p}  (n={n[p]:,})" for p in prod.index], fontsize=8)
    ax.set_xlabel("Mean reply VADER compound (all three variants)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.set_title("Reply sentiment by complaint product")
    save(plt, "fig5_product.png")


def fig_compliance(df, plt):
    """Instruction following: sentence-count / format compliance and truncation."""
    metrics = [("sentence_compliant", "Follows the length /\nformat instruction"),
               ("truncated", "Hit the max_tokens cap\n(reply cut off)"),
               ("markdown", "Uses markdown\nformatting")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    fig.subplots_adjust(wspace=0.35)
    x = np.arange(len(VARIANTS)); w = 0.36
    for ax, (col, title) in zip(axes, metrics):
        for i, m in enumerate(MODELS):
            vals = [df[(df.Prompt_Variant == v) & (df.Model == m)][col].mean() * 100 for v in VARIANTS]
            bars = ax.bar(x + (i - 0.5) * w, vals, width=w - 0.04, color=COLOR[m], label=m)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center",
                        fontsize=7.5, color=TEXT)
        ax.set_xticks(x); ax.set_xticklabels([VARIANT_LABEL[v].replace(" ", "\n", 1) for v in VARIANTS], fontsize=8)
        ax.set_ylim(0, 115); ax.set_ylabel("% of replies"); ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 1.08))
    save(plt, "fig6_compliance.png")


def fig_outcome(df, plt):
    if "response_category" not in df:
        return
    oc = df.groupby(["response_category", "Model"])["promise_outcome"].mean().unstack("Model") * 100
    n = df.groupby("response_category")["Row"].nunique()
    oc = oc.loc[n.sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(oc) + 1.2))
    y = np.arange(len(oc))
    for yi, (_, r) in zip(y, oc.iterrows()):
        ax.plot([r["ChatGPT"], r["Mistral"]], [yi, yi], color=GRID, linewidth=2, zorder=1)
    for m in MODELS:
        ax.scatter(oc[m], y, s=42, color=COLOR[m], zorder=2, label=m, edgecolor="white", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{c}  (n={n[c]:,})" for c in oc.index], fontsize=8)
    ax.set_xlabel("% of replies that promise a concrete outcome (refund, removal, reversal...)")
    ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Promised outcomes vs what the company actually did")
    save(plt, "fig7_outcome_promises.png")


# --------------------------------------------------------------------------- main
def main():
    cache = os.path.join(TAB, "reply_features.csv.gz")
    if os.getenv("ANALYSIS_FROM_CACHE") and os.path.exists(cache):
        print("Loading cached features from", cache)
        df = pd.read_csv(cache, dtype={"Row": str}, keep_default_na=False)
        comp = df.drop_duplicates("Row")[["Row", "narr_compound", "narr_neg", "narrative_chars",
                                           "redaction_ratio"]].copy()
    else:
        print("Loading...")
        df, comp = load()
        print(f"  {len(df):,} replies over {df['Row'].nunique():,} complaints")
        print("Computing features (VADER, readability, markers)...")
        df, comp = add_features(df, comp)
        df.to_csv(cache, index=False, compression="gzip")

    print("Statistics...")
    summ, mc, vc = summary_tables(df)
    ctx = complaint_context_tables(df)

    print("Figures...")
    plt = setup_mpl()
    fig_distributions(df, plt, "Response_Chars", "Reply length (characters)", "fig1_length.png")
    fig_distributions(df, plt, "compound", "Reply sentiment (VADER compound)", "fig2_sentiment.png")
    fig_markers(df, plt)
    fig_mirroring(df, plt)
    fig_product(df, plt)
    fig_compliance(df, plt)
    fig_distributions(df, plt, "fk_grade", "Flesch-Kincaid grade level", "fig8_readability.png", clip=(0, 25))

    # console digest
    pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
    print("\n=== Means by model x variant ===")
    cols = ["Model", "Variant", "n"] + [l for _, l in METRICS] + [MARKER_LABEL.get(f, f) for f in FLAGS]
    print(summ[cols].T.to_string())
    print("\n=== Paired model comparison (ChatGPT - Mistral) ===")
    print(mc[["Variant", "Metric", "ChatGPT mean", "Mistral mean", "Diff (ChatGPT - Mistral)",
              "Cohen d (paired)", "p"]].to_string(index=False))
    print("\n=== Variant effects vs V1 (paired within model) ===")
    print(vc[["Model", "Comparison", "Metric", "Variant mean", "V1 mean", "Diff", "Cohen d (paired)", "p"]]
          .to_string(index=False))
    print("\n=== Mirroring ===")
    print(ctx["mirroring"].to_string(index=False))
    print("\n=== By narrative sentiment quintile ===")
    print(ctx["quintile"].to_string(index=False))
    if "outcome" in ctx:
        print("\n=== By company outcome ===")
        print(ctx["outcome"].to_string(index=False))
    if "ols" in ctx:
        print("\n=== OLS (reply compound), clustered by complaint ===")
        print(ctx["ols"].summary().tables[1])
    # complaint-side descriptives
    print("\n=== Complaint narratives ===")
    print(comp[["narr_compound", "narr_neg", "narrative_chars", "redaction_ratio"]].describe().T.to_string())
    print("\n=== Products (complaints) ===")
    print(df.drop_duplicates("Row")["Product_family"].value_counts().to_string())


if __name__ == "__main__":
    main()
