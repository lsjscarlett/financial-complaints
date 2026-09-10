"""Robustness analyses for the paper: variance decomposition and an invented-facts audit.

Runs from analysis/tables/reply_features.csv.gz (written by eda_sentiment.py) and
dataset/complaints_10k.csv. Writes tables to analysis/tables/ and one figure.

1. Variance decomposition. The design is fully crossed and balanced (every complaint
   has one reply per model x prompt), so the sums of squares for complaint, model,
   prompt, and model x prompt are orthogonal. Reports eta-squared for each source
   per reply feature: how much of the variation in a feature is explained by which
   prompt was used, which model answered, and which complaint was being answered.

2. Invented-facts audit. CFPB narratives redact every dollar amount as {$1,234.00},
   every date as XX/XX/XXXX, and every name as XXXX, and the prompt never names the
   company. So any calendar date, phone number, or personal name in a reply is
   invented, and a dollar amount is invented unless it matches one in the narrative.
"""

import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")

MODELS = ["ChatGPT", "Mistral"]
VARIANTS = ["v1_terse", "v2_empathetic", "v3_structured"]
VARIANT_LABEL = {"v1_terse": "V1 terse", "v2_empathetic": "V2 empathetic", "v3_structured": "V3 structured"}
COLOR = {"ChatGPT": "#2a78d6", "Mistral": "#eb6834"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load():
    df = pd.read_csv(os.path.join(TAB, "reply_features.csv.gz"), dtype={"Row": str}, keep_default_na=False)
    comp = pd.read_csv(os.path.join(ROOT, "dataset", "complaints_10k.csv"), dtype=str, keep_default_na=False)
    comp = comp.rename(columns={"row_id": "Row"})[["Row", "Consumer complaint narrative"]]  # Company is already in the feature table
    return df.merge(comp, on="Row", how="left")


# ------------------------------------------------------ 1. variance decomposition
FEATURES = [
    ("Response_Chars", "Length (chars)"), ("sentences", "Sentences"), ("compound", "VADER compound"),
    ("fk_grade", "FK grade"), ("apology", "Apology"), ("ownership", "Ownership"),
    ("time_bound", "Time-bound commitment"), ("escalation", "Escalation"),
    ("asks_info", "Asks for information"), ("gratitude", "Thanks the customer"),
    ("placeholder", "Leaves a [placeholder]"),
]


def variance_decomposition(df):
    rows = []
    for col, label in FEATURES:
        y = df[col].astype(float)
        g = y.mean()
        ss_total = ((y - g) ** 2).sum()
        # balanced crossed design: between-group SS are orthogonal
        def ss_between(keys):
            m = df.assign(_y=y).groupby(keys)["_y"].agg(["mean", "size"])
            return (m["size"] * (m["mean"] - g) ** 2).sum()
        ss_model = ss_between(["Model"])
        ss_prompt = ss_between(["Prompt_Variant"])
        ss_cell = ss_between(["Model", "Prompt_Variant"])
        ss_inter = ss_cell - ss_model - ss_prompt
        ss_row = ss_between(["Row"])
        ss_resid = ss_total - ss_cell - ss_row
        rows.append({"Feature": label, "Prompt": ss_prompt / ss_total, "Model": ss_model / ss_total,
                     "Model x Prompt": ss_inter / ss_total, "Complaint": ss_row / ss_total,
                     "Residual": ss_resid / ss_total})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, "variance_decomposition.csv"), index=False)
    return out


def fig_variance(vd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.titlecolor": TEXT, "axes.titleweight": "bold",
                         "legend.frameon": False})
    # ordinal-ish sources: use the blue sequential ramp for prompt/model/interaction, gray for the rest
    sources = ["Prompt", "Model", "Model x Prompt", "Complaint", "Residual"]
    colors = ["#104281", "#2a78d6", "#86b6ef", "#b5b4ad", "#e6e5e1"]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(vd) + 1.2))
    y = np.arange(len(vd))[::-1]
    left = np.zeros(len(vd))
    for src, c in zip(sources, colors):
        vals = vd[src].values * 100
        ax.barh(y, vals, left=left, color=c, label=src, height=0.62, edgecolor="white", linewidth=1)
        for yi, l, v in zip(y, left, vals):
            if v >= 8:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if c in ("#104281", "#2a78d6") else TEXT)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels(vd["Feature"])
    ax.set_xlim(0, 100); ax.set_xlabel("% of variance (eta-squared)"); ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=8)
    ax.set_title("What explains the variation in a reply? Prompt vs model vs complaint")
    p = os.path.join(FIG, "fig9_variance_decomposition.png")
    plt.savefig(p); plt.close(); print("  wrote", p)


# ------------------------------------------------------- 2. invented-facts audit
MONTH = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
RE_DATE = re.compile(rf"\b(?:{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})\b", re.I)
RE_AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
RE_PHONE = re.compile(r"\b(?:1[-. ]?)?(?:\(\d{3}\)|\d{3})[-. ]\d{3}[-. ]\d{4}\b")
# a salutation with an actual name; "Dear Customer", "Dear Sir", and "Dear XXXX" (the
# redaction token copied as a name) are excluded
RE_SALUTATION = re.compile(r"\bdear\s+(?:(?:mr|mrs|ms|dr)\.?\s+)?(?!(?:customer|valued|sir|madam|team|xxxx|consumer|client|member)\b)[A-Z][a-z]{2,}\b", re.I)
RE_PLACEHOLDER = re.compile(r"\[[^\]\n]{2,60}\]")
RE_NARR_AMOUNT = re.compile(r"\{\$\s?([\d,]+(?:\.\d{2})?)\}")
RE_DEADLINE = re.compile(r"\b(?:within|in)\s+(?:the next\s+)?(\d+|one|two|three|five|seven|ten|fourteen|thirty)\s*(?:-|to\s+\d+\s*)?(business |working )?(days?|hours?|weeks?)\b", re.I)


def norm_amount(a):
    return a.replace("$", "").replace(",", "").replace(" ", "").strip()


def company_tokens(name):
    stop = {"inc", "inc.", "llc", "corp", "corporation", "company", "co", "co.", "bank", "the", "of",
            "and", "&", "group", "financial", "services", "n.a.", "na", "credit", "union", "n.a"}
    toks = [t.strip(".,").lower() for t in name.split()]
    return [t for t in toks if len(t) >= 4 and t not in stop]


def invented_facts(df):
    out = {}
    narr = df["Consumer complaint narrative"].astype(str)
    reply = df["Response"].astype(str)
    reply_lower = reply.str.lower()

    # placeholders are not dates
    reply_np = reply.str.replace(RE_PLACEHOLDER.pattern, " ", regex=True)
    out["date"] = reply_np.apply(lambda t: bool(RE_DATE.search(t)))
    out["phone"] = reply_np.apply(lambda t: bool(RE_PHONE.search(t)))
    out["salutation_name"] = reply_np.apply(lambda t: bool(RE_SALUTATION.search(t)))

    # amounts: invented unless present in the narrative's {$...} redactions
    def amount_status(r, n):
        found = [norm_amount(a) for a in RE_AMOUNT.findall(r)]
        if not found:
            return "none"
        have = {norm_amount(a) for a in RE_NARR_AMOUNT.findall(n)}
        # also accept amounts written plainly in the narrative
        have |= {norm_amount(a) for a in RE_AMOUNT.findall(n)}
        return "grounded" if all(a in have or a.rstrip("0").rstrip(".") in {h.rstrip("0").rstrip(".") for h in have}
                                 for a in found) else "invented"
    st = [amount_status(r, n) for r, n in zip(reply, narr)]
    out["amount_any"] = pd.Series([s != "none" for s in st], index=df.index)
    out["amount_invented"] = pd.Series([s == "invented" for s in st], index=df.index)

    # company name: the prompt never contains it, so a match means the model inferred it
    # from the narrative (grounded) or guessed it (invented)
    def company_status(r, n, c):
        toks = company_tokens(c)
        if not toks:
            return "n/a"
        rl, nl = r.lower(), n.lower()
        hit = [t for t in toks if t in rl]
        if not hit:
            return "none"
        return "grounded" if any(t in nl for t in hit) else "invented"
    cs = [company_status(r, n, c) for r, n, c in zip(reply, narr, df["Company"].astype(str))]
    out["company_named"] = pd.Series([s in ("grounded", "invented") for s in cs], index=df.index)
    out["company_invented"] = pd.Series([s == "invented" for s in cs], index=df.index)
    out["deadline"] = reply_np.apply(lambda t: bool(RE_DEADLINE.search(t)))

    flags = pd.DataFrame(out)
    # company_invented is excluded: company-name token matching is too noisy to count as fabrication
    flags["any_invented"] = flags[["date", "phone", "salutation_name", "amount_invented"]].any(axis=1)
    res = pd.concat([df[["Row", "Model", "Prompt_Variant"]], flags], axis=1)

    summary = res.groupby(["Model", "Prompt_Variant"])[list(flags.columns)].mean().reset_index()
    summary["Prompt_Variant"] = summary["Prompt_Variant"].map(VARIANT_LABEL)
    summary.to_csv(os.path.join(TAB, "invented_facts_by_model_variant.csv"), index=False)

    # examples for the paper appendix
    ex = df.assign(**flags)
    examples = []
    for flag in ("date", "amount_invented", "company_invented", "salutation_name", "phone"):
        e = ex[ex[flag]].sample(min(3, int(ex[flag].sum())), random_state=1)
        for _, r in e.iterrows():
            examples.append({"flag": flag, "Model": r["Model"], "Variant": r["Prompt_Variant"],
                             "Row": r["Row"], "Response": r["Response"][:400]})
    pd.DataFrame(examples).to_csv(os.path.join(TAB, "invented_facts_examples.csv"), index=False)
    return summary, res


def main():
    print("Loading features...")
    df = load()
    print(f"  {len(df):,} replies")

    print("Variance decomposition...")
    vd = variance_decomposition(df)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(vd.to_string(index=False))
    fig_variance(vd)

    print("\nInvented-facts audit (share of replies)...")
    summary, res = invented_facts(df)
    print(summary.to_string(index=False))
    print("\nOverall:", res[["date", "phone", "salutation_name", "amount_any", "amount_invented",
                            "company_named", "company_invented", "deadline", "any_invented"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
