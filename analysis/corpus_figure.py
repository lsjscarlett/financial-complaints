"""Figure 0: what the 10,000 complaints look like (narrative length, narrative sentiment,
year received, company outcome). Reads dataset/complaints_10k.csv and the feature table
for the narrative VADER scores; writes analysis/figures/fig0_corpus.png."""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLUE, BLUE_LIGHT, TEXT, TEXT2, GRID = "#2a78d6", "#86b6ef", "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "axes.titlecolor": TEXT, "axes.titleweight": "bold",
                         "axes.titlesize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    c = pd.read_csv(os.path.join(ROOT, "dataset", "complaints_10k.csv"), dtype=str, keep_default_na=False)
    c = c.rename(columns={"row_id": "Row"})
    for k in ("narrative_chars", "year"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    f = pd.read_csv(os.path.join(HERE, "tables", "reply_features.csv.gz"), dtype={"Row": str},
                    keep_default_na=False, usecols=["Row", "narr_compound"]).drop_duplicates("Row")
    c = c.merge(f, on="Row", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(hspace=0.55, wspace=0.75)

    ax = axes[0, 0]
    ax.hist(c["narrative_chars"].clip(upper=3000), bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(1500, color=TEXT2, linewidth=1)
    ax.text(1530, ax.get_ylim()[1] * 0.9, "prompt truncation\n(12% of narratives)", fontsize=7.5, color=TEXT2, va="top")
    ax.set_xlabel("Narrative length (characters, clipped at 3,000)"); ax.set_ylabel("Complaints")
    ax.set_title("A. Narrative length (median 663)"); ax.grid(axis="x", visible=False)

    ax = axes[0, 1]
    ax.hist(c["narr_compound"], bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Narrative VADER compound"); ax.set_ylabel("Complaints")
    ax.set_title("B. Narrative sentiment (57% below zero)"); ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    yr = c["year"].value_counts().sort_index()
    ax.bar(yr.index.astype(int), yr.values, color=BLUE, width=0.7)
    ax.set_xlabel("Year received"); ax.set_ylabel("Complaints"); ax.set_title("C. Year received")
    ax.grid(axis="x", visible=False); ax.set_xticks(yr.index.astype(int)); ax.tick_params(axis="x", labelsize=7.5, rotation=45)

    ax = axes[1, 1]
    oc = c["response_category"].value_counts(normalize=True).sort_values() * 100
    y = np.arange(len(oc))
    ax.barh(y, oc.values, color=BLUE, height=0.6)
    for yi, v in zip(y, oc.values):
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=7.5, color=TEXT)
    ax.set_yticks(y); ax.set_yticklabels(oc.index, fontsize=8); ax.set_xlim(0, 100)
    ax.set_xlabel("% of complaints"); ax.set_title("D. Company's recorded outcome"); ax.grid(axis="y", visible=False)

    p = os.path.join(HERE, "figures", "fig0_corpus.png")
    plt.savefig(p); plt.close(); print("wrote", p)


if __name__ == "__main__":
    main()
