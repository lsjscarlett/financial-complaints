"""Analyse the order-swapped pairwise verdicts (analysis/tables/pairwise_judgments.csv).

For every pair and judge, the two verdicts (one per presentation order) are combined:
  consistent   the same reply wins in both orders
  position     a different reply wins in each order, i.e. the judge followed the position
  tie          at least one verdict is a tie and the other is not a clear contradiction
Position bias is reported per judge as the share of pairs decided by position and, separately,
the share of all verdicts that chose A vs B. Head-to-head win rates use consistent pairs only,
with a 95% cluster-bootstrap CI over complaints, and are compared with the absolute-score ranking
from llm_judge.py (judge_ratings*.csv).

Writes analysis/tables/pairwise_position_bias.csv, pairwise_winrates.csv, pairwise_vs_absolute.csv.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
CRITERIA = ["overall", "concreteness", "tone"]
B = 1000
VAR_OF = {"V1": "v1_terse", "A00": "v2_empathetic", "V3": "v3_structured", "000": "f_000_base",
          "00C": "f_00C_constraints", "A0C": "f_A0C_empathetic_constraints", "ABC": "f_ABC_empathetic_format_constraints"}


def combine(v1, v2):
    """v1: verdict with left shown as A; v2: verdict with left shown as B. Returns
    ('left'|'right'|'tie'|'position', ...)."""
    left1 = {"A": "left", "B": "right", "TIE": "tie"}.get(v1, "")
    left2 = {"A": "right", "B": "left", "TIE": "tie"}.get(v2, "")
    if not left1 or not left2:
        return ""
    if left1 == left2:
        return left1                      # includes tie/tie
    if "tie" in (left1, left2):
        return "tie"
    return "position"


def main():
    d = pd.read_csv(os.path.join(TAB, "pairwise_judgments.csv"), dtype=str, keep_default_na=False)
    d = d[d["error"] == ""]
    pv = d.pivot_table(index=["Row", "pair_type", "context", "left", "right", "Judge"], columns="order",
                       values=CRITERIA, aggfunc="first")
    pv = pv.dropna(subset=[(c, o) for c in CRITERIA for o in ("1", "2")])
    w = pv.index.to_frame(index=False)
    for c in CRITERIA:
        w[f"{c}_1"] = pv[(c, "1")].values
        w[f"{c}_2"] = pv[(c, "2")].values
        w[c] = [combine(a, b) for a, b in zip(w[f"{c}_1"], w[f"{c}_2"])]
    print(f"{len(w):,} pairs with both orders, judges: {sorted(w['Judge'].unique())}")

    # position bias
    rows = []
    for j, g in w.groupby("Judge"):
        for c in CRITERIA:
            first = pd.concat([g[f"{c}_1"], g[f"{c}_2"]])
            rows.append({"Judge": j, "criterion": c, "n_pairs": len(g),
                         "share_A_chosen": (first == "A").mean(), "share_B_chosen": (first == "B").mean(),
                         "share_tie": (first == "TIE").mean(),
                         "consistent": g[c].isin(["left", "right"]).mean(),
                         "decided_by_position": (g[c] == "position").mean(),
                         "tie_or_mixed": (g[c] == "tie").mean()})
    pb = pd.DataFrame(rows)
    pb.to_csv(os.path.join(TAB, "pairwise_position_bias.csv"), index=False)

    # win rates on consistent pairs, cluster bootstrap over complaints
    rng = np.random.default_rng(7)
    rows = []
    for (j, pt, ctx, l, r), g in w.groupby(["Judge", "pair_type", "context", "left", "right"]):
        for c in CRITERIA:
            dec = g[g[c].isin(["left", "right"])]
            n = len(dec)
            if n == 0:
                continue
            win = (dec[c] == "left").astype(float).values
            boots = np.array([win[rng.integers(0, n, n)].mean() for _ in range(B)])
            rows.append({"Judge": j, "pair_type": pt, "context": ctx, "left": l, "right": r, "criterion": c,
                         "n_decided": n, "n_pairs": len(g), "left_win_rate": win.mean(),
                         "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5)})
    wr = pd.DataFrame(rows)
    wr.to_csv(os.path.join(TAB, "pairwise_winrates.csv"), index=False)

    # compare with absolute scores: mean overall(left) - overall(right) on the same complaints and judge
    jr = pd.concat([pd.read_csv(os.path.join(TAB, f), dtype=str, keep_default_na=False)
                    for f in ("judge_ratings.csv", "judge_ratings_study2.csv")], ignore_index=True)
    jr = jr[jr["error"] == ""]
    jr["overall"] = pd.to_numeric(jr["overall"], errors="coerce")
    jr["concreteness"] = pd.to_numeric(jr["concreteness"], errors="coerce")
    jr["tone"] = pd.to_numeric(jr["tone"], errors="coerce")
    jr = jr.set_index(["Judge", "Row", "Model", "Prompt_Variant"])
    rows = []
    for (j, pt, ctx, l, r), g in w.groupby(["Judge", "pair_type", "context", "left", "right"]):
        lm, lc = l.split("|"); rm, rc = r.split("|")
        for c in CRITERIA:
            diffs, wins = [], []
            for _, p in g.iterrows():
                try:
                    a = jr.loc[(j, p["Row"], lm, VAR_OF[lc]), c]; b = jr.loc[(j, p["Row"], rm, VAR_OF[rc]), c]
                except KeyError:
                    continue
                if pd.isna(a) or pd.isna(b):
                    continue
                diffs.append(a - b)
                wins.append({"left": 1.0, "right": 0.0, "tie": 0.5}.get(p[c], np.nan))
            if not diffs:
                continue
            diffs, wins = np.array(diffs), np.array(wins)
            ok = ~np.isnan(wins)
            abs_pref = "left" if np.mean(diffs) > 0 else "right"
            pair_pref = "left" if np.nanmean(wins) > 0.5 else "right"
            rows.append({"Judge": j, "pair_type": pt, "context": ctx, "left": l, "right": r, "criterion": c,
                         "n": int(ok.sum()), "absolute_mean_diff": np.mean(diffs), "pairwise_left_share": np.nanmean(wins),
                         "same_direction": abs_pref == pair_pref,
                         "item_agreement": float(np.mean(np.sign(diffs[ok]) == np.sign(wins[ok] - 0.5))) if ok.any() else np.nan})
    va = pd.DataFrame(rows)
    va.to_csv(os.path.join(TAB, "pairwise_vs_absolute.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print("\n=== Position bias by judge ===")
    print(pb.to_string(index=False))
    print("\n=== Head-to-head win rate of the left-hand reply (consistent pairs), overall criterion ===")
    print(wr[wr["criterion"] == "overall"][["Judge", "pair_type", "context", "left", "right", "n_decided", "left_win_rate", "ci_lo", "ci_hi"]].to_string(index=False))
    print("\n=== Pairwise vs absolute scores: same direction? (overall) ===")
    print(va[va["criterion"] == "overall"][["Judge", "left", "right", "n", "absolute_mean_diff", "pairwise_left_share", "same_direction", "item_agreement"]].to_string(index=False))


if __name__ == "__main__":
    main()
