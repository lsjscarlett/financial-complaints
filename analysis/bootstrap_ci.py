"""Uncertainty for the headline contrasts: cluster-bootstrap CIs and logistic robustness checks.

Study 1 (reply_features.csv.gz, 60,000 replies): for each feature, the paired ChatGPT - Mistral
gap within each prompt and the V2 - V1, V3 - V1, V3 - V2 contrasts within each model, with a 95%
percentile CI from a cluster bootstrap over complaints (B resamples of complaint ids). Binary
features are additionally modelled with logistic regression y ~ Model * Prompt (complaint-
clustered SEs), reported as odds ratios and average marginal effects, so that the linear
probability estimates in the paper have a check that respects the 0-1 bounds.

Study 2 (factorial cells, 2,999 complaints): logistic y ~ A * B * C per model with clustered
SEs for the binary features, next to the linear-probability coefficients already in
study2_factorial_effects.csv, and cluster-bootstrap CIs for the three main effects.

Environment: BOOT_B (default 1000), BOOT_STUDY2=0 to skip Study 2 (it recomputes features).
Writes analysis/tables/ci_study1_contrasts.csv, ci_study1_logit.csv, ci_study2_logit.csv,
ci_study2_main_effects.csv.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
sys.path.insert(0, HERE)
B = int(os.getenv("BOOT_B") or "1000")
SEED = 11
FEATURES = [("Response_Chars", "Length (chars)"), ("sentences", "Sentences"), ("compound", "VADER compound"),
            ("fk_grade", "FK grade"), ("apology", "Apology"), ("ownership", "Ownership"),
            ("gratitude", "Thanks the customer"), ("time_bound", "Time-bound commitment"),
            ("escalation", "Escalation"), ("placeholder", "Leaves a [placeholder]"),
            ("promise_outcome", "Promises an outcome"), ("hedging", "Hedging")]
BINARY = ["apology", "ownership", "gratitude", "time_bound", "escalation", "placeholder", "promise_outcome", "hedging"]
VLABEL = {"v1_terse": "V1", "v2_empathetic": "V2", "v3_structured": "V3"}
warnings.filterwarnings("ignore")


def boot_mean(x, rng):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, (B, n))
    b = x[idx].mean(axis=1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def study1_contrasts(f, rng):
    rows = []
    for col, label in FEATURES:
        w = f.pivot_table(index="Row", columns=["Model", "Prompt_Variant"], values=col)
        for v in VLABEL:
            d = (w[("ChatGPT", v)] - w[("Mistral", v)]).dropna()
            m, lo, hi = boot_mean(d, rng)
            rows.append({"contrast": f"ChatGPT - Mistral | {VLABEL[v]}", "feature": label, "n": len(d),
                         "estimate": m, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0) or (hi < 0)})
        for mdl in ("ChatGPT", "Mistral"):
            for a, b in (("v2_empathetic", "v1_terse"), ("v3_structured", "v1_terse"), ("v3_structured", "v2_empathetic")):
                d = (w[(mdl, a)] - w[(mdl, b)]).dropna()
                m, lo, hi = boot_mean(d, rng)
                rows.append({"contrast": f"{VLABEL[a]} - {VLABEL[b]} | {mdl}", "feature": label, "n": len(d),
                             "estimate": m, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0) or (hi < 0)})
    return pd.DataFrame(rows)


def study1_logit(f):
    rows = []
    d = f.copy()
    d["mistral"] = (d["Model"] == "Mistral").astype(int)
    d["v2"] = (d["Prompt_Variant"] == "v2_empathetic").astype(int)
    d["v3"] = (d["Prompt_Variant"] == "v3_structured").astype(int)
    for col in BINARY:
        y = d[col].astype(float)
        if y.nunique() < 2:
            continue
        X = sm.add_constant(pd.DataFrame({"mistral": d["mistral"], "v2": d["v2"], "v3": d["v3"],
                                          "mistral:v2": d["mistral"] * d["v2"], "mistral:v3": d["mistral"] * d["v3"]}))
        try:
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["Row"].values})
        except Exception as e:
            print("logit skip", col, e); continue
        # average marginal effects by finite difference on the fitted model
        def ame(var):
            X1, X0 = X.copy(), X.copy()
            X1[var] = 1; X0[var] = 0
            for inter in ("mistral:v2", "mistral:v3"):
                X1[inter] = X1["mistral"] * X1[inter.split(":")[1]]
                X0[inter] = X0["mistral"] * X0[inter.split(":")[1]]
            return float((fit.predict(X1) - fit.predict(X0)).mean())
        for term in ["mistral", "v2", "v3", "mistral:v2", "mistral:v3"]:
            rec = {"feature": col, "term": term, "odds_ratio": np.exp(fit.params[term]),
                   "or_ci_lo": np.exp(fit.conf_int().loc[term, 0]), "or_ci_hi": np.exp(fit.conf_int().loc[term, 1]),
                   "p": fit.pvalues[term]}
            if ":" not in term:
                rec["avg_marginal_effect"] = ame(term)
            rows.append(rec)
    return pd.DataFrame(rows)


def study2(rng):
    import study2_factorial as s2
    df = s2.load()
    d = df[df["Prompt_Variant"].isin(s2.CELLS)].copy()
    logit_rows, me_rows = [], []
    for mdl in s2.MODELS:
        dm = d[d["Model"] == mdl].copy()
        full = dm.groupby("Row")["Prompt_Variant"].nunique()
        dm = dm[dm["Row"].isin(full[full == 8].index)].copy()
        for col in BINARY:
            dm["_y"] = dm[col].astype(float)
            if dm["_y"].nunique() < 2:
                continue
            try:
                fit = smf.glm("_y ~ A * B * C", data=dm, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": dm["Row"].values})
                ci = fit.conf_int()
                for term in ["A", "B", "C", "A:B", "A:C", "B:C", "A:B:C"]:
                    logit_rows.append({"Model": mdl, "feature": col, "term": term, "odds_ratio": np.exp(fit.params[term]),
                                       "or_ci_lo": np.exp(ci.loc[term, 0]), "or_ci_hi": np.exp(ci.loc[term, 1]),
                                       "p": fit.pvalues[term], "n_complaints": dm["Row"].nunique()})
            except Exception as e:
                print("study2 logit skip", mdl, col, e)
        # bootstrap main effects (mean over on-cells minus mean over off-cells, per complaint)
        for col, label in s2.FEATURES:
            w = dm.pivot_table(index="Row", columns="cell", values=col).dropna()
            for k, factor in enumerate("ABC"):
                on = [c for c in w.columns if c[k] == factor]
                off = [c for c in w.columns if c[k] == "0"]
                diff = w[on].mean(axis=1) - w[off].mean(axis=1)
                m, lo, hi = boot_mean(diff, rng)
                me_rows.append({"Model": mdl, "feature": label, "factor": factor, "n": len(diff), "estimate": m,
                                "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0) or (hi < 0)})
    return pd.DataFrame(logit_rows), pd.DataFrame(me_rows)


def main():
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    f = pd.read_csv(os.path.join(TAB, "reply_features.csv.gz"), dtype={"Row": str}, keep_default_na=False,
                    usecols=["Row", "Model", "Prompt_Variant"] + [c for c, _ in FEATURES])
    for c, _ in FEATURES:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[f["Model"].isin(["ChatGPT", "Mistral"])]
    print(f"Study 1: {len(f):,} replies, {f['Row'].nunique():,} complaints, B = {B}")
    c1 = study1_contrasts(f, rng)
    c1.to_csv(os.path.join(TAB, "ci_study1_contrasts.csv"), index=False)
    print("\n=== Study 1 contrasts with 95% cluster-bootstrap CIs (VADER, placeholder, apology shown) ===")
    print(c1[c1["feature"].isin(["VADER compound", "Leaves a [placeholder]", "Apology"])].to_string(index=False))
    print(f"\n{int((~c1['excludes_zero']).sum())} of {len(c1)} contrasts have a CI that includes zero:")
    print(c1[~c1["excludes_zero"]][["contrast", "feature", "estimate", "ci_lo", "ci_hi"]].to_string(index=False))
    l1 = study1_logit(f)
    l1.to_csv(os.path.join(TAB, "ci_study1_logit.csv"), index=False)
    print("\n=== Study 1 logistic y ~ Model * Prompt, odds ratios (clustered SEs) ===")
    print(l1.to_string(index=False))
    if (os.getenv("BOOT_STUDY2") or "1") != "0":
        l2, me = study2(rng)
        l2.to_csv(os.path.join(TAB, "ci_study2_logit.csv"), index=False)
        me.to_csv(os.path.join(TAB, "ci_study2_main_effects.csv"), index=False)
        print("\n=== Study 2 logistic y ~ A * B * C per model, odds ratios ===")
        print(l2.to_string(index=False))
        print("\n=== Study 2 main effects with bootstrap CIs ===")
        print(me.to_string(index=False))


if __name__ == "__main__":
    main()
