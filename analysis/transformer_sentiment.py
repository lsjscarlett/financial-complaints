"""Robustness check: re-score reply sentiment with a transformer model instead of VADER.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest (negative / neutral / positive)
on a random subset of complaints (all 6 replies each) and compares the model x prompt
picture with the VADER one. Writes analysis/tables/transformer_sentiment.csv and
transformer_vs_vader.csv.

Environment: TS_COMPLAINTS (default 3000), TS_BATCH (default 32).
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
N = int(os.getenv("TS_COMPLAINTS") or "3000")
BATCH = int(os.getenv("TS_BATCH") or "32")
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 4)

    df = pd.read_csv(os.path.join(TAB, "reply_features.csv.gz"), dtype={"Row": str}, keep_default_na=False,
                     usecols=["Row", "Model", "Prompt_Variant", "Response", "compound", "narr_compound"])
    rows = pd.Series(df["Row"].unique()).sample(min(N, df["Row"].nunique()), random_state=42)
    df = df[df["Row"].isin(set(rows))].reset_index(drop=True)
    print(f"Scoring {len(df):,} replies from {len(rows):,} complaints with {MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    probs = np.zeros((len(df), len(labels)), dtype=np.float32)
    texts = df["Response"].astype(str).tolist()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=256, return_tensors="pt")
            probs[i:i + BATCH] = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
            if (i // BATCH) % 50 == 0:
                print(f"  {i:,}/{len(texts):,}", flush=True)
    for j, l in enumerate(labels):
        df[f"p_{l}"] = probs[:, j]
    df["roberta_score"] = df["p_positive"] - df["p_negative"]  # -1 .. 1, comparable in sign to VADER
    df["roberta_label"] = [labels[k] for k in probs.argmax(1)]
    df.drop(columns=["Response"]).to_csv(os.path.join(TAB, "transformer_sentiment.csv"), index=False)

    from scipy import stats
    rho, _ = stats.spearmanr(df["roberta_score"], df["compound"])
    print(f"\nSpearman(RoBERTa score, VADER compound) over replies: {rho:.3f}")
    g = df.groupby(["Prompt_Variant", "Model"]).agg(
        vader=("compound", "mean"), roberta=("roberta_score", "mean"),
        share_negative=("roberta_label", lambda s: (s == "negative").mean()),
        share_neutral=("roberta_label", lambda s: (s == "neutral").mean()),
        share_positive=("roberta_label", lambda s: (s == "positive").mean()), n=("compound", "size")).reset_index()
    g.to_csv(os.path.join(TAB, "transformer_vs_vader.csv"), index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(g.to_string(index=False))
    # paired model difference within V2, the finding that depends most on the sentiment tool
    w = df[df["Prompt_Variant"] == "v2_empathetic"].pivot_table(index="Row", columns="Model", values="roberta_score")
    d = (w["ChatGPT"] - w["Mistral"]).dropna()
    print(f"\nV2 ChatGPT - Mistral (RoBERTa score): mean {d.mean():.3f}, paired d {d.mean() / d.std(ddof=1):.2f}, "
          f"Wilcoxon p {stats.wilcoxon(d).pvalue:.2g}")
    mir = df.groupby(["Model", "Prompt_Variant"]).apply(
        lambda g: stats.spearmanr(g["roberta_score"], g["narr_compound"])[0]).rename("rho_reply_vs_narrative")
    print("\nMirroring with RoBERTa reply score:\n" + mir.to_string())


if __name__ == "__main__":
    main()
