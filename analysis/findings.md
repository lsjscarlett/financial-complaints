# How ChatGPT and Mistral answer real consumer-finance complaints: EDA and sentiment findings

Working notes toward the paper. Every number below comes from `analysis/eda_sentiment.py`
run over the full dataset; tables with confidence intervals and test statistics are in
`analysis/tables/`, figures in `analysis/figures/`.

## 1. Data

| | |
| --- | --- |
| Complaints | 10,000 CFPB complaints with a consumer narrative, selected for category coverage (`dataset/complaints_10k.csv`) |
| Products | Debt collection 3,140; credit card / prepaid 1,898; mortgage 1,677; checking / savings 772; vehicle loan 675; credit reporting 627; student loan 616; payday / personal loan 595 |
| Prompts | V1 terse (2 sentences, no persona), V2 empathetic (persona, acknowledge + own + one next step, 3-4 sentences), V3 structured (persona, fixed 3-label format, compliance constraints) |
| Models | ChatGPT (`gpt-4o-mini`) and Mistral (`mistral-small-latest`), default temperature, one sample per prompt. Claude (`claude-sonnet-5`) pending credits |
| Replies analysed | 60,000 (10,000 × 3 prompts × 2 models); 0 truncated by `max_tokens` |
| Narratives | median 663 characters; VADER compound mean -0.12 (they are complaints, but many read neutral to VADER); median redaction ratio 0 |

Every complaint received all six replies, so model comparisons are paired on the complaint
within a prompt, and prompt comparisons are paired on the complaint within a model. With
10,000 pairs nearly every difference is significant at p < 0.001; the effect sizes
(paired Cohen's d) and the raw rates are what matter.

## 2. Headline findings

### 2.1 The prompt moves the reply far more than the model does

Switching prompts changes reply length, tone, and content by one to eight standard
deviations; switching models within a prompt changes them by a fraction of one.

| Paired Cohen's d | Length (chars) | Sentences | VADER compound | FK grade |
| --- | --- | --- | --- | --- |
| **Prompt** V2 vs V1, ChatGPT | 2.96 | 3.98 | 0.18 | -2.12 |
| **Prompt** V3 vs V1, ChatGPT | 1.60 | 8.07 | -1.18 | -1.43 |
| **Prompt** V2 vs V1, Mistral | 1.87 | 2.10 | -0.50 | -0.68 |
| **Prompt** V3 vs V1, Mistral | 1.20 | 2.16 | -0.28 | -0.64 |
| **Model** ChatGPT vs Mistral, V1 | 0.19 | -0.41 | 0.39 | 0.48 |
| **Model** ChatGPT vs Mistral, V2 | 0.75 | 1.40 | 1.06 | -0.59 |
| **Model** ChatGPT vs Mistral, V3 | 0.02 | -0.01 | -0.50 | 0.04 |

The one exception is the empathetic prompt, where the two models diverge sharply (next
finding). Under the rigid V3 format the two models become nearly indistinguishable on
length, sentence count, and readability (d ≤ 0.04), which is itself a result: a strict
output schema erases most model-level stylistic variation.

### 2.2 The empathetic prompt splits the models: ChatGPT gets warmer, Mistral gets darker

Both models were given the same persona and asked to acknowledge frustration, take
ownership, and give one next step. They complied in opposite emotional registers
(`fig2_sentiment.png`, `fig4_mirroring.png`).

| V2 empathetic | ChatGPT | Mistral |
| --- | --- | --- |
| VADER compound, mean | **0.61** | **-0.07** |
| VADER compound, median | 0.79 | -0.09 |
| Thanks the customer | 79% | 7% |
| Escalates to a team / specialist | 50% | 70% |
| Gives a time-bound commitment | 6% | 69% |
| Apologises | 96% | 100% |
| Takes ownership ("I'll personally", "we will") | 98% | 99% |

Reading the replies explains the sentiment gap: Mistral restates the customer's problem in
the customer's own negative vocabulary ("closed without warning", "repeated calls",
"potential FDCPA violations") and then commits to a dated action, whereas ChatGPT
reassures ("I want to assure you", "thank you for your patience") and closes warmly.
VADER scores the first as negative and the second as positive, but a compliance reader
would likely call Mistral's the more concrete reply. Sentiment alone is therefore a
misleading quality measure here; Section 4 proposes a rubric.

The OLS in `tables/ols_reply_sentiment.txt` puts the interaction at -0.44 compound points
(Mistral × V2), the largest term in the model after the V3 format effect.

### 2.3 ChatGPT ignores the length instruction in V2; Mistral follows it

V2 asks for "3-4 sentences". ChatGPT complies 46% of the time and writes 5 or more
sentences in most of the rest (mean 4.6 sentences, 496 characters). Mistral complies 99%
of the time (mean 3.5 sentences, 436 characters). Neither model was cut off by the token
cap, so this is a choice, not a truncation artefact (`fig6_compliance.png`).

On V1 ("2 sentences") the pattern reverses mildly: ChatGPT 99% compliant, Mistral 83%.
On V3 both hit the three-label format in 99.7-100% of replies.

### 2.4 One reply in seven contains an unfilled template placeholder

Neither prompt asked for a letter, but the models often wrote one, complete with
bracketed slots they never filled: `[Customer's Name]` (3,849 replies), `[Your Name]`
(2,135), `[specific date, e.g., Friday]` (729), `[Your Position]`, `[phone number]`.

| Contains a `[placeholder]` | ChatGPT | Mistral |
| --- | --- | --- |
| V1 terse | 12% | 12% |
| V2 empathetic | **40%** | 15% |
| V3 structured | 3% | 4% |

ChatGPT's V2 replies are the worst case: the persona prompt triggers a letter register
with a salutation and sign-off. Mistral's characteristic slip is the follow-up date, i.e.
it "commits" to a date it then leaves as a placeholder. The structured format almost
eliminates the problem for both models. For deployment this is the most actionable
finding: a placeholder in a customer reply is an obvious failure, and a one-line prompt
constraint or a regex post-filter would catch it.

### 2.5 The structured, compliance-constrained format changes what gets said

V3 replaces the free-text reply with `Acknowledgement / Next step / What we need from you`
plus constraints against promised outcomes, admitted liability, and invented figures.
Relative to V1 (`fig3_markers.png`):

- Apologies vanish: 70% and 66% of V1 replies apologise; 0% of V3 replies do. The format
  did not forbid apologies, but the "no admitted liability" constraint appears to
  suppress them.
- Time-bound commitments jump from 0.4% / 13% (V1) to 83% / 95% (V3). The "by when" in
  the Next step label is enough to make both models name a deadline.
- Requests for information nearly disappear (45-49% in V1 to 0.5-9% in V3). ChatGPT
  answers "Nothing at this time" in 99% of V3 replies; Mistral does so in 62% and asks for
  a document in the rest.
- Mistral renders the labels as markdown bold in 98% of V3 replies; ChatGPT never does.
  Anything pasted into a plain-text channel will show literal asterisks.

Promised outcomes ("will be refunded", "we will remove") are rare in every condition
(≤ 0.4%), so the V3 constraint against them is largely redundant with the models'
default caution. Hedging language is also rare (2-6%).

### 2.6 Replies mirror the complaint's tone, and the models are not equally sensitive to it

Reply sentiment correlates with the complaint narrative's sentiment in every cell
(Spearman ρ 0.18-0.39, all p < 0.001; `fig4_mirroring.png`, `tables/sentiment_mirroring.csv`).
Mirroring is strongest under the structured prompt (ρ 0.39 ChatGPT, 0.38 Mistral),
because the Acknowledgement line restates the complaint. Across narrative-sentiment
quintiles the ChatGPT reply moves from 0.20 (most negative complaints) to 0.51 (least
negative); Mistral from -0.06 to 0.36. Mistral also escalates more often for the angriest
complaints (42% in Q1 vs 35% in Q5); ChatGPT's escalation rate is flat (~20%).

Reply length tracks narrative length too (ρ up to 0.42 for ChatGPT V2), i.e. longer
complaints get longer answers even though the instruction fixed the sentence count.

### 2.7 Product matters, debt collection most of all

Holding model, prompt, and narrative sentiment constant, debt-collection complaints get
the most negative replies (OLS coefficient -0.09 vs checking / savings; credit reporting
and payday / personal loan +0.11). Mistral's mean reply sentiment on debt collection is
below zero (-0.04) while ChatGPT's is 0.20 (`fig5_product.png`). Debt-collection
narratives are the ones that describe harassment and threats, and Mistral's tendency to
restate the problem carries that vocabulary into the reply.

### 2.8 What the company actually did has no effect on the reply

The CFPB records how the company resolved each complaint (explanation, monetary relief,
non-monetary relief). Reply sentiment, apology rate, and promised-outcome rate are flat
across these categories (`tables/reply_by_company_outcome.csv`). The models cannot see the
outcome, so this is expected, but it is a useful null: the LLM reply carries no
information about whether the complaint was legitimate or was upheld.

### 2.9 Readability

Both models write at a college reading level when unconstrained (V1 Flesch-Kincaid grade
13.9 ChatGPT, 12.6 Mistral). The V2 request for "plain language" brings ChatGPT down to
grade 9.7 and Mistral to 10.9; V3 lands both near grade 11 (`fig8_readability.png`).
Only ChatGPT under V2 approaches the grade 8-9 level usually recommended for consumer
communications.

## 3. Threats to validity

- **VADER is a lexicon model built on social-media text.** It rewards words like
  "thank" and "assure" and penalises "frustration" and "closed", regardless of who is
  frustrated. Findings 2.2 and 2.7 are partly an artefact of that. A transformer sentiment
  model or an LLM-as-judge rubric should be run as a robustness check.
- **Regex markers are crude.** They count surface phrases, not intent. The curly-apostrophe
  bug found during this analysis (Mistral writes "I’ll") shows how fragile they are; the
  patterns now normalise quotes, but a manual audit of a few hundred replies per marker is
  needed before publication.
- **One sample per prompt at default temperature.** Within-model variance is unmeasured.
  Re-sampling a subset (say 500 complaints × 5 samples) would give it.
- **Two models, one size class each.** `gpt-4o-mini` and `mistral-small-latest` are both
  small, cheap tiers; results may not transfer to frontier models. Claude is missing
  pending API credits.
- **Sentence counting** uses a regex splitter; replies with sign-offs or bullet fragments
  can be over-counted. The compliance figures for V2 are robust to this (the gap between
  46% and 99% is far larger than any splitter error), but the exact rates are approximate.
- **No ground truth for quality.** Everything here is descriptive. The paper needs a human
  or rubric-based rating of a sample to say which style is better.

## 4. Suggested paper framing and next steps

Working title: *Prompt beats model: how instruction framing shapes LLM replies to real
consumer finance complaints*.

1. **RQ1** How much of the variation in reply style is due to the prompt versus the model?
   (Finding 2.1; variance decomposition on the feature table.)
2. **RQ2** Do models interpret "empathetic" the same way? (2.2, 2.6.)
3. **RQ3** Do structured, compliance-constrained formats suppress desirable behaviour along
   with undesirable behaviour? (2.5: apologies and information requests disappear.)
4. **RQ4** What deployment failure modes appear at scale? (2.4 placeholders, 2.5 markdown,
   2.3 length non-compliance.)

Next analyses, in priority order:

- **Rubric rating** of a stratified sample (e.g. 300 complaints × 6 replies) on
  acknowledgement accuracy, concreteness of the next step, absence of hallucinated facts,
  and tone, by two human raters or an LLM judge with a validated agreement check.
- **Hallucination check**: does the reply mention entities, amounts, or dates not present
  in the complaint? Finding 2.4 hints at invented dates; a named-entity comparison would
  quantify it.
- **Transformer sentiment** (e.g. a RoBERTa sentiment model) as a robustness check on
  2.2 and 2.7.
- **Claude** once credits are available; the pipeline resumes automatically.
- **Sub-issue level analysis**: the 10k set was chosen for category coverage, so per
  issue / sub-issue breakdowns are possible for the larger categories.

## 5. Reproduction

```
pip install -r requirements.txt
python analysis/eda_sentiment.py              # ~10 min: features, stats, figures
ANALYSIS_FROM_CACHE=1 python analysis/eda_sentiment.py   # redo stats + figures only
```
