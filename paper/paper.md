# Prompt beats model: how instruction framing shapes LLM replies to real consumer-finance complaints

*Draft manuscript. Figures are in `analysis/figures/`, tables in `analysis/tables/`; every number is
reproducible from the scripts in `analysis/`. Citations are marked `[cite]` where the author should
insert references; none have been invented.*

## Abstract

Financial institutions are beginning to use large language models (LLMs) to draft first responses
to customer complaints, a regulated communication where tone, concreteness, and factual restraint all
matter. We study how such drafts vary with the choice of model and the wording of the instruction. We
take 10,000 complaints with consumer narratives from the U.S. Consumer Financial Protection Bureau
(CFPB) public database, selected for coverage across product, issue, and sub-issue categories, and
generate a reply to each with two small commercial models (OpenAI `gpt-4o-mini` and Mistral
`mistral-small-latest`) under three prompts: a bare two-sentence instruction, an empathetic
customer-service persona, and a rigid three-field format with compliance constraints. This yields
60,000 replies in a fully crossed, balanced design. We measure length, lexicon and transformer
sentiment, readability, twelve rhetorical markers, instruction compliance, unfilled template
placeholders, and invented facts, and we obtain rubric ratings from two LLM judges on a stratified
subsample.

Three results stand out. First, the prompt explains far more of the variation in a reply than the
model does: 60-72% of the variance in length and sentence count and 47-68% of the variance in
apologising, taking ownership, and committing to a deadline is attributable to the prompt, against
0-13% for the model. Under the rigid format the two models become nearly indistinguishable. Second,
the empathetic prompt is interpreted in opposite emotional registers: ChatGPT becomes uniformly
warm (79% of replies thank the customer; mean VADER compound 0.61) while Mistral restates the
customer's grievance in the customer's own words and commits to a dated action (69% give a
deadline; mean compound -0.07). Lexicon sentiment ranks these backwards from what a compliance
reader would prefer. Third, the dominant deployment failure is not fabrication, which occurs in under
1% of replies, but unfilled template placeholders such as "[Customer's Name]", present in 14% of all
replies and 40% of ChatGPT's empathetic replies; a structured output format reduces this to 3-4%. We
also find that ChatGPT ignores an explicit "3-4 sentences" instruction more than half the time, that
the compliance-constrained format suppresses apologies entirely, and that reply sentiment mirrors the
complaint's own sentiment in every condition. We discuss implications for prompt design and
evaluation in regulated customer communication.

## 1. Introduction

Complaint handling is a high-volume, high-stakes text task. In the United States, the CFPB has
published millions of consumer complaints since 2011, a substantial share with a free-text
narrative written by the consumer, and requires companies to respond within a fixed period [cite].
Institutions are experimenting with LLMs to draft these responses [cite], and vendors market
"empathetic" and "compliant" reply generation. Yet the two design choices a deployer actually
controls, which model to call and how to phrase the instruction, are rarely compared on the same
inputs at scale.

This paper asks a simple question: when an LLM drafts a reply to a real complaint, how much of what
comes back is determined by the model, how much by the prompt, and how much by the complaint itself?
We answer it with a balanced factorial design over 10,000 real complaints, two models, and three
prompts, measuring the replies along dimensions that matter to a compliance reviewer rather than only
to a benchmark.

Our contributions are:

1. A public, reproducible corpus of 60,000 LLM replies to 10,000 CFPB complaints, with the prompts,
   the selection procedure, and per-reply features, extendable to further models.
2. A variance decomposition showing that prompt wording dominates model choice for nearly every
   stylistic and content feature we measure (Section 5.1).
3. Evidence that "empathetic" is not a model-independent instruction: the same persona prompt makes
   one model warmer and the other more concrete and more negative in tone (Section 5.2).
4. A characterisation of failure modes at scale, in which unfilled placeholders, silent length
   non-compliance, and unrequested markdown are far more common than invented facts (Sections
   5.3-5.5).
5. A comparison of lexicon sentiment, transformer sentiment, and LLM-judge rubric ratings as
   evaluation instruments for this task (Section 5.7).

## 2. Related work

*To be expanded by the author.* Relevant strands: LLMs for customer service and complaint response
[cite]; prompt sensitivity and instruction following in LLMs [cite]; empathy in dialogue systems and
its measurement [cite]; sentiment analysis of consumer complaints, including prior work on the CFPB
database [cite]; LLM-as-a-judge evaluation and its biases, including self-preference and verbosity
bias [cite]; and hallucination taxonomies distinguishing fabricated specifics from unsupported
claims [cite].

## 3. Data

### 3.1 Source and selection

We use the CFPB Consumer Complaint Database public export (`complaints.csv`, downloaded September 2026).
From it we keep complaints that (i) have both an `Issue` and a `Sub-issue`, with the sub-issue not
a copy of the issue; (ii) have a consumer narrative of at least 200 characters; (iii) have at most
25% of narrative characters inside the CFPB's `XXXX` redaction tokens; and (iv) are not exact
narrative duplicates. From the resulting pool we select 10,000 complaints by category-capped
sampling: every (`Product`, `Sub-product`, `Issue`, `Sub-issue`) combination receives up to *C*
rows, where *C* is the smallest cap that reaches 10,000, and within a combination the rows with
the highest quality score are taken. The quality score rewards narrative length in the 300-2,500
character range, detailed category labels, and low redaction. The selection therefore over-represents
rare categories relative to the raw database and prevents the dominant credit-reporting categories
from crowding out others; it is not a random sample of complaints and should not be read as one.

The selected set spans eight product families (Table 1), 60 distinct issues, 266 distinct sub-issues, and 1,957 distinct
(`Product`, `Sub-product`, `Issue`, `Sub-issue`) combinations. Narratives have a median length of 663
characters; the median redaction ratio is 0.

**Table 1. Composition of the 10,000-complaint set.**

| Product family | Complaints |
| --- | ---: |
| Debt collection | 3,140 |
| Credit card / prepaid card | 1,898 |
| Mortgage | 1,677 |
| Checking / savings account | 772 |
| Vehicle loan or lease | 675 |
| Credit reporting | 627 |
| Student loan | 616 |
| Payday / personal loan | 595 |

### 3.2 What the models see

Each complaint is rendered as three lines: `Issue:`, `Sub-issue:` (omitted when blank or identical
to the issue), and `Complaint:` followed by the narrative, truncated at a word boundary to 1,500
characters. The company name, the CFPB's recorded outcome, dates, and all other metadata are withheld.
Because the CFPB redacts every dollar amount as `{$1,234.00}`, every date as `XX/XX/XXXX`, and every
name as `XXXX`, the input contains no calendar dates, no names, and no company identity unless the
consumer wrote it into the narrative. This property makes fabrication detectable (Section 4.4).

## 4. Method

### 4.1 Prompts

Three prompts wrap the same complaint text; they differ only in framing (full text in Appendix A).

- **V1 terse.** "Respond to the following consumer complaint in 2 sentences." No persona, no tone
  guidance. The baseline. `max_tokens` = 150.
- **V2 empathetic.** A persona ("an experienced customer service representative at a financial
  institution, known for being warm and genuinely helpful"), an instruction to reply in 3-4
  sentences, acknowledge the frustration, take ownership on behalf of the institution, describe one
  concrete next step, use plain language, and not ask the customer to repeat information. `max_tokens`
  = 300.
- **V3 structured.** The same persona "operating under CFPB complaint-handling rules", a fixed
  three-field format (`Acknowledgement:` / `Next step:` / `What we need from you:`), and explicit
  constraints: do not promise a specific outcome, do not admit legal liability, do not give legal or
  tax advice, do not invent account numbers, dates, or dollar amounts. `max_tokens` = 350.

### 4.2 Models

`gpt-4o-mini` (OpenAI) and `mistral-small-latest` (Mistral AI), called through their public APIs in
September 2026 at default sampling settings, one sample per prompt. A third model, Anthropic's
`claude-sonnet-5`, is part of the pipeline but its replies were not available at the time of
writing. All 60,000 calls succeeded; no reply hit its token cap.

### 4.3 Reply measures

For every reply we compute:

- **Length**: characters, words, sentences (regex splitter).
- **Sentiment**: VADER compound, positive, negative, and neutral shares [cite]; on a 3,000-complaint
  subsample (18,000 replies) also the positive-minus-negative probability from a RoBERTa sentiment
  classifier (`cardiffnlp/twitter-roberta-base-sentiment-latest`) [cite]. The same VADER score is
  computed for the complaint narrative.
- **Readability**: Flesch reading ease and Flesch-Kincaid grade (`textstat`).
- **Rhetorical markers**: twelve case-insensitive regular expressions, after normalising typographic
  apostrophes, for apology, empathy / acknowledgement, ownership, time-bound commitment, hedging,
  requests for information, escalation, promised outcomes, mention of a regulator or credit bureau,
  markdown formatting, thanking the customer, and an unfilled `[placeholder]`. Patterns are listed
  in Appendix B.
- **Instruction compliance**: V1 exactly two sentences; V2 three or four sentences; V3 all three
  labels present.
- **Invented facts** (Section 4.4).

### 4.4 Invented-facts audit

Because the input contains no dates, names, phone numbers, or company identity, a reply containing a
calendar date, a phone number, or a salutation with a personal name has invented it. A dollar amount
is counted as invented unless it matches an amount in the narrative. We report each flag and their
union. Relative deadlines ("within 5 business days") are counted separately as commitments, not
fabrications. Company names are reported descriptively only, as token matching against the CFPB's
company field proved too noisy to attribute.

### 4.5 LLM-judge rubric

On a stratified subsample of 300 complaints (proportional by product family, fixed seed), all six
replies to each complaint (1,800 replies) are rated by two judges, `gpt-4o-mini` and
`mistral-small-latest`, at temperature 0. Each judge sees the complaint exactly as the reply model
saw it and one reply, without any indication of which model or prompt produced it, and returns
1-5 scores for acknowledgement accuracy, concreteness of the next step, tone, grounding (absence of
invented specifics), and overall send-ability, plus booleans for the presence of a placeholder, a
promised outcome, and an admission of liability (rubric text in Appendix C). Using both judges lets
us report inter-judge agreement and test for self-preference, i.e. whether each judge scores its own
model's replies higher than the other judge does.

### 4.6 Statistics

The design is fully crossed and balanced: each complaint has exactly one reply per model × prompt
cell. Model comparisons are therefore paired on the complaint within a prompt, and prompt comparisons
are paired on the complaint within a model. For continuous measures we report paired Cohen's *d* and
the Wilcoxon signed-rank test; for binary markers, the exact McNemar test. With 10,000 pairs
essentially every difference is significant at *p* < 0.001, so we emphasise effect sizes and raw
rates. The balanced design also permits an orthogonal decomposition of each feature's total sum of
squares into prompt, model, model × prompt, complaint, and residual components, which we report as
η². To relate reply sentiment to complaint characteristics we fit OLS with model × prompt fixed
effects, product family, narrative sentiment, log narrative length, and redaction ratio, with standard
errors clustered by complaint.

## 5. Results

### 5.1 The prompt explains more than the model

Figure 9 and Table 2 give the variance decomposition. The prompt accounts for 60% of the variance in
reply length, 72% in sentence count, 68% in whether the reply apologises, 53% in whether it takes
ownership, and 47% in whether it commits to a deadline. The model's main effect never exceeds 13%
(thanking the customer) and is below 3% for length, sentences, readability, apology, and ownership.
The model × prompt interaction is comparable to or larger than the model main effect for sentiment
(11% vs 4%), sentence count (9% vs 3%), and thanking (14% vs 13%): the models differ mostly in how
they respond to a particular prompt, not in a fixed house style. The complaint itself explains 27%
of sentiment variance and 21% of escalation variance, but only 8-17% of the other features.

**Table 2. Share of variance (η²) explained by each source. Balanced design, 60,000 replies.**

| Feature | Prompt | Model | Model × Prompt | Complaint | Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Length (chars) | 0.60 | 0.01 | 0.01 | 0.10 | 0.27 |
| Sentences | 0.72 | 0.03 | 0.09 | 0.03 | 0.13 |
| VADER compound | 0.09 | 0.04 | 0.11 | 0.27 | 0.48 |
| Flesch-Kincaid grade | 0.33 | 0.00 | 0.05 | 0.17 | 0.44 |
| Apology | 0.68 | 0.00 | 0.00 | 0.08 | 0.24 |
| Ownership | 0.53 | 0.00 | 0.00 | 0.08 | 0.39 |
| Time-bound commitment | 0.47 | 0.08 | 0.06 | 0.07 | 0.32 |
| Escalation | 0.24 | 0.03 | 0.00 | 0.21 | 0.51 |
| Asks for information | 0.30 | 0.00 | 0.00 | 0.15 | 0.54 |
| Thanks the customer | 0.18 | 0.13 | 0.14 | 0.13 | 0.42 |
| Leaves a [placeholder] | 0.08 | 0.01 | 0.03 | 0.15 | 0.72 |

The paired effect sizes tell the same story from the other direction (Table 3). Changing the prompt
moves length, sentence count, and readability by one to eight standard deviations; changing the model
within a prompt moves them by 0.02 to 1.4. Under V3 the two models are indistinguishable on length,
sentence count, and grade level (*d* ≤ 0.04): a strict output schema erases most model-level
stylistic variation.

**Table 3. Paired Cohen's *d* for prompt changes (within model) and model changes (within prompt).**

| Comparison | Length | Sentences | VADER compound | FK grade |
| --- | ---: | ---: | ---: | ---: |
| Prompt: V2 vs V1, ChatGPT | 2.96 | 3.98 | 0.18 | -2.12 |
| Prompt: V3 vs V1, ChatGPT | 1.60 | 8.07 | -1.18 | -1.43 |
| Prompt: V2 vs V1, Mistral | 1.87 | 2.10 | -0.50 | -0.68 |
| Prompt: V3 vs V1, Mistral | 1.20 | 2.16 | -0.28 | -0.64 |
| Model: ChatGPT vs Mistral, V1 | 0.19 | -0.41 | 0.39 | 0.48 |
| Model: ChatGPT vs Mistral, V2 | 0.75 | 1.40 | 1.06 | -0.59 |
| Model: ChatGPT vs Mistral, V3 | 0.02 | -0.01 | -0.50 | 0.04 |

### 5.2 "Empathetic" is not a model-independent instruction

The V2 prompt is where the models diverge (Figures 2-4; Table 4). Both apologise (96%, 100%) and
both take ownership (98%, 99%), so both followed the instruction as written. But ChatGPT's replies
are uniformly warm: 79% thank the customer, the mean VADER compound is 0.61 (median 0.79), and only 6%
name a deadline. Mistral's replies are neutral to negative in lexicon sentiment (mean -0.07, median
-0.09), thank the customer 7% of the time, escalate to a named team in 70% of replies, and give a
time-bound commitment in 69%.

**Table 4. V2 empathetic prompt: selected measures by model. All differences *p* < 0.001 (paired).**

| Measure | ChatGPT | Mistral |
| --- | ---: | ---: |
| VADER compound, mean | 0.61 | -0.07 |
| Thanks the customer | 79% | 7% |
| Escalates to a team / specialist | 50% | 70% |
| Time-bound commitment | 6% | 69% |
| Apologises | 96% | 100% |
| Takes ownership | 98% | 99% |
| Follows "3-4 sentences" | 46% | 99% |
| Mean sentences | 4.6 | 3.5 |
| Contains a [placeholder] | 40% | 15% |

Reading the replies explains the sentiment gap. Mistral restates the grievance in the customer's own
vocabulary ("closed without warning", "repeated calls", "potential FDCPA violations") and then commits
to a dated action; ChatGPT reassures ("I want to assure you", "thank you for your patience") and closes
warmly. A lexicon scores the first as negative and the second as positive. The RoBERTa classifier
[Section 5.7: fill in] and the judge ratings [Section 5.7: fill in] indicate whether this ordering
survives instruments that see context. In the OLS the Mistral × V2 interaction is -0.44 compound
points, the largest coefficient after the V3 format effect (Appendix Table C1).

### 5.3 Length compliance is model-specific

V2 asks for three to four sentences. ChatGPT complies in 46% of replies and writes five or more
sentences in most of the rest (mean 4.6 sentences, 496 characters); Mistral complies in 99% (mean 3.5
sentences, 436 characters). No reply in the corpus reached its token cap, so this is a choice rather
than truncation (Figure 6). On V1 ("2 sentences") the pattern reverses mildly: ChatGPT 99%, Mistral
83%. On V3 both produce the three-label format in 99.7-100% of replies. Reply length also tracks
narrative length (Spearman ρ up to 0.42 for ChatGPT V2) even though the instruction fixed the sentence
count.

### 5.4 The dominant failure mode is the unfilled placeholder

Neither prompt asked for a letter, but the models often wrote one, with bracketed slots they never
filled: `[Customer's Name]` in 3,849 replies, `[Your Name]` in 2,135, `[specific date, e.g., Friday]`
in 729, then `[Your Position]`, `[phone number]`, and `[insert contact information]`. Overall 14% of
replies contain at least one (Table 5). ChatGPT's V2 replies are the worst case at 40%: the persona
prompt triggers a letter register with a salutation and sign-off. Mistral's characteristic slip is the
follow-up date, i.e. it "commits" to a date it then leaves as a placeholder. The V3 format almost
eliminates the problem for both models.

**Table 5. Share of replies with an unfilled `[placeholder]`.**

| | ChatGPT | Mistral |
| --- | ---: | ---: |
| V1 terse | 12% | 12% |
| V2 empathetic | 40% | 15% |
| V3 structured | 3% | 4% |

By contrast, hard fabrication is rare (Table 6). Across all 60,000 replies, 0.04% contain a calendar
date, 0.03% a personal name in the salutation, 0.27% a phone number, and 0.52% a dollar amount not
present in the complaint; the union is 0.85%. The two visible pockets are Mistral's V1 replies, which
insert a customer-service telephone number for the institution the consumer named in 1.6% of cases
(e.g. "Please contact Citi customer service directly at 1-800-…"), and Mistral's V2 replies, which
state a specific refund amount not in the narrative in 1.6% of cases. Both models name the
institution when the consumer mentioned it (12-29% of replies), which is grounded rather than
invented but may be undesirable in a reply that is supposed to come from that institution.

**Table 6. Invented-facts audit, share of replies.**

| | Date | Phone | Name in salutation | Invented amount | Any | Relative deadline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ChatGPT V1 | 0.0% | 0.0% | 0.1% | 0.1% | 0.2% | 0.1% |
| ChatGPT V2 | 0.0% | 0.0% | 0.0% | 0.3% | 0.3% | 5.6% |
| ChatGPT V3 | 0.2% | 0.0% | 0.0% | 0.2% | 0.4% | 82.9% |
| Mistral V1 | 0.0% | 1.6% | 0.1% | 0.6% | 2.2% | 6.5% |
| Mistral V2 | 0.0% | 0.0% | 0.0% | 1.6% | 1.6% | 65.5% |
| Mistral V3 | 0.0% | 0.0% | 0.0% | 0.4% | 0.4% | 91.5% |

### 5.5 The compliance-constrained format changes what gets said

Relative to V1, the V3 format (Figure 3):

- **Removes apologies entirely.** 70% (ChatGPT) and 66% (Mistral) of V1 replies apologise; 0% of V3
  replies do. The format did not forbid apologies; the "do not admit legal liability" constraint
  appears to have been generalised to any expression of regret.
- **Makes deadlines near-universal.** Time-bound commitments rise from 0.4% / 13% to 83% / 95%. The
  "by when" in the `Next step` label is enough.
- **Suppresses requests for information.** 45-49% of V1 replies ask the customer for something;
  0.5-9% of V3 replies do. ChatGPT answers "Nothing at this time" in 99% of V3 replies; Mistral does so
  in 62% and asks for a document in the rest.
- **Triggers markdown in one model.** Mistral renders the three labels in bold in 98% of V3 replies;
  ChatGPT never does. In a plain-text channel this appears as literal asterisks.
- **Is redundant on outcomes.** Promised outcomes ("will be refunded", "we will remove") occur in
  ≤ 0.4% of replies in every condition, including V1, so the constraint against them changes nothing.
  Hedging language is also rare (2-6%).

### 5.6 Replies mirror the complaint, and product matters

Reply sentiment correlates with narrative sentiment in every cell (Spearman ρ 0.18-0.39, all
*p* < 0.001; Figure 4), most strongly under V3 (0.39 ChatGPT, 0.38 Mistral) because the
`Acknowledgement` line restates the complaint. Across quintiles of narrative sentiment the mean
ChatGPT reply moves from 0.20 (most negative complaints) to 0.51 (least negative), and the Mistral
reply from -0.06 to 0.36. Mistral escalates more often for the angriest complaints (42% in the most
negative quintile vs 35% in the least); ChatGPT's escalation rate is flat at about 20%.

Holding model, prompt, and narrative sentiment constant, debt-collection complaints receive the most
negative replies (OLS coefficient -0.09 vs checking / savings; credit reporting and payday / personal
loan +0.11). Mistral's mean reply sentiment on debt collection is below zero (-0.04) while ChatGPT's is
0.20 (Figure 5). Debt-collection narratives describe harassment and threats, and Mistral's tendency to
restate the problem carries that vocabulary into the reply.

The CFPB's recorded outcome for each complaint (closed with explanation, monetary relief, non-monetary
relief) has no relationship with reply sentiment, apology rate, or promised outcomes (Appendix
Table C2). The models cannot see the outcome, so this is expected; it is a useful null showing that
the reply carries no information about whether the complaint was upheld.

Both models write at a college reading level when unconstrained (V1 Flesch-Kincaid grade 13.9
ChatGPT, 12.6 Mistral). The V2 request for plain language brings ChatGPT to grade 9.7 and Mistral to
10.9; V3 puts both near grade 11 (Figure 8). Only ChatGPT under V2 approaches the grade 8-9 level
usually recommended for consumer communications [cite].

### 5.7 What the judges say, and how the instruments disagree

**Rubric ratings.** Table 7 and Figure 10 give the mean rubric scores from the two blind LLM judges
on the 300-complaint subsample (1,794 replies with both ratings; six calls failed to parse). The
judges agree on the shape of the results: both prompts improve on V1 on every criterion except
grounding (paired *d* 0.3-1.6); Mistral's V2 replies are the best cell for acknowledgement,
concreteness, tone, and overall quality under both judges (overall 4.21 and 3.99 out of 5); and under
V3 the two models are rated within 0.15 of each other, mirroring the collapse of stylistic
differences in Section 5.1. ChatGPT's V2 replies, the warmest by lexicon sentiment, are rated well
below Mistral's on concreteness (3.18 vs 4.12 by the GPT judge; 2.62 vs 4.19 by the Mistral judge).

**Table 7. Mean rubric scores (1-5) and flag rates by judge, prompt, and model. n = 299 per cell.**

| Judge | Prompt | Model | Acknowl. | Concrete. | Tone | Grounding | Overall | Placeholder | Promises outcome | Admits liability |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4o-mini | V1 | ChatGPT | 3.32 | 2.34 | 4.45 | 5.00 | 2.96 | 26% | 3% | 10% |
| gpt-4o-mini | V1 | Mistral | 3.70 | 2.93 | 4.61 | 4.98 | 3.36 | 17% | 16% | 22% |
| gpt-4o-mini | V2 | ChatGPT | 4.20 | 3.18 | 4.97 | 4.99 | 3.80 | 43% | 12% | 10% |
| gpt-4o-mini | V2 | Mistral | 4.71 | 4.12 | 5.00 | 4.99 | 4.21 | 17% | 49% | 52% |
| gpt-4o-mini | V3 | ChatGPT | 4.56 | 3.24 | 4.75 | 5.00 | 3.72 | 3% | 2% | 1% |
| gpt-4o-mini | V3 | Mistral | 4.44 | 3.23 | 4.70 | 4.99 | 3.69 | 8% | 2% | 1% |
| mistral-small | V1 | ChatGPT | 2.81 | 2.11 | 4.12 | 4.59 | 2.46 | 11% | 1% | 0% |
| mistral-small | V1 | Mistral | 3.36 | 2.94 | 4.39 | 4.66 | 3.10 | 8% | 9% | 2% |
| mistral-small | V2 | ChatGPT | 3.17 | 2.62 | 4.38 | 3.97 | 3.02 | 40% | 2% | 0% |
| mistral-small | V2 | Mistral | 4.20 | 4.19 | 4.84 | 4.33 | 3.99 | 16% | 21% | 18% |
| mistral-small | V3 | ChatGPT | 3.93 | 3.52 | 4.50 | 4.98 | 3.51 | 3% | 0% | 0% |
| mistral-small | V3 | Mistral | 3.81 | 3.38 | 4.40 | 4.96 | 3.38 | 5% | 0% | 0% |

**The concreteness that the judges reward comes with a compliance cost.** The same Mistral V2
replies that score highest overall are flagged by the judges as promising a specific outcome in
21-49% of cases and as admitting liability in 18-52% ("I take full ownership of the miscommunication",
"it's completely unacceptable that you're not getting the full refund you're owed"). ChatGPT's V2
replies, and both models' V3 replies, are flagged for these in 0-12% of cases. The regex marker for
promised outcomes (Section 5.5) caught ≤ 0.4% because it looked for explicit "will be refunded"
phrasing; the judges read "ensure the refund is processed" as a promise. The V3 constraints work as
intended here: both flags fall to ≤ 2% under V3 for both models.

**Grounding.** The GPT judge rates grounding at 4.98-5.00 in every cell and is effectively
insensitive. The Mistral judge is stricter and penalises V2 (3.97 ChatGPT, 4.33 Mistral vs 4.6-5.0
elsewhere) for unsupported specifics such as "I'll escalate this to our compliance team today". This
is consistent with the regex audit, which found almost no hard fabrication (Section 5.4): what the
stricter judge is penalising is invented process, not invented facts.

**Agreement and self-preference.** Inter-judge Spearman correlations are 0.71 for concreteness,
0.59 for acknowledgement, 0.58 for overall, 0.38 for tone, and 0.06 for grounding (Table 8). On the
placeholder flag the judges agree with each other (κ = 0.81) and with the regex marker (κ = 0.82 and
0.92), which validates the regex used in Section 5.4. Agreement on the compliance flags is weaker
(κ = 0.52 for promised outcomes, 0.31 for admitted liability), with the GPT judge flagging two to
four times as often as the Mistral judge.

Each judge does favour its own family: on the paired ChatGPT-minus-Mistral difference, the GPT
judge is 0.23 points more favourable to ChatGPT than the Mistral judge is (overall; *d* = 0.27,
*p* < 0.001), with similar shifts on every criterion. But the self-preference shifts the size of the
gap, not its direction: both judges still rate Mistral's replies higher overall (GPT judge -0.26,
Mistral judge -0.50). The Mistral judge is also harsher in general (its scores average 0.4 points
lower).

**Table 8. Inter-judge agreement and self-preference.**

| Criterion | Spearman ρ (judges) | Within 1 point | ChatGPT − Mistral, GPT judge | ChatGPT − Mistral, Mistral judge | Self-preference shift (*d*) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acknowledgement | 0.59 | 84% | -0.26 | -0.49 | 0.22 |
| Concreteness | 0.71 | 99% | -0.50 | -0.76 | 0.28 |
| Tone | 0.38 | 100% | -0.04 | -0.21 | 0.23 |
| Grounding | 0.06 | 84% | 0.00 | -0.14 | 0.13 |
| Overall | 0.58 | 95% | -0.26 | -0.50 | 0.27 |

**Sentiment is anti-correlated with judged quality.** Across the 1,800 rated replies, the VADER
compound correlates *negatively* with the judges' overall score (Spearman -0.14 GPT judge, -0.26
Mistral judge) and with concreteness (-0.24, -0.30). Length correlates positively with judged
quality (0.49 and 0.27 for overall). The lexicon instrument therefore ranks replies in roughly the
opposite order from a rubric-based judge on this task, and the GPT judge's length correlation is
large enough that verbosity bias [cite] cannot be excluded as part of the reason V2 replies score
well.

**Transformer sentiment.** *[To be completed from `analysis/tables/transformer_vs_vader.csv`.]*

## 6. Discussion

**Prompt engineering is the main lever, and it is not neutral.** For a deployer choosing between two
small commercial models, the wording of the instruction changes the reply far more than the model
does. This is good news for portability but bad news for the assumption that a prompt validated on one
model will behave the same on another: the empathetic prompt produced two quite different products.

**Sentiment is the wrong yardstick for empathy.** The instrument that most analyses of "empathetic"
AI reach for, lexicon sentiment, ranked the reply that restated the customer's problem and committed
to a date below the reply that thanked the customer and promised to be in touch. A rubric that scores
acknowledgement accuracy and concreteness separately from tone is needed, and Section 5.7 shows how
far the instruments diverge.

**Constraints generalise.** "Do not admit liability" removed every apology. Deployers who want an
apology and no admission need to say both.

**The realistic risk is not hallucination.** With inputs that redact every date, name, and amount,
the models almost never invented one. What they did, in one reply out of seven, was leave a template
slot unfilled, and in a fraction of cases insert a customer-service telephone number from memory.
Both are trivially detectable with a regular expression; neither was guarded against by any of the
three prompts. The structured format, which prevents most placeholders, is the cheapest mitigation we
observed.

## 7. Limitations

- Two models in the same small, inexpensive tier; results may not transfer to frontier models.
  The pipeline is model-agnostic and the Claude arm is pending.
- One sample per prompt at default temperature; within-model variance is unmeasured.
- Regular-expression markers count surface phrases, not intent, and are sensitive to details such as
  typographic apostrophes (a bug we found and fixed). A manual audit of a few hundred replies per
  marker is planned.
- VADER is a social-media lexicon; the RoBERTa classifier is also trained on tweets. The judge
  ratings are from the same two model families that produced the replies, which raises
  self-preference concerns that we test but cannot fully rule out.
- The complaint set is category-stratified, not representative of complaint volume.
- Everything here is descriptive. No human rated the replies, and we do not know how consumers would
  receive them.

## 8. Conclusion

On 60,000 replies to 10,000 real complaints, the instruction, not the model, determined most of what
an LLM said and how it said it, the same "empathetic" instruction yielded opposite emotional
registers from two models, and the operational risk that actually materialised was the unfilled
placeholder, not the invented fact. The corpus, code, and feature tables are released for further
work.

## Appendix A. Prompt templates

See `PROMPT_VARIANTS` in `generate_llm_responses.py`.

## Appendix B. Marker patterns

See `MARKERS` in `analysis/eda_sentiment.py` and the audit patterns in `analysis/robustness.py`.

## Appendix C. Judge rubric and supplementary tables

Rubric text: `RUBRIC` in `analysis/llm_judge.py`. Table C1: `analysis/tables/ols_reply_sentiment.txt`.
Table C2: `analysis/tables/reply_by_company_outcome.csv`.
