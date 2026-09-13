# Study 2: which part of the prompt does what

*Candidate section for the paper, drafted separately like `eda.md`. Numbers come from
`analysis/study2_factorial.py`; tables are in `analysis/tables/study2_*.csv`, the figure is
`analysis/figures/fig11_factorial.png`.*

## S2.1 Design

Study 1 compared three realistic prompts that each changed several things at once, so it could
show *that* the prompt dominates but not *which part* of the prompt does the work. Study 2 takes the
empathetic prompt apart. Three binary factors are crossed in a 2×2×2 design, all with the same
three-to-four-sentence target so that length is not confounded:

- **A, empathetic framing**: V2's persona ("an experienced customer service representative …
  known for being warm and genuinely helpful") plus its directives (acknowledge the frustration,
  take ownership, one concrete next step, plain language, do not ask the customer to repeat
  themselves).
- **B, structured format**: the three labelled fields `Acknowledgement / Next step / What we need
  from you`.
- **C, compliance constraints**: do not promise a specific outcome, do not admit legal liability,
  do not give legal or tax advice, do not invent account numbers, dates, or dollar amounts.

The all-off cell (000) is "Reply to the following consumer complaint in 3-4 sentences."; the A00
cell is V2 itself, taken from Study 1. Cells are labelled by the factors that are on (e.g. `A0C`
is framing plus constraints, no format). Two further cells sit outside the cube: a **hygiene**
cell (V2 plus "Write plain text only: no salutation, no sign-off, no markdown, and no bracketed
placeholders such as [Customer's Name] or [date]") and two **surface paraphrases** of V2 that keep
every instruction but reword it.

All ten prompts were run on a 2,999-complaint subset of the Study 1 corpus, stratified by product
family with the same seed as the judge sample (299 of the 300 judge complaints are included), for
both models: 59,980 new replies. Every complaint has all eight cube cells for each model, so
effects are estimated within complaint. We report cell means, and for each feature an OLS of the
form *y ~ A × B × C* with complaint fixed effects and standard errors clustered by complaint.
Because most features are binary and saturate at 0 or 1, the higher-order interaction terms are
large and mostly reflect ceilings and floors; the cell means are the primary evidence and the
regression terms are read as "does adding this factor change anything, given what is already on".

![Figure 11](../analysis/figures/fig11_factorial.png)

**Figure 11. Reply features across the eight cells.** Cells labelled by the factors that are on.

**Table S2.1. Cell means for the features that changed most. Shares in %, n = 2,999 per cell.**

| Cell | Model | Apology | Time-bound | Thanks | Placeholder | Asks info | Ownership | Markdown | VADER | Chars | Sent. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 000 | ChatGPT | 64 | 1 | 89 | **95** | 44 | 50 | 0 | 0.91 | 557 | 5.0 |
| 000 | Mistral | 79 | 34 | 72 | 22 | 42 | 52 | 7 | 0.65 | 482 | 4.5 |
| A00 (V2) | ChatGPT | 97 | 6 | 78 | 40 | 1 | 98 | 0 | 0.60 | 495 | 4.6 |
| A00 (V2) | Mistral | 100 | 68 | 7 | 15 | 1 | 99 | 0 | -0.06 | 435 | 3.5 |
| 0B0 | ChatGPT | 0 | 50 | 2 | 10 | 3 | 98 | 0 | -0.06 | 389 | 3.0 |
| 0B0 | Mistral | 0 | 96 | 0 | 18 | 42 | 98 | **91** | 0.15 | 448 | 3.0 |
| 00C | ChatGPT | 4 | 0 | 99 | 14 | 20 | 15 | 0 | 0.90 | 444 | 4.1 |
| 00C | Mistral | 14 | 3 | 98 | 3 | 17 | 46 | 0 | 0.75 | 358 | 4.1 |
| AB0 | ChatGPT | 5 | 69 | 9 | 3 | 0 | 99 | 4 | -0.18 | 426 | 3.2 |
| AB0 | Mistral | 19 | 75 | 0 | 7 | 0 | 100 | 48 | -0.20 | 410 | 3.0 |
| A0C | ChatGPT | 90 | 0 | 77 | 25 | 0 | 97 | 0 | 0.61 | 458 | 4.3 |
| A0C | Mistral | 100 | 43 | 9 | 6 | 0 | 98 | 0 | -0.07 | 395 | 3.3 |
| 0BC | ChatGPT | 0 | 65 | 2 | 3 | 1 | 98 | 0 | -0.03 | 376 | 3.0 |
| 0BC | Mistral | 0 | 93 | 0 | 15 | 25 | 99 | 33 | 0.14 | 363 | 3.0 |
| ABC | ChatGPT | 1 | 62 | 4 | 1 | 0 | 99 | 2 | -0.26 | 404 | 3.1 |
| ABC | Mistral | 11 | 68 | 0 | 6 | 0 | 100 | 25 | -0.19 | 385 | 3.0 |

## S2.2 Findings

### The bare baseline is the worst cell, not the empathetic one

The all-off cell answers a question Study 1 could not: what does a small model do with a
complaint when told only how long to be? ChatGPT writes a letter. 95% of its 000 replies contain an
unfilled placeholder, 89% thank the customer, and the mean length (557 characters, 5.0 sentences)
is the longest of any cell despite the "3-4 sentences" instruction. It also ignores the sentence
target more than under V2 (5.0 vs 4.6), so Study 1's finding that ChatGPT over-writes under V2 was
not the persona's fault: without the persona it over-writes more. Mistral's 000 replies are
shorter and less templated (22% placeholders) but still ask the customer for information 42% of
the time and commit to a deadline only 34% of the time.

Every factor reduces placeholders from that baseline. Empathetic framing alone brings ChatGPT from
95% to 40% (the V2 rate from Study 1), constraints alone to 14%, the format alone to 10%, and any
two factors together to 1-3%. The main-effect estimates are -0.55 (A), -0.85 (B), and -0.81 (C)
for ChatGPT, with large positive interactions because the floor is reached. The persona was
mitigating the problem in Study 1, not causing it; the comparison that made V2 look bad was
against V1, whose two-sentence budget leaves no room for a salutation.

### Apologies are removed by the format and by the constraints independently, and restored by the framing

Study 1 found that V3 contained no apologies and could not say whether the format or the
"do not admit liability" constraint was responsible. Study 2 says both are sufficient on their own
and neither is necessary. The format alone drives apologies from 64-79% to 0% for both models
(B main effect -0.64 and -0.79). The constraints alone drive them to 4% and 14% (C main effect
-0.60 and -0.65). But the empathetic framing overrides the constraints: with A on, adding C leaves
apologies at 90% (ChatGPT) and 100% (Mistral), an A × C interaction of +0.54 and +0.65. The framing
does not override the format: with B on, apologies stay at 0-19% whether or not A is present. The
Study 1 result was therefore a format effect. A deployer who wants an apology under a three-field
format has to put an apology field in the format; asking for empathy is not enough.

### The constraints alone make replies less concrete, not more careful

The constraints cell without framing or format (00C) is the least useful cell in the design. For
ChatGPT, ownership falls from 50% to 15%, time-bound commitments to 0%, escalation to 1%, and
thanks rise to 99%; hedging rises from 12% to 28%, the highest in the cube. Replies become
"Thank you for bringing this to our attention. We take your concerns seriously and will review
your account … we cannot comment on specific outcomes." Mistral shows the same pattern with smaller
magnitudes. Told what not to say and nothing about what to say, both models retreat into
boilerplate. The constraints only work as intended in combination: with A or B on, they cost
little (ownership and deadlines are unchanged) and the promise-of-outcome rate, already under 1%
everywhere, stays there.

### The Study 1 model split is the framing factor, and it is Mistral's reaction to it

Under 000 the two models are close on tone (VADER 0.91 vs 0.65) and both thank the customer
(89% vs 72%). Turning on A moves ChatGPT modestly (VADER -0.31, thanks -12 points) and Mistral
dramatically (VADER -0.71, thanks -65 points, escalation +48, time-bound +35). The A main effect
on sentiment is the largest single-factor effect for Mistral in the design. So the "empathetic
prompt splits the models" finding of Study 1 is confirmed as a framing effect, and it is
specifically Mistral that reads "acknowledge, own, one concrete next step" as an instruction to
restate the grievance and commit, while ChatGPT reads it as an instruction to reassure. The
format factor then closes the gap again: under any B cell the models' sentiment, thanks, and
deadline rates are within a few points of each other.

### The format drives deadlines; markdown is a Mistral-plus-format artefact that constraints dampen

Time-bound commitments are a format effect (B main effect +0.50 ChatGPT, +0.62 Mistral) with a
Mistral-specific framing contribution (+0.35). Under the format the models differ mainly in
degree: Mistral names a deadline in 93-96% of format replies without framing, ChatGPT in 50-65%.

Markdown is nearly absent in every free-text cell for both models and appears only when Mistral
sees the labelled format: 91% under the plain format, 48% when framing is added, 33% when
constraints are added, and 25% with both. The constraints paragraph, which says nothing about
formatting, halves Mistral's bolding; this is the kind of side effect that only a factorial design
exposes.

### Requests for information are suppressed by framing and, for ChatGPT, by the format

44% and 42% of 000 replies ask the customer for something. The framing removes that almost
entirely (A main effect -0.43 and -0.42), consistent with its "do not ask the customer to repeat
information" directive. The format removes it for ChatGPT (3%) but not for Mistral (42%), because
Mistral fills the `What we need from you` field with a request while ChatGPT writes "Nothing at
this time"; Study 1 saw the same asymmetry under V3.

## S2.3 The hygiene cell: one line fixes the placeholder problem, at a price

Adding "plain text only, no salutation, no sign-off, no markdown, no bracketed placeholders" to
V2 works as a mitigation (Table S2.2). ChatGPT's placeholder rate goes from 40% to 0.0% and its
salutation rate from 50% to 0%; Mistral's placeholders go from 15% to 2%. The XXXX echo, in which
the redacted name is copied into a greeting, disappears with the greeting.

The price is tone and, for Mistral, concreteness. ChatGPT's replies lose the letter register they
were built around: thanks fall from 78% to 42%, apologies from 97% to 83%, VADER from 0.60 to 0.31,
and length by 51 characters. Mistral's replies get slightly more negative (-0.06 to -0.20) and its
time-bound commitments fall from 68% to 53%, which suggests that some of Mistral's deadlines lived
in sign-off lines ("I'll follow up by [date]") that the hygiene instruction removed along with the
placeholder. A deployer who adds this line should expect to re-tune the tone instructions.

**Table S2.2. Hygiene cell vs V2, paired on the complaint. n = 2,999.**

| | ChatGPT V2 | ChatGPT hygiene | Mistral V2 | Mistral hygiene |
| --- | ---: | ---: | ---: | ---: |
| Placeholder | 40.2% | 0.0% | 15.3% | 2.0% |
| Salutation | 50.3% | 0.0% | 0.7% | 0.0% |
| Sign-off | 22.5% | 14.4% | 5.8% | 9.7% |
| Thanks the customer | 77.6% | 41.6% | 7.3% | 0.6% |
| Apology | 96.5% | 82.9% | 99.8% | 92.6% |
| Time-bound commitment | 6.1% | 3.4% | 68.4% | 53.0% |
| VADER compound | 0.60 | 0.31 | -0.06 | -0.20 |
| Length (chars) | 495 | 444 | 435 | 419 |

## S2.4 Paraphrase noise: rewording moves replies as much as switching models did

Two paraphrases of V2 preserve every instruction and change only the wording. They should, if
prompts are read for content, produce replies indistinguishable from V2. They do not
(Table S2.3).

For ChatGPT, both paraphrases raise the placeholder rate from 40% to 95% and 81%. The trigger is
visible in the wording: "Write your response to them" and "Answer them directly" cue a letter
with a salutation, where V2's "Reply directly to them in 3-4 sentences" cues a message. Both
paraphrases also lengthen replies (paired *d* 0.64 and 0.89) and raise sentiment (0.39 and 0.38).
For Mistral the paraphrases pull in opposite directions: the first shortens replies (*d* -0.45),
lowers sentiment by 0.29, and cuts time-bound commitments by 32 points; the second lengthens
replies (*d* 0.75) and leaves sentiment and deadlines closer to V2. The two paraphrases differ from
each other by *d* = 1.15 on Mistral's reply length and by 20 points on its deadline rate.

For comparison, the ChatGPT-vs-Mistral model effect within V2 in Study 1 was *d* = 0.75 on length
and 1.06 on sentiment. Surface wording, holding content fixed, moves the same features by
0.4-1.2 standard deviations. This has two consequences. Study 1's factor labels ("V2 empathetic")
denote a particular wording as much as a particular content, and any single-prompt comparison of
models is confounded with how each model happens to read that wording. And the cheapest lever a
deployer has is not the model or even the instruction content but the phrasing, which is also the
least principled one.

**Table S2.3. Paraphrases of V2, paired differences from V2 and from each other. n = 2,999.**

| Feature | Model | Para 1 − V2 | Para 2 − V2 | Para 2 − Para 1 |
| --- | --- | ---: | ---: | ---: |
| Placeholder (points) | ChatGPT | +55 | +41 | -14 |
| Placeholder (points) | Mistral | +2 | -1 | -2 |
| Length (chars, *d*) | ChatGPT | +46 (0.64) | +67 (0.89) | +21 (0.28) |
| Length (chars, *d*) | Mistral | -37 (-0.45) | +69 (0.75) | +105 (1.15) |
| VADER compound (*d*) | ChatGPT | +0.18 (0.39) | +0.17 (0.38) | -0.01 |
| VADER compound (*d*) | Mistral | -0.29 (-0.46) | -0.05 (-0.08) | +0.23 (0.37) |
| Time-bound (points) | ChatGPT | -4 | -4 | 0 |
| Time-bound (points) | Mistral | -32 | -12 | +20 |
| Escalation (points) | ChatGPT | -21 | -5 | +17 |
| Escalation (points) | Mistral | -25 | -11 | +14 |
| Thanks (points) | ChatGPT | +15 | +1 | -14 |

## S2.5 What Study 2 changes in the paper's claims

1. **"Prompt beats model" stands, with a sharper mechanism.** The framing, format, and constraint
   factors each move most features by tens of points; the model main effect stays small. But a
   third of that prompt effect is wording, not content (S2.4), which the paper must say.
2. **The placeholder finding reverses in attribution.** The persona reduces placeholders; the bare
   instruction and the paraphrases produce the most. The deployment advice becomes: never ship a
   length-only prompt, and add the hygiene line.
3. **The apology finding is a format effect**, not a constraint effect, and the empathetic framing
   overrides the "no liability" constraint on apologies but not the format.
4. **Constraints without positive instructions backfire** into hedged boilerplate.
5. **The empathy split is a framing effect specific to Mistral**, and the format erases it.
6. **Judged quality is not yet measured for Study 2 cells.** Running the two judges over the 299
   overlapping complaints for the eight cube cells (about 4,800 ratings per judge) would let the
   paper say which cell a rubric prefers, and whether the constraints' cost in concreteness is
   visible to a judge. This is the natural next step and reuses `analysis/llm_judge.py` unchanged
   except for the variant list.
