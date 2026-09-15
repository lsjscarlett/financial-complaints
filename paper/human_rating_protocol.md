# Human rating pass: protocol

The purpose is to anchor the LLM judges. The paper currently rests on two LLM judges from the same
families that produced the replies; a human pass on a subsample lets it report (a) inter-rater
agreement among humans, (b) agreement between each LLM judge and humans, criterion by criterion,
(c) which automatic instrument (VADER, RoBERTa, GPT judge, Mistral judge) best predicts human
overall quality, and (d) whether the model gap and the cell ordering the judges found survive human
rating. Everything below is designed so that those four numbers come out of the analysis script
unchanged.

## 1. What gets rated

**Sample.** 50 complaints, drawn as the first 50 of the 300-complaint judge sample in its stratified
order (so they are also in the 2,999-complaint factorial subset, and every reply already has both
LLM-judge ratings). For each complaint, six replies per model:

| Cell | Why it is in |
| --- | --- |
| V1 terse | the realistic baseline |
| V2 empathetic (= A00) | the anchor prompt; where the models split |
| V3 structured | the realistic compliance bundle |
| 000 (bare "3-4 sentences") | the factorial baseline; the placeholder extreme |
| A0C (framing + constraints) | the "sorry but not liable" cell |
| ABC (framing + format + constraints) | the factorial corner closest to V3 |

That is 50 × 6 × 2 = **600 replies**, of which every one is rated at least once and a subset
twice:

- **Double-coded block**: the first 15 complaints (15 × 12 = 180 replies) are rated by both raters.
  These give the inter-rater agreement estimates. 180 items is enough to put Krippendorff's α
  within about ±0.1.
- **Split block**: the remaining 35 complaints are divided between the raters by complaint (18 to
  rater A, 17 to rater B), so each reply there is rated once and each rater reads a complaint once
  for all twelve of its replies.

Rater A therefore rates 396 replies (33 complaints) and rater B 384 (32 complaints). At a realistic
pace of 60-75 seconds per reply, that is about 7-8 hours per rater, against 16-20 hours for full
double coding of 80 complaints. Cell means use the mean of the two raters where both rated an item
and the single rating elsewhere; every cell keeps 50 complaints per model. If more time is
available, raise `HR_DOUBLE` (more agreement precision) or `HR_COMPLAINTS` (more power for cell
contrasts) in the build script; `HR_COMPLAINTS=80 HR_DOUBLE=80` reproduces the full design. Do not
drop cells, because the cell contrasts are the point.

**Raters.** Two people who have read customer complaints professionally (customer service,
compliance, ombudsman, legal), or failing that two trained graduate raters plus a calibration round.
Raters must not have seen the paper's results or figures, and must not know which model or prompt
produced any reply. A third rater is worth adding only if the first two disagree badly on the
calibration set.

## 2. Blinding and materials

`analysis/human_rating_build.py` produces, in `analysis/human_rating/`:

- `rater_A.xlsx` and `rater_B.xlsx`: one row per reply assigned to that rater, in an order
  randomised separately for each rater, with columns `item_id`, `complaint_id`, the complaint as
  the models saw it (issue, sub-issue, narrative), the reply, then empty score columns with 1-5
  drop-downs and yes/no drop-downs, and a free-text `note` column. Replies within a complaint are
  **not** grouped, so a rater cannot line the six cells up and infer the prompt from the pattern.
  The two sheets share the 180 double-coded replies and otherwise differ; do not tell the raters
  which items are shared.
- `key.csv`: the mapping from `item_id` to model, cell, and which rater(s) it was assigned to. The
  coordinator keeps it; raters never open it. Do not put it on the shared drive with the sheets.
- `calibration.xlsx`: 30 replies (5 complaints × 6 cells, one model each, drawn from complaints 51-55
  of the sample so they do not overlap the main pass), used in Step 4.
- `rubric.md`: the rubric with anchors, printed below, for the raters.

Replies are shown verbatim, including any markdown asterisks, any `[placeholder]` text, and any
preamble the model wrote before the reply itself (such as a bare "**Response:**" line), because
those are things the paper measures. About half the replies run over several lines, so rows are
sized to show the whole cell; if a rater's spreadsheet program still shows only the first line
(for example "Dear [Consumer's Name],"), select all rows and auto-fit row height before rating.

## 3. The rubric

It is the same rubric the LLM judges used, so that human-judge agreement is a like-for-like
comparison. Raters score each reply on five criteria from 1 to 5 and answer three yes/no questions.
Anchors are given so that raters share a scale; the LLM judges did not see the anchors, which is
deliberate: the paper reports whether a judge given only the criterion text lands on the same scale
as humans given anchors.

**Acknowledgement** (does the reply reflect what *this* customer's complaint is about?)
5 = names the specific problem and the specific circumstance (e.g. "the collector texted you at
night after you asked them to stop"); 4 = names the problem correctly but generically; 3 = partly
right, or right at the category level only ("your debt collection concern"); 2 = vague enough to fit
most complaints; 1 = wrong or about something else.

**Concreteness** (does the reply give a specific, actionable next step?)
5 = a specific action, who does it, and by when; 4 = a specific action and one of who/when; 3 = a
specific action with neither; 2 = "we will review / look into this"; 1 = reassurance only, no action.

**Tone** (is it appropriate for a regulated financial institution writing to a customer?)
5 = professional, respectful, and warm without being saccharine or defensive; 4 = minor lapse
(slightly stiff, slightly gushing); 3 = noticeably off (over-familiar, corporate boilerplate,
lecturing); 2 = would embarrass the institution; 1 = rude, dismissive, or inappropriate.

**Grounding** (does it avoid inventing facts, figures, dates, names, or process not supported by the
complaint?) 5 = nothing invented; 4 = one harmless assumption; 3 = invents process ("our compliance
team", "within 24 hours") that the institution may not have; 2 = invents a fact, figure, or contact
detail; 1 = several inventions or a promise the institution could not know it can keep.

**Overall** (as a first response to send to this customer)
5 = send as is; 4 = send after a trivial edit; 3 = usable after a real edit; 2 = would need
rewriting; 1 = unusable. Overall is a judgement, not an average of the other four; a reply with an
unfilled `[Customer's Name]` cannot score above 2 here because it cannot be sent.

**Yes/no flags.**
`has_placeholder`: contains an unfilled template slot such as `[Customer's Name]` or `[date]`.
`promises_outcome`: promises a specific outcome (refund, removal, reversal, waiver, closure), not
merely a review or investigation. "We will ensure the refund is processed" counts as yes.
`admits_liability`: concedes that the institution did something wrong or unlawful. "I take full
ownership of the miscommunication" counts as yes; "I'm sorry for the frustration" alone does not.

Also: `would_send` (yes/no): if you were the agent, would you send this reply unchanged?

## 4. Procedure

1. **Briefing (30 minutes).** Walk both raters through the rubric and the anchors together, using
   three example replies you pick from the corpus that are clearly 5, 3, and 1 on overall. Tell them
   the replies were machine-generated by different systems and that the study is about the
   systems, not about them. Warn them that some narratives describe harassment, debt, and distress.
2. **Calibration (about 45 minutes).** Both raters score the 30-item calibration sheet
   independently. Compute agreement with `human_rating_analysis.py --calibration`. Then meet, go
   through every item where the two overall scores differ by 2 or more or any flag disagrees, and
   agree on how the anchor applies. Amend the anchor text if a rule was missing. Do **not** change
   calibration scores after discussion; they are reported as the pre-calibration agreement.
3. **Main pass.** Each rater works through their own sheet in the given order, in sessions of no
   more than two hours, over no more than two weeks. No discussion between raters until both are
   finished. Raters may leave a note on any item.
4. **Adjudication (optional).** For double-coded items where overall differs by 2 or more, a third
   person, or the two raters together, records an adjudicated score in a separate column. Report
   agreement on the original scores; use adjudicated scores only for the cell means if you want a
   single human number per reply.
5. **Analysis.** `python3 analysis/human_rating_analysis.py` reads the two returned sheets, joins
   the key, and produces the four numbers above plus the tables listed in Section 6.

## 5. What to tell the raters, and what not to

Tell them: the complaint is real and public (CFPB), names and figures were redacted by the CFPB
before publication, the reply is machine-written, and they should score the reply as the customer
would receive it. Do not tell them how many systems there are, which cells exist, that some prompts
were "empathetic" or "constrained", or anything about the paper's findings. If a rater asks whether a
reply "should" apologise, the answer is that they should score what is on the page against the
anchors.

## 6. Analysis outputs

`human_rating_analysis.py` writes to `analysis/tables/human_*.csv` and prints:

- **Inter-rater agreement** on the 180 double-coded items: Krippendorff's α (ordinal) and
  quadratic-weighted κ per 1-5 criterion; Cohen's κ per flag; exact and within-one agreement.
- **Human vs each LLM judge**: Spearman ρ and within-one agreement per criterion, using the mean of
  the two human scores; κ per flag against each judge and against the regex markers.
- **Instrument validity**: Spearman ρ between human overall and VADER compound, RoBERTa score, GPT
  judge overall, Mistral judge overall, and reply length; and a linear model of human overall on the
  instruments to see which add information.
- **Cell means under human rating**, by model and cell, with 95% CIs, next to the same cells' LLM-judge
  means; the human ChatGPT − Mistral gap per cell next to each judge's gap (the self-preference check
  against ground truth).
- **Calibration report**: agreement on the 30-item set before discussion.

Sample-size note: with 50 complaints, a paired difference of 0.35 points on a 1-5 scale with SD of
differences around 1.0 is detectable at α = 0.05 with power about 0.7, and 0.45 points at power
about 0.9 (80 complaints would give 0.6 and 0.85 for 0.25 and 0.35 points). The contrasts the paper
leans on are larger than that: the LLM judges report cell gaps of 0.3-1.7 and a model gap on V2 of
about 0.5, so the design can confirm or overturn them; contrasts under about 0.35 will come out as
"not distinguishable", which is itself a result. Single rating on the split block adds rater noise
to the cell means but does not bias them, because the split is by complaint and each rater sees
every cell of every complaint they rate.

## 7. Ethics and data handling

The complaints are already public and redacted by the CFPB; no new personal data is collected from
consumers. The raters are annotators, not subjects, but check your institution's rules: many treat
paid annotation as exempt, some require a short protocol. Pay raters for the full time including
calibration. Keep `key.csv` and the completed sheets in the repository only after the pass is done
and the raters have been told their scores will be published in anonymised form (rater A / rater B).
