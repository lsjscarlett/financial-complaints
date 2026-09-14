# Rating rubric

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
