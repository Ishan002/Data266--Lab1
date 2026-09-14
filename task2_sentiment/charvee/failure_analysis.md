# Task 2 — Manual Error Review (Charvee)

Model reviewed: `experimental_gru` (my best-performing model). 20 errors reviewed on the
3,000-review test set: 5 confident false positives, 5 confident false negatives, 5
near-threshold errors, 5 slice-specific failures (worst slice: **long reviews**, 13.9% error
rate, though notably my GRU narrows the long-review gap far more than my other two models
do). Full raw cases in `failure_analysis_raw_cases.json`.

## 1. Confident false positives (true=negative, predicted=positive, prob≈1.0)

**Example:** *"I love Paris, but don't love the Vegas version of the hotel. Sure it has the
Eiffel Tower... Sure it has one of my favorite restaurants... Sure, if your room faces the
Blvd then you have a great view..."* (true: negative, pred: positive, prob 0.999)

**Error type:** Repeated-concession sentence structure fools the model. The reviewer uses
"Sure, X..." three times to list positives before delivering an overall negative verdict —
a rhetorical pattern (concession before criticism) that the GRU, which only sees the final
hidden state, doesn't capture from a single forward pass.

**Proposed fix:** Add explicit discourse-marker features (count of "sure", "but", "however")
as auxiliary inputs, or switch to a bidirectional GRU so the final-state summary isn't solely
built from a left-to-right pass that ends up dominated by the review's early content style.

## 2. Confident false negatives (true=positive, predicted=negative, prob≈0.001–0.002)

**Example:** *"Rooms are less than $50 a night...what did you expect?? Ok so it's not the
nicest place to stay on the strip, but the rooms are so ridiculously cheap..."*
(true: positive, pred: negative, prob 0.001)

**Error type:** Value-framing sentiment. The review is positive in the "great value for the
price" sense, using words that sound negative in isolation ("not the nicest") that are
actually part of a positive value argument. This is a genuinely hard pragmatic case even for
a human skimming quickly.

**Proposed fix:** Would need training examples specifically covering "cheap but good value"
framing, or a rule that discounts negation immediately followed by a "but + positive clause"
pattern.

## 3. Near-threshold errors (|prob − 0.5| < 0.01)

**Example:** *"went to this store while i was at the mall, noticed they sell illegal baby
turtles .. nuff said"* (true: negative, pred: positive, prob 0.500)

**Error type:** No explicit sentiment words at all — the negativity is entirely implicit
("illegal baby turtles" is understood as bad by a human reader via world knowledge, not
sentiment vocabulary). The model has no lexical signal to work with here, so it defaults to
its class prior.

**Proposed fix:** This class of error (sentiment implied by fact rather than stated by
adjective) is arguably out of scope for a from-scratch bag/sequence model without external
knowledge; flagging these as low-confidence for human review is more realistic than trying to
architect around it.

## 4. Slice-specific failures (long reviews, worst-performing slice)

**Example:** A review about buying a car opens positively about the salesman ("amazing!!!")
before the actual 2-star rating context is only implied through details later in the review;
true label negative, predicted positive with prob 0.944.

**Error type:** Long-review recency/primacy imbalance — same pattern the teammate found in
his BiLSTM's long-review errors, suggesting this is a shared limitation of any single fixed-
window sequence encoder rather than specific to one architecture choice.

**Proposed fix:** Increase MAX_LEN beyond 150 tokens for this slice specifically (a
length-adaptive truncation strategy), or add an auxiliary loss that supervises sentiment
prediction from the last third of the review alone.

## Takeaway

The GRU's errors cluster around discourse-level phenomena the model has no mechanism to
capture from a single final hidden state — concession structure, value-framing, and
long-review recency effects. This mirrors the teammate's BiLSTM findings closely (see his
`failure_analysis.md`), which is a useful team-level finding: both recurrent architectures,
despite different specifics, hit the same wall once local lexical sentiment gives way to
discourse-level structure.
