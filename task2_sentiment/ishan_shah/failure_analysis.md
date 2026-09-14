# Task 2 — Manual Error Review (Ishan Shah)

Model reviewed: `experimental_bilstm` (chosen as the most architecturally complex of my
three models). 20 errors reviewed on the 3,000-review test set: 5 confident false positives,
5 confident false negatives, 5 near-threshold errors, 5 slice-specific failures (worst slice:
**long reviews**, 14.7% error rate vs. 12.2%/9.5% for short/medium). Full raw cases with
complete text are in `failure_analysis_raw_cases.json`.

## 1. Confident false positives (true=negative, predicted=positive, prob≈1.0)

**Example:** *"...I ordered the grilled cheese with creamy tomato soup. Because I figured it
was impossible to go wrong with a comfort food staple and I'm tired of burgers at the
airport..."* (true: negative, pred: positive, prob 1.000)

**Error type:** Positive-word saturation / missed pivot. The review opens with genuinely
positive statements ("Love that they have Four Peaks beer") before pivoting to criticism
later ("wishing I had made a different food choice"). The BiLSTM's final hidden state is
dominated by the early positive framing.

**Proposed fix:** Add stronger negation/contrast-conjunction handling (e.g. explicit
features for "but", "however" as sentiment-pivot markers) so the model learns to weight
post-pivot content more heavily.

## 2. Confident false negatives (true=positive, predicted=negative, prob≈0.0–0.002)

**Example:** *"The steaks here are very good. It's the same quality as the other high end
places... I love the fact that they serve them on platters sizzling with butter..."*
(true: positive, pred: negative, prob 0.000)

**Error type:** Missed positive signal, likely a vocabulary/stemming artifact. Several of
these false negatives are unambiguously positive to a human reader but were stemmed in a way
that may have collided informative words with unrelated stems, diluting their embedding
signal.

**Proposed fix:** Compare stemmed vs. non-stemmed vocabularies on a validation split to check
for harmful stem collisions (e.g. "sizzl" from "sizzling" merging with unrelated words);
consider lemmatization instead of stemming as a less aggressive normalization.

## 3. Near-threshold errors (|prob − 0.5| < 0.02)

**Example:** *"Very nice selection of quality foods."* (true: positive, pred: negative,
prob 0.498)

**Error type:** Genuinely ambiguous / low-information short review. This 6-word review has
almost no context for the model to work with, and the prediction is essentially a coin flip.

**Proposed fix:** Route very short reviews (e.g. <15 words) to a rule-based lexicon lookup or
flag them as low-confidence for human review rather than forcing a binary threshold decision.

## 4. Slice-specific failures (long reviews, worst-performing slice)

**Example:** A ~180-word review praising a burger place ends with several caveats about wait
time and parking; true label negative, predicted positive with prob 0.943.

**Error type:** Long-review dilution. With many more positive adjectives ("best burgers",
"years through Yelp") than negative ones by raw count, the pooled/recurrent signal skews
positive even though the reviewer's overall verdict (reflected in the star rating that
produced the polarity label) was negative.

**Proposed fix:** Weight sentence-final content more heavily (reviews often save their
verdict for the end), or truncate/summarize long reviews to their last N sentences before
encoding, rather than truncating at a fixed token count from the start.

## Takeaway

Three of the four error categories point to the same underlying issue: the model aggregates
sentiment signal across the whole review without any mechanism for weighting *which part* of
the review matters most (early positive framing overwhelming a later negative verdict, and
vice versa). This is consistent with the per-slice metrics showing the "long" slice as the
worst performer for all three of my models, not just the BiLSTM.
