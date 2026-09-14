# Task 1 — Sequence Model Failure Analysis (Charvee)

Model: Post-LN char-GPT, 6 layers / 4 heads / 96-dim, trained 60 epochs on an independent
100K-char TinyStories slice (offset 150,000). Three failure cases below, drawn from
`outputs/generated_samples.json`.

## Case 1 — Degenerate repetition loop (greedy decoding)

**Prompt:** "The little dog"

**Generated:**
> "The little dogh to the said, \"The wat so ithe so care and to the some and the said to the gin the was and and his was happpy and and the was to the the sairke..."

**Failure type:** Repetition.

**Observation:** Same signature as any small greedy-decoded LM: the model collapses onto a
short cycle of the highest-frequency tokens ("the", "was", "and"). Interestingly this model's
greedy loop is slightly more varied than a pure "the the the" cycle — it interleaves "and",
"was", and fragments of quoted dialogue ("The wat so...") — which is consistent with this
model reaching a lower validation loss (1.73) than the teammate's shallower model, i.e. it
has learned a richer local transition structure even though greedy decoding still exposes
the same fundamental instability.

## Case 2 — Broken word formation (temperature sampling)

**Prompt:** "Once upon a time"

**Generated:**
> "Once upon a timered then the fefulrt. Emoovere sacled ran hould a befaed ways the git plired hand upp eaying stree thee brmet hear..."

**Failure type:** Broken grammar / non-words.

**Observation:** As with the teammate's model, character-level statistics look
English-like (correct letter frequencies, plausible endings like "-ed", "-ered") but many
tokens are not real words ("fefulrt", "Emoovere", "sacled", "befaed"). Because this model is
deeper (6 layers) it produces slightly longer runs of well-formed sub-word chunks before
breaking down than the 4-layer model does, but it still cannot consistently assemble full
dictionary words — again a data/capacity ceiling rather than an architectural bug.

## Case 3 — Loss of coherence / abrupt topic drift

**Prompt:** "One day, a girl"

**Generated:**
> "One day, a girle the ta spene to and tumether ashe and it she arard got thinss ming in the girllted and the bromes, the dad momme fur.\" \n\nMack and was boyard to Joe comef and ther wa ampled to reve skeam of and ames sen day dille..."

**Failure type:** Loss of coherence.

**Observation:** The continuation opens with a plausible sentence shape referencing "the
girl" but drifts within one paragraph break into an unrelated fragment introducing new,
ungrounded names ("Mack", "Joe") that have no connection to the prompt. This is a coherence
failure rather than a grammar failure: the local word/character shapes are more convincing
than in Case 2, but the narrative has no memory of its own subject beyond roughly one
sentence — expected given the 96-token attention window and the small training set.

## Takeaway

Charvee's deeper (6-layer) Post-LN model achieves better next-character statistics overall
(lower val loss, higher top-1 accuracy — see `metrics_report.csv`) but shows the same three
qualitative failure modes as the shallower Pre-LN model: greedy repetition, broken word
formation under sampling, and short-range-only coherence. This suggests these failures are
governed more by training-data scale (100K characters) than by the specific architectural
choices compared here.
