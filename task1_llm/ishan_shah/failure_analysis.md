# Task 1 — Sequence Model Failure Analysis (Ishan Shah)

Model: Pre-LN char-GPT, 4 layers / 4 heads / 128-dim, trained 60 epochs on a 100K-char
TinyStories slice. Three failure cases below, drawn from `outputs/generated_samples.json`.

## Case 1 — Degenerate repetition loop (greedy decoding)

**Prompt:** "One day, a girl"

**Generated:**
> "One day, a girl to the was to the to the stare sthe was and said the part the to fet the he and and the was the was to the warke..."

**Failure type:** Repetition / mode collapse.

**Observation:** Greedy decoding drives the model into a high-probability attractor loop
around the words "the", "was", "to" — the most frequent tokens in TinyStories. Because
greedy always takes the arg-max character, once the model enters this loop there is no
mechanism to escape it: every "the" strongly predicts a following space, which strongly
predicts a capital/common word, which loops back to "the". This is the classic failure
mode of small, low-entropy language models under deterministic decoding, and it disappears
almost entirely once temperature sampling is used instead.

## Case 2 — Broken grammar / lexical inconsistency (temperature sampling)

**Prompt:** "Once upon a time"

**Generated:**
> "Once upon a time was and s.\nShe pome the wand wave had ch ared mard ost to ound he cack..."

**Failure type:** Broken grammar / invalid word formation.

**Observation:** At temperature 0.8 the model produces plausible English *letter statistics*
(character bigrams/trigrams look right — "th", "ing", "ed" endings appear) but frequently
emits non-words ("pome", "ared", "mard", "cack"). This shows the model has learned local
character co-occurrence patterns but, with only ~0.8M parameters and 100K characters of
training data, has not learned long-enough-range dependencies to consistently complete real
English words or maintain subject-verb agreement across a sentence.

## Case 3 — Loss of coherence / structural hallucination (special-token leakage)

**Prompt:** "The little dog"

**Generated:**
> "The little dogher aind. Ighe then shered thafow boont han hom noutt stine rofnd bel pofot thavenœM_SEP>\nOnce ty icacere f a tour the bay..."

**Failure type:** Loss of coherence / hallucinated structural artifact.

**Observation:** The training corpus concatenates stories with a literal `<STORY_SEP>`
delimiter. The model has partially memorized this token's characters but reproduces it
corrupted ("œM_SEP>") and mid-sentence, then abruptly pivots to unrelated content ("Once
ty icacere f a tour..."), as if starting a new story. This is a hallucination in the sense
that the model inserts structure from its training format (story boundaries) into a context
where it doesn't belong, and it never recovers the original narrative thread about "the
little dog."

## Takeaway

All three failures trace back to the same root cause: a ~0.8M-parameter model trained on
only 100K characters for 60 epochs has enough capacity to learn character-level statistics
and short local patterns (word shapes, common function words) but not enough capacity or
data to learn long-range narrative coherence. Repeated 4-gram rate (52.3%) and low
distinct-1 (0.025) in the final metrics corroborate this: the model leans heavily on a small
set of very frequent character sequences.
