# Task 3 — Blinded Human Audit (lab spec 3.2.6)

This is the one piece of Task 3 that genuinely requires two humans (Ishan and Charvee) —
it cannot be automated or filled in on your behalf.

## Steps

1. Run the sheet builder once (already done — see `audit_images/`, 30 blinded samples,
   15 from each member's photo→Monet CycleGAN output, filenames give no hint of which
   member/model produced which image):
   ```bash
   python build_audit_sheet.py
   ```
2. **Both of you independently**, without looking at `audit_key.csv` or discussing scores
   with each other first, open every image in `audit_images/` and fill in your own copy of
   `audit_sheet_template.csv` (copy it to `rater_ishan.csv` and `rater_charvee.csv`) with a
   1–5 Likert score for each column:
   - `style_score_1to5` — how convincingly Monet-esque is the brushwork/color palette?
   - `content_score_1to5` — is the original photo's scene/content still recognizable?
   - `artifact_free_score_1to5` — absence of obvious GAN artifacts (color blotches, seams,
     texture smearing); 5 = no visible artifacts, 1 = heavily artifacted.
3. Once both `rater_ishan.csv` and `rater_charvee.csv` are complete, compute agreement:
   ```bash
   python compute_agreement.py rater_ishan.csv rater_charvee.csv
   ```
   This prints Cohen's kappa and percent agreement per criterion, and writes
   `audit_agreement_results.json`.
4. Only now open `audit_key.csv` to see which samples came from which member's model, and
   report the per-member mean scores and the overall inter-rater agreement in the final
   report's Task 3 section.

## Status

`audit_images/` and `audit_key.csv`/`audit_sheet_template.csv` are generated. The two rater
CSVs (`rater_ishan.csv`, `rater_charvee.csv`) and `audit_agreement_results.json` are **not
yet created** — this is the one deliverable in this repo waiting on the two of you.
