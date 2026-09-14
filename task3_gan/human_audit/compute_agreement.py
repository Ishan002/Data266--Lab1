"""
Task 3 -- compute inter-rater agreement (Cohen's kappa + % agreement) from two completed
audit sheets. Run this AFTER both raters have independently filled in their own copy of
audit_sheet_template.csv (see README.md in this folder).

Usage:
    python compute_agreement.py rater_ishan.csv rater_charvee.csv
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common")))
from eval_gan import cohens_kappa, percent_agreement  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    df_a = pd.read_csv(sys.argv[1]).sort_values("sample_id").reset_index(drop=True)
    df_b = pd.read_csv(sys.argv[2]).sort_values("sample_id").reset_index(drop=True)
    assert (df_a["sample_id"] == df_b["sample_id"]).all(), "sample_id order mismatch between raters"

    results = {}
    for col in ["style_score_1to5", "content_score_1to5", "artifact_free_score_1to5"]:
        kappa = cohens_kappa(df_a[col], df_b[col])
        pct = percent_agreement(df_a[col], df_b[col])
        results[col] = {"cohens_kappa": kappa, "percent_agreement": pct}
        print(f"{col}: Cohen's kappa={kappa:.3f} | % agreement={pct:.1%}")

    overall_a = df_a[["style_score_1to5", "content_score_1to5", "artifact_free_score_1to5"]].mean().mean()
    overall_b = df_b[["style_score_1to5", "content_score_1to5", "artifact_free_score_1to5"]].mean().mean()
    print(f"\nMean overall human audit score -- rater A: {overall_a:.2f}/5 | rater B: {overall_b:.2f}/5")

    import json
    with open("audit_agreement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote audit_agreement_results.json")


if __name__ == "__main__":
    main()
