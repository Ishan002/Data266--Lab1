"""
Task 3 -- build the blinded 30-sample human audit sheet (lab spec 3.2.6).

Picks 30 FIXED generated samples (15 from each member's photo->Monet outputs, chosen
deterministically so both raters see the same set) and writes:
  - audit_images/ -- the 30 images, renamed to blinded ids (sample_01.jpg ... sample_30.jpg)
    with NO filename hint about which member/model produced them
  - audit_key.csv -- the hidden mapping from blinded id -> (member, source filename), used
    only after both raters finish, to compute inter-rater agreement per member
  - audit_sheet_template.csv -- what each rater actually fills in: sample_id + three empty
    columns (style_score, content_score, artifact_free_score), 1-5 Likert each

This only builds the blinded sheet. The actual audit -- two humans (Ishan and Charvee Saraiya)
independently rating all 30 images -- has to happen outside this script; see
task3_gan/human_audit/README.md.
"""
import csv
import os
import random
import shutil

random.seed(123)  # fixed seed -> the same 30 samples every time this is run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUDIT_IMG_DIR = os.path.join(HERE, "audit_images")
os.makedirs(AUDIT_IMG_DIR, exist_ok=True)

members = ["ishan_shah", "charvee_saraiya"]
n_per_member = 15

rows = []
sample_idx = 1
for member in members:
    pred_dir = os.path.join(ROOT, member, "outputs", "pred_B2A")  # photo -> Monet translations
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".jpg") and f != "sample_grid.png")
    chosen = random.sample(files, min(n_per_member, len(files)))
    for fname in chosen:
        blind_id = f"sample_{sample_idx:02d}"
        shutil.copy(os.path.join(pred_dir, fname), os.path.join(AUDIT_IMG_DIR, f"{blind_id}.jpg"))
        rows.append({"sample_id": blind_id, "member": member, "source_file": fname})
        sample_idx += 1

random.shuffle(rows)  # blind the order too, not just the filename

with open(os.path.join(HERE, "audit_key.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "member", "source_file"])
    w.writeheader()
    w.writerows(rows)

with open(os.path.join(HERE, "audit_sheet_template.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "style_score_1to5", "content_score_1to5", "artifact_free_score_1to5"])
    for r in rows:
        w.writerow([r["sample_id"], "", "", ""])

print(f"Wrote {len(rows)} blinded images to {AUDIT_IMG_DIR}")
print("Wrote audit_key.csv (hidden mapping -- don't look at this until after rating!)")
print("Wrote audit_sheet_template.csv (copy this twice, once per rater, before filling in)")
