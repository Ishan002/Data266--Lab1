# Task 3 shared data

Downloaded from the original CycleGAN paper's public dataset mirror (UC Berkeley), which
provides the same Monet<->Photo unpaired image collections used by Kaggle's "I'm Something
of a Painter Myself" competition:

```bash
cd task3_gan/data
curl -o monet2photo.zip https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip
python -c "import zipfile; zipfile.ZipFile('monet2photo.zip').extractall('.')"
# then reorganize the extracted trainA/trainB/testA/testB into the folder names used here:
python - <<'PY'
import shutil, os
shutil.move('monet2photo/trainA', 'monet_jpg')
shutil.move('monet2photo/trainB', 'photo_jpg')
shutil.move('monet2photo/testA', 'monet_jpg_test')
shutil.move('monet2photo/testB', 'photo_jpg_test')
os.rmdir('monet2photo')
os.remove('monet2photo.zip')
PY
```

Resulting folders (not committed to git — see `.gitignore` — regenerate with the commands
above):

- `monet_jpg/` — 1,072 real Monet paintings (training pool)
- `photo_jpg/` — 6,287 real photos (training pool)
- `monet_jpg_test/` — 121 held-out Monet paintings (used for FID/KID/LPIPS evaluation only)
- `photo_jpg_test/` — 751 held-out photos (used for FID/KID/LPIPS evaluation only)

Each member draws their own independent random subsample from `monet_jpg`/`photo_jpg` for
training (see each member's `train_cyclegan.py` — different seed, different sample size) and
from `monet_jpg_test`/`photo_jpg_test` for evaluation.
