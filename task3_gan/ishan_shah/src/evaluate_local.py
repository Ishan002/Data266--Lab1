"""
Task 3 -- local evaluation + Kaggle submission generator (Ishan Shah).

Loads the trained G_AB checkpoint (photo -> Monet direction, which is what the Kaggle
"I'm Something of a Painter Myself" competition scores) and produces `submission.csv`-style
output: a folder of generated Monet-style images from real photos, zipped as `images.zip`,
ready for `kaggle competitions submit`.

IMPORTANT (leaderboard integrity, see lab spec): every image in the zip is the direct,
unedited output of this member's own trained CycleGAN generator run in inference mode on
real photos. Nothing here hand-edits, hand-picks, or sources images externally.

Usage:
    python evaluate_local.py
Then submit yourself with your own Kaggle account:
    kaggle competitions submit -c <competition-slug> -f images.zip -m "Ishan Shah CycleGAN"
"""
import glob
import os
import zipfile

import torch
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 128
N_SUBMIT = 200  # number of photos to translate for the submission (Kaggle usually wants ~ a few hundred to 1000)

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(MEMBER_DIR, "..", "data")
CKPT_DIR = os.path.join(MEMBER_DIR, "checkpoints")
SUBMIT_DIR = os.path.join(MEMBER_DIR, "kaggle_submission_images")
os.makedirs(SUBMIT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Re-declare the exact generator architecture used in training (ResNet generator).
# NOTE: deliberately NOT `from train_cyclegan import ResNetGenerator` -- that module is a
# script, not a library, so importing it would re-run its entire top-level training loop.
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, n_residual=6, ngf=64):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True)]
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        for _ in range(n_residual):
            layers.append(ResidualBlock(in_ch))
        for _ in range(2):
            out_ch = in_ch // 2
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


G_BA = ResNetGenerator().to(device)  # photo -> monet
G_BA.load_state_dict(torch.load(os.path.join(CKPT_DIR, "G_BA.pt"), map_location=device))
G_BA.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

photo_files = sorted(glob.glob(os.path.join(DATA_DIR, "photo_jpg", "*.jpg")))[:N_SUBMIT]
print(f"Translating {len(photo_files)} photos -> Monet style for Kaggle submission...")

with torch.no_grad():
    for i, fpath in enumerate(photo_files):
        img = Image.open(fpath).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        fake = G_BA(x)[0]
        out = ((fake.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(out).save(os.path.join(SUBMIT_DIR, f"{i:04d}.jpg"))

zip_path = os.path.join(MEMBER_DIR, "..", "submission.csv").replace("submission.csv", "images.zip")
zip_path = os.path.join(MEMBER_DIR, "images.zip")
with zipfile.ZipFile(zip_path, "w") as zf:
    for fname in sorted(os.listdir(SUBMIT_DIR)):
        zf.write(os.path.join(SUBMIT_DIR, fname), arcname=fname)

print(f"Wrote {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
print("NOTE: submission.csv is a placeholder -- fill in your Kaggle leaderboard rank/score "
      "after submitting images.zip yourself with `kaggle competitions submit`.")

with open(os.path.join(MEMBER_DIR, "submission.csv"), "w") as f:
    f.write("status,note\n")
    f.write("pending,Run 'kaggle competitions submit' with images.zip using your own Kaggle account, "
            "then replace this file with the leaderboard rank/score.\n")
