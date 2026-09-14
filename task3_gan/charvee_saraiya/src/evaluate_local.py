"""
Task 3 -- local evaluation + Kaggle submission generator (Charvee Saraiya).

Loads the trained G_AB checkpoint (photo -> Monet direction) and produces `submission.csv`-
style output: a folder of generated Monet-style images from real photos, zipped as
`images.zip`, ready for `kaggle competitions submit`.

IMPORTANT (leaderboard integrity, see lab spec): every image in the zip is the direct,
unedited output of this member's own trained CycleGAN generator run in inference mode on
real photos.

Usage:
    python evaluate_local.py
Then submit yourself with your own Kaggle account:
    kaggle competitions submit -c <competition-slug> -f images.zip -m "Charvee Saraiya CycleGAN"
"""
import glob
import os
import zipfile

import torch
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 128
N_SUBMIT = 200

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(MEMBER_DIR, "..", "data")
CKPT_DIR = os.path.join(MEMBER_DIR, "checkpoints")
SUBMIT_DIR = os.path.join(MEMBER_DIR, "kaggle_submission_images")
os.makedirs(SUBMIT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Re-declare the exact generator architecture used in training (U-Net generator).
# NOTE: deliberately NOT `from train_cyclegan import UNetGenerator` -- that module is a
# script, not a library, so importing it would re-run its entire top-level training loop.
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.down1 = UNetDown(3, ngf, norm=False)
        self.down2 = UNetDown(ngf, ngf * 2)
        self.down3 = UNetDown(ngf * 2, ngf * 4)
        self.down4 = UNetDown(ngf * 4, ngf * 8)
        self.bottleneck = UNetDown(ngf * 8, ngf * 8)
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)
        self.up2 = UNetUp(ngf * 16, ngf * 4, dropout=0.5)
        self.up3 = UNetUp(ngf * 8, ngf * 2)
        self.up4 = UNetUp(ngf * 4, ngf)
        self.final = nn.Sequential(nn.ConvTranspose2d(ngf * 2, 3, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        b = self.bottleneck(d4)
        u1 = self.up1(b, d4); u2 = self.up2(u1, d3); u3 = self.up3(u2, d2); u4 = self.up4(u3, d1)
        return self.final(u4)


G_BA = UNetGenerator().to(device)  # photo -> monet
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
