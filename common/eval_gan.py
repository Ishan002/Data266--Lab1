"""
Shared evaluation utilities for Task 3 (CycleGAN image style transfer).

Measurement-only shared code (no generator/discriminator architecture here), so both
members' reported numbers are computed the same way and stay directly comparable, while each
member's own CycleGAN architecture, losses, and training loop remain entirely their own.

Implements the Task 3 metrics list from the lab spec:
  FID/KID (both directions), cycle-reconstruction L1, LPIPS, content-preservation cosine
  similarity, and helpers for the human audit's inter-rater agreement (Cohen's kappa).

NOTE: FID/KID/LPIPS/content-similarity all use fixed, pretrained feature extractors
(Inception-v3, AlexNet, ResNet-18) purely to *measure* image quality/content-preservation.
Per the lab's leaderboard-integrity rule, these networks are never used to generate or touch
any image that gets submitted to Kaggle -- they are evaluation-only, exactly like using a
ruler to measure a drawing does not mean the ruler drew it.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def to_uint8_batch(imgs_float):
    """imgs_float: (N,3,H,W) in [-1,1] -> (N,3,H,W) uint8 in [0,255], resized to >=299 for FID."""
    imgs = (imgs_float.clamp(-1, 1) + 1) / 2 * 255
    imgs = imgs.to(torch.uint8)
    if imgs.shape[-1] < 75:
        imgs = F.interpolate(imgs.float(), size=(75, 75), mode="bilinear").to(torch.uint8)
    return imgs


def compute_fid_kid(real_imgs, fake_imgs, device="cuda", subset_size=None):
    """real_imgs, fake_imgs: (N,3,H,W) float tensors in [-1,1]. Returns dict with FID and KID."""
    real_u8 = to_uint8_batch(real_imgs).to(device)
    fake_u8 = to_uint8_batch(fake_imgs).to(device)

    fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)
    fid.update(real_u8, real=True)
    fid.update(fake_u8, real=False)
    fid_score = fid.compute().item()

    n = min(real_u8.shape[0], fake_u8.shape[0])
    ss = subset_size or max(2, min(n, 50))
    kid = KernelInceptionDistance(feature=64, subset_size=ss, normalize=False).to(device)
    kid.update(real_u8, real=True)
    kid.update(fake_u8, real=False)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid_score), "kid_mean": float(kid_mean), "kid_std": float(kid_std)}


def cycle_reconstruction_l1(originals, reconstructions):
    """Mean L1 distance between x and G_BA(G_AB(x)) (or vice versa), both in [-1,1]."""
    return float(F.l1_loss(originals, reconstructions).item())


class ContentFeatureExtractor:
    """Lightweight pretrained ResNet-18 truncated to global-pool features, for a
    content-preservation cosine-similarity metric between an input and its translation.
    Evaluation-only; never used to produce submitted images (see module docstring)."""

    def __init__(self, device="cuda"):
        import torchvision.models as models
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        net.fc = torch.nn.Identity()
        net.eval()
        self.net = net.to(device)
        self.device = device
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def cosine_similarity(self, imgs_a, imgs_b):
        """imgs_a, imgs_b: (N,3,H,W) in [-1,1]. Returns mean cosine similarity."""
        a = (imgs_a.clamp(-1, 1) + 1) / 2
        b = (imgs_b.clamp(-1, 1) + 1) / 2
        a = self.preprocess(a).to(self.device)
        b = self.preprocess(b).to(self.device)
        fa = self.net(a)
        fb = self.net(b)
        cos = F.cosine_similarity(fa, fb, dim=1)
        return float(cos.mean().item())


def lpips_distance(imgs_a, imgs_b, device="cuda"):
    """Perceptual distance (lower = more similar). imgs in [-1,1], (N,3,H,W)."""
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    with torch.no_grad():
        d = loss_fn(imgs_a.to(device), imgs_b.to(device))
    return float(d.mean().item())


def cohens_kappa(rater_a, rater_b):
    """Simple Cohen's kappa for two raters' categorical (or binary) judgments."""
    rater_a = np.asarray(rater_a)
    rater_b = np.asarray(rater_b)
    categories = sorted(set(rater_a) | set(rater_b))
    n = len(rater_a)
    po = np.mean(rater_a == rater_b)
    pe = sum(
        (np.mean(rater_a == c)) * (np.mean(rater_b == c)) for c in categories
    )
    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return float(kappa)


def percent_agreement(rater_a, rater_b):
    rater_a = np.asarray(rater_a)
    rater_b = np.asarray(rater_b)
    return float(np.mean(rater_a == rater_b))
