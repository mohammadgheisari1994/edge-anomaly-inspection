"""PatchCore-style anomaly detector (Roth et al., CVPR 2022), built so the
whole thing - backbone, patch features and nearest-neighbour scoring -
exports to a single ONNX graph for edge deployment.

Training needs only normal images: patch features from a frozen
ImageNet backbone are stored in a memory bank (reduced with greedy
coreset selection); at test time each patch is scored by its distance
to the nearest normal patch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def adaptive_pool_matrix(in_dim: int, out_dim: int) -> torch.Tensor:
    """Matrix equivalent of F.adaptive_avg_pool1d(in_dim -> out_dim).

    Written as a fixed linear map so it exports cleanly to ONNX.
    """
    w = torch.zeros(in_dim, out_dim)
    for i in range(out_dim):
        start = (i * in_dim) // out_dim
        end = math.ceil((i + 1) * in_dim / out_dim)
        w[start:end, i] = 1.0 / (end - start)
    return w


class FeatureExtractor(nn.Module):
    """Image [B,3,H,W] in [0,1] -> patch embeddings [B, h*w, D]."""

    def __init__(self, backbone: str = "wide_resnet50_2", pretrained: bool = True,
                 target_dim: int = 1024):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.local = nn.AvgPool2d(3, stride=1, padding=1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        with torch.no_grad():
            c = self._concat(torch.zeros(1, 3, 224, 224)).shape[1]
        self.register_buffer("proj", adaptive_pool_matrix(c, target_dim))
        self.eval()

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2, f3 = self.local(f2), self.local(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._concat(x)                      # B, C, h, w
        f = f.flatten(2).transpose(1, 2)         # B, h*w, C
        return f @ self.proj                     # B, h*w, D


class PatchCore(nn.Module):
    """Image [B,3,224,224] -> patch anomaly map [B,1,h,w] (L2 distance)."""

    def __init__(self, extractor: FeatureExtractor, bank: torch.Tensor, grid: int = 28):
        super().__init__()
        self.extractor = extractor
        self.grid = grid
        self.register_buffer("bank_t", bank.t().contiguous())            # D, M
        self.register_buffer("bank_sq", (bank * bank).sum(1).view(1, 1, -1))  # 1,1,M
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extractor(x)                                   # B, N, D
        d2 = (f * f).sum(-1, keepdim=True) - 2.0 * (f @ self.bank_t) + self.bank_sq
        dist = d2.min(dim=-1).values.clamp_min(0.0).sqrt()      # B, N
        return dist.view(-1, 1, self.grid, self.grid)


@torch.no_grad()
def greedy_coreset(feats: torch.Tensor, n: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection on a random projection (PatchCore sec. 3.2)."""
    n = min(n, feats.shape[0])
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)
    z = feats @ proj.to(feats.device)
    first = int(torch.randint(0, z.shape[0], (1,), generator=g))
    chosen = [first]
    min_d = ((z - z[first]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(min_d))
        chosen.append(i)
        min_d = torch.minimum(min_d, ((z - z[i]) ** 2).sum(1))
    return torch.tensor(chosen, dtype=torch.long)
