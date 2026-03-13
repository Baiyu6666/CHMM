#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AE/VAE Manifold Benchmark (on-manifold only training; eval uses GT on/off sampling)

Features:
1) Multiple datasets:
   - 3D: spiral (1D curve in R3), sphere surface (2D manifold in R3),
         paraboloid surface z=x^2+y^2 (2D manifold in R3),
         two-sphere union outer boundary (2D manifold in R3, keeps exterior surface only)
   - 2D: circle boundary (1D manifold in R2), square boundary (1D manifold in R2)

2) Train AE and/or VAE
3) Evaluate as one-class classifier using recon error ||x - D(E(x))||:
   - classify as ON if error <= threshold
   - compute confusion matrix, acc, precision, recall, f1, AUROC (score = -error)
   - GT on/off samples are generated with fixed seed for reproducibility
4) Plot:
   - latent scatter: encoded eval points (AE: z, VAE: mu), colored by true on/off
   - latent sampling -> decode -> scatter in original space
5) Plot:
   - arrows from original points to projected points P(x)=D(E(x))

Notes:
- "ON manifold" means within threshold under the model’s reconstruction/projection mapping.
- This is not guaranteed to match true manifold globally; it’s a benchmark tool.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def recon_error_l2(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    # per-sample L2 norm
    return torch.norm(x - x_hat, dim=1)


def batch_iter(x: torch.Tensor, batch_size: int, shuffle: bool = True):
    n = x.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        j = idx[s : s + batch_size]
        yield x[j]


# -----------------------------
# Dataset definitions (GT samplers)
# -----------------------------

@dataclass
class DatasetSpec:
    name: str
    dim: int
    latent_dim_default: int
    train_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_off_sampler: Callable[[int, np.random.Generator], np.ndarray]
    # bounding box for plotting / off sampling reference
    plot_bounds: Tuple[np.ndarray, np.ndarray]  # (lo, hi)


def sample_spiral_on(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    True 3D helix curve (1D manifold):
        x = cos(theta)
        y = sin(theta)
        z = a * theta
    """
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25  # controls pitch
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0  # shift to roughly center around z=0
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_spiral_off(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Points not near the helix curve.
    """
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0

    # move points radially away from helix
    dx = rng.uniform(0.4, 1.0, size=n)
    dy = rng.uniform(0.4, 1.0, size=n)
    dz = rng.uniform(0.4, 1.0, size=n)

    x += dx * rng.choice([-1, 1], size=n)
    y += dy * rng.choice([-1, 1], size=n)
    z += dz * rng.choice([-1, 1], size=n)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_sphere_on(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    center=(0, 0, 0),
) -> np.ndarray:
    # uniform on sphere surface via normal distribution
    v = rng.normal(size=(n, 3))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    pts = radius * v + np.array(center, dtype=float)[None, :]
    return pts.astype(np.float32)


def sample_sphere_off(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    center=(0, 0, 0),
) -> np.ndarray:
    # sample in a cube and reject near-surface band
    lo = np.array(center) - 1.8 * radius
    hi = np.array(center) + 1.8 * radius
    out = []
    while len(out) < n:
        m = int((n - len(out)) * 1.5) + 16
        pts = rng.uniform(lo, hi, size=(m, 3))
        r = np.linalg.norm(pts - np.array(center)[None, :], axis=1)
        # keep points away from surface band [0.85R, 1.15R]
        mask = (r < 0.85 * radius) | (r > 1.15 * radius)
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def sample_paraboloid_on(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_scale: float = 1.0,
) -> np.ndarray:
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z = z_scale * (x ** 2 + y ** 2)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_paraboloid_off(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_max: float = 3.0,
) -> np.ndarray:
    # sample in a box; force z to differ from x^2+y^2 by a margin
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z = rng.uniform(0.0, z_max, size=n)
    z_surface = x ** 2 + y ** 2
    # push z away from surface
    sign = rng.choice([-1.0, 1.0], size=n)
    delta = rng.uniform(0.35, 1.0, size=n)
    z2 = np.clip(z_surface + sign * delta, 0.0, z_max)
    return np.stack([x, y, z2], axis=1).astype(np.float32)


def sample_two_sphere_outer_on(n: int, rng: np.random.Generator) -> np.ndarray:
    # Two spheres; keep union outer boundary:
    # sphere A: center (-0.8, 0, 0), R=1
    # sphere B: center (+0.8, 0, 0), R=1
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0

    # sample candidates from both surfaces, then keep those not inside the other sphere
    m = int(n * 2.2) + 64
    pts_a = sample_sphere_on(m, rng, radius=r, center=ca)
    pts_b = sample_sphere_on(m, rng, radius=r, center=cb)
    pts = np.concatenate([pts_a, pts_b], axis=0)

    da = np.linalg.norm(pts - ca[None, :], axis=1)
    db = np.linalg.norm(pts - cb[None, :], axis=1)
    # on A surface means da≈R; keep those with db>=R (not inside B)
    # on B surface means db≈R; keep those with da>=R (not inside A)
    keep = (db >= r - 1e-6) | (da >= r - 1e-6)
    pts = pts[keep]

    if pts.shape[0] < n:
        # fallback: just return what we have (rare)
        return pts.astype(np.float32)
    idx = rng.choice(pts.shape[0], size=n, replace=False)
    return pts[idx].astype(np.float32)


def sample_two_sphere_outer_off(n: int, rng: np.random.Generator) -> np.ndarray:
    # sample around both spheres but avoid the union boundary band
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    lo = np.array([-2.5, -2.0, -2.0])
    hi = np.array([+2.5, +2.0, +2.0])

    out = []
    while len(out) < n:
        m = int((n - len(out)) * 1.8) + 32
        pts = rng.uniform(lo, hi, size=(m, 3))
        da = np.linalg.norm(pts - ca[None, :], axis=1)
        db = np.linalg.norm(pts - cb[None, :], axis=1)
        # distance to union boundary roughly min(|da-R|, |db-R|)
        dist_to_boundary = np.minimum(np.abs(da - r), np.abs(db - r))
        # keep those sufficiently far from boundary
        mask = dist_to_boundary > 0.25
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def sample_circle_on(n: int, rng: np.random.Generator, r: float = 1.0) -> np.ndarray:
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_circle_off(n: int, rng: np.random.Generator, r: float = 1.0) -> np.ndarray:
    # sample in box and keep away from radius band
    out = []
    while len(out) < n:
        m = int((n - len(out)) * 1.6) + 16
        pts = rng.uniform(-1.8, 1.8, size=(m, 2))
        rad = np.linalg.norm(pts, axis=1)
        mask = (rad < 0.75 * r) | (rad > 1.25 * r)
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def sample_square_on(n: int, rng: np.random.Generator, half: float = 1.0) -> np.ndarray:
    # boundary of axis-aligned square: x=±half or y=±half
    # choose edges uniformly
    edge = rng.integers(0, 4, size=n)
    u = rng.uniform(-half, half, size=n)
    x = np.empty(n)
    y = np.empty(n)
    # 0: top y=half, x=u
    mask = edge == 0
    x[mask] = u[mask]
    y[mask] = half
    # 1: bottom y=-half
    mask = edge == 1
    x[mask] = u[mask]
    y[mask] = -half
    # 2: right x=half
    mask = edge == 2
    x[mask] = half
    y[mask] = u[mask]
    # 3: left x=-half
    mask = edge == 3
    x[mask] = -half
    y[mask] = u[mask]
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_square_off(n: int, rng: np.random.Generator, half: float = 1.0) -> np.ndarray:
    # sample in a larger box and avoid boundary band
    out = []
    while len(out) < n:
        m = int((n - len(out)) * 1.6) + 16
        pts = rng.uniform(-1.8 * half, 1.8 * half, size=(m, 2))
        # distance to square boundary in L_inf: abs(max(|x|,|y|)-half)
        linf = np.maximum(np.abs(pts[:, 0]), np.abs(pts[:, 1]))
        dist_to_boundary = np.abs(linf - half)
        mask = dist_to_boundary > 0.18 * half
        sel = pts[mask]
        out.append(sel)
        out_arr = np.concatenate(out, axis=0)
        out = [out_arr[:n]]
    return out[0].astype(np.float32)


def build_datasets() -> Dict[str, DatasetSpec]:
    ds = {}

    ds["spiral3d"] = DatasetSpec(
        name="spiral3d",
        dim=3,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_spiral_off(n, rng),
        plot_bounds=(np.array([-2.2, -2.2, -2.8]), np.array([2.2, 2.2, 2.8])),
    )

    ds["sphere3d"] = DatasetSpec(
        name="sphere3d",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_sphere_on(n, rng, radius=1.0, center=(0, 0, 0)),
        eval_on_sampler=lambda n, rng: sample_sphere_on(n, rng, radius=1.0, center=(0, 0, 0)),
        eval_off_sampler=lambda n, rng: sample_sphere_off(n, rng, radius=1.0, center=(0, 0, 0)),
        plot_bounds=(np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0])),
    )

    ds["paraboloid3d"] = DatasetSpec(
        name="paraboloid3d",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_off_sampler=lambda n, rng: sample_paraboloid_off(n, rng, xy_range=1.2, z_max=3.0),
        plot_bounds=(np.array([-1.8, -1.8, 0.0]), np.array([1.8, 1.8, 3.2])),
    )

    ds["twosphere3d"] = DatasetSpec(
        name="twosphere3d",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_two_sphere_outer_off(n, rng),
        plot_bounds=(np.array([-2.8, -2.2, -2.2]), np.array([2.8, 2.2, 2.2])),
    )

    ds["circle2d"] = DatasetSpec(
        name="circle2d",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_circle_on(n, rng, r=1.0),
        eval_on_sampler=lambda n, rng: sample_circle_on(n, rng, r=1.0),
        eval_off_sampler=lambda n, rng: sample_circle_off(n, rng, r=1.0),
        plot_bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    )

    ds["square2d"] = DatasetSpec(
        name="square2d",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_off_sampler=lambda n, rng: sample_square_off(n, rng, half=1.0),
        plot_bounds=(np.array([-2.3, -2.3]), np.array([2.3, 2.3])),
    )

    return ds


# -----------------------------
# Models: AE and VAE
# -----------------------------

class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        enc_layers = []
        d = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc_layers += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        d = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec_layers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class VAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        # encoder trunk
        enc_layers = []
        d = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.enc_trunk = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(d, latent_dim)
        self.logvar_head = nn.Linear(d, latent_dim)

        # decoder
        dec_layers = []
        d = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec_layers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_trunk(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# -----------------------------
# Training
# -----------------------------

@dataclass
class TrainConfig:
    epochs: int = 400
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    # VAE only
    beta_final: float = 1.0
    warmup_epochs: int = 200


def train_ae(model: AutoEncoder, x_train: torch.Tensor, cfg: TrainConfig, device: torch.device) -> None:
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        losses = []
        for xb in batch_iter(x_train, cfg.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            x_hat = model(xb)
            loss = ((xb - x_hat) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if (ep + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"[AE] epoch {ep + 1:4d}/{cfg.epochs}  loss={np.mean(losses):.6f}")


def train_vae(model: VAE, x_train: torch.Tensor, cfg: TrainConfig, device: torch.device) -> None:
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        losses = []
        recon_losses = []
        kl_losses = []
        beta = cfg.beta_final * min(1.0, (ep + 1) / max(1, cfg.warmup_epochs))
        for xb in batch_iter(x_train, cfg.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(xb)
            recon = ((xb - x_hat) ** 2).mean()
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon.detach().cpu()))
            kl_losses.append(float(kl.detach().cpu()))
        if (ep + 1) % max(1, cfg.epochs // 10) == 0:
            print(
                f"[VAE] epoch {ep + 1:4d}/{cfg.epochs}  loss={np.mean(losses):.6f} "
                f"recon={np.mean(recon_losses):.6f}  kl={np.mean(kl_losses):.6f}  beta={beta:.3f}"
            )


# -----------------------------
# Evaluation (one-class style)
# -----------------------------

@dataclass
class EvalResult:
    threshold: float
    cm: np.ndarray
    acc: float
    prec: float
    rec: float
    f1: float
    auroc: float


def choose_threshold(
    errors_on: np.ndarray,
    errors_off: np.ndarray,
    method: str = "percentile",
    q: float = 95.0,
) -> float:
    """
    Choose threshold on recon error.

    Default: percentile of ON errors (e.g., 95th percentile).
    """
    if method == "percentile":
        return float(np.percentile(errors_on, q))
    if method == "midpoint":
        return float(0.5 * (np.median(errors_on) + np.median(errors_off)))
    raise ValueError(f"Unknown threshold method: {method}")


def estimate_threshold(
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    errors_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_val_on: np.ndarray,
    threshold_method: str = "percentile",
    threshold_q: float = 95.0,
    device: torch.device = "cpu",
) -> float:
    """Use ONLY on-manifold validation data to estimate threshold."""
    xt = torch.tensor(x_val_on.astype(np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        x_proj = project_fn(xt)
        err = errors_fn(xt, x_proj).cpu().numpy()

    if threshold_method == "percentile":
        thr = float(np.percentile(err, threshold_q))
    else:
        raise ValueError(f"Unknown threshold_method={threshold_method}")
    return thr


def eval_with_threshold(
    model_name: str,
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    errors_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_test_on: np.ndarray,
    x_test_off: np.ndarray,
    thr: float,
    device: torch.device = "cpu",
) -> Tuple[EvalResult, Dict[str, np.ndarray]]:
    """Evaluate on test set using a FIXED threshold."""
    x = np.concatenate([x_test_on, x_test_off], axis=0).astype(np.float32)
    y_true = np.concatenate([np.ones(len(x_test_on)), np.zeros(len(x_test_off))], axis=0).astype(int)

    xt = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_proj = project_fn(xt)
        err = errors_fn(xt, x_proj).cpu().numpy()

    y_pred = (err <= thr).astype(int)  # 1=ON, 0=OFF

    # ---- metrics (replace with your numpy versions if you removed sklearn) ----
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    scores = -err
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")

    res = EvalResult(
        threshold=thr,
        cm=cm,
        acc=float(acc),
        prec=float(prec),
        rec=float(rec),
        f1=float(f1),
        auroc=float(auroc),
    )

    print(f"\n[{model_name}] threshold(from val_on)={thr:.6f}")
    print(f"[{model_name}] CM (rows true [ON,OFF], cols pred [ON,OFF])\n{cm}")
    print(
        f"[{model_name}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
        f"f1={f1:.4f} auroc={auroc:.4f}"
    )

    cache = {"x": x, "y_true": y_true, "err": err, "x_proj": x_proj.cpu().numpy()}
    return res, cache


def visualize_all(
    ds,
    x_train,
    ae_pack=None,   # (cache, z_eval, x_dec)
    vae_pack=None,  # (cache, z_eval, x_dec)
    latent_dim=1,
    max_arrows_2d=30,
    max_arrows_3d=30,
    arrow_color="tab:gray",
):
    """
    2x3 layout:
      row 0: AE  -> [latent, decode, projection+arrows]
      row 1: VAE -> [latent, decode, projection+arrows]

    ae_pack / vae_pack:
      cache: dict with keys ["x", "y_true", "x_proj", "err"] from eval
      z_eval: (N, latent_dim) encoded eval points (AE: z, VAE: mu)
      x_dec: (M, dim) decoded points from latent sampling
    """
    fig = plt.figure(figsize=(16, 10))

    def _plot_latent(ax, z_eval, y_true, z_sample, title):
        on = y_true == 1
        off = y_true == 0

        if z_eval.shape[1] == 1:
            jitter = 0.02 * np.random.randn(len(z_eval))
            # jitter are random values to visually separate points along y-axis; they don't represent any real value
            ax.scatter(z_eval[on, 0], jitter[on], s=12, label="GT ON")
            ax.scatter(z_eval[off, 0], jitter[off], s=12, label="GT OFF")

            # new sampled latent points
            jitter2 = 0.02 * np.random.randn(len(z_sample))
            ax.scatter(
                z_sample[:, 0],
                jitter2,
                s=20,
                marker="x",
                color="red",
                label="sampled z",
            )

            ax.set_yticks([])
            ax.set_xlabel("z[0]")
        else:
            ax.scatter(z_eval[on, 0], z_eval[on, 1], s=12, label="GT ON")
            ax.scatter(z_eval[off, 0], z_eval[off, 1], s=12, label="GT OFF")

            ax.scatter(z_sample[:, 0], z_sample[:, 1], s=25, marker="x", color="red", label="sampled z")

            ax.set_xlabel("z[0]")
            ax.set_ylabel("z[1]")

        ax.set_title(title)
        ax.legend(loc="best")

    def _plot_decode(ax, x_dec, title):
        # add legend required
        if ds.dim == 2:
            ax.scatter(x_train[:, 0], x_train[:, 1], s=8, alpha=0.20, label="train (GT ON)")
            ax.scatter(x_dec[:, 0], x_dec[:, 1], s=14, alpha=0.9, label="decoded")
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.legend(loc="best")
        else:
            ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=8, alpha=0.12, label="train (GT ON)")
            ax.scatter(x_dec[:, 0], x_dec[:, 1], x_dec[:, 2], s=18, alpha=0.9, label="decoded")
            ax.set_title(title)
            ax.legend(loc="best")

    def _plot_projection(ax, cache, title):
        # add legend required; arrows as dashed lines with same color
        x = cache["x"]
        y_true = cache["y_true"]
        x_proj = cache["x_proj"]

        on = y_true == 1
        off = y_true == 0

        if ds.dim == 2:
            ax.scatter(x[on, 0], x[on, 1], s=10, alpha=0.35, label="GT ON (orig)")
            ax.scatter(x[off, 0], x[off, 1], s=10, alpha=0.35, label="GT OFF (orig)")

            n = len(x)
            idx = np.arange(n)
            if n > max_arrows_2d:
                idx = np.random.choice(n, size=max_arrows_2d, replace=False)

            # dashed line segments: same color for all
            for i in idx:
                ax.plot(
                    [x[i, 0], x_proj[i, 0]],
                    [x[i, 1], x_proj[i, 1]],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    color=arrow_color,
                )

            ax.scatter(x_proj[idx, 0], x_proj[idx, 1], s=14, alpha=0.85, label="projected")
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.legend(loc="best")
        else:
            ax.scatter(x[on, 0], x[on, 1], x[on, 2], s=12, alpha=0.22, label="GT ON (orig)")
            ax.scatter(x[off, 0], x[off, 1], x[off, 2], s=12, alpha=0.22, label="GT OFF (orig)")

            n = len(x)
            idx = np.arange(n)
            if n > max_arrows_3d:
                idx = np.random.choice(n, size=max_arrows_3d, replace=False)

            for i in idx:
                ax.plot(
                    [x[i, 0], x_proj[i, 0]],
                    [x[i, 1], x_proj[i, 1]],
                    [x[i, 2], x_proj[i, 2]],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    color=arrow_color,
                )

            ax.scatter(
                x_proj[idx, 0],
                x_proj[idx, 1],
                x_proj[idx, 2],
                s=16,
                alpha=0.85,
                label="projected",
            )
            ax.set_title(title)
            ax.legend(loc="best")

    # Helper to create correct axes (2D vs 3D) per column
    def _make_ax(row, col):
        # col: 0 latent(2D), 1 decode(2D/3D), 2 projection(2D/3D)
        pos = row * 3 + col + 1  # 1..6
        if col == 0:
            return fig.add_subplot(2, 3, pos)  # latent always 2D plot
        if ds.dim == 2:
            return fig.add_subplot(2, 3, pos)
        return fig.add_subplot(2, 3, pos, projection="3d")

    # --- Row 0: AE ---
    if ae_pack is not None:
        cache, z_eval, x_dec, z_samp = ae_pack
        ax = _make_ax(0, 0)
        _plot_latent(ax, z_eval, cache["y_true"], z_samp, title="AE: latent (E(x))")

        ax = _make_ax(0, 1)
        _plot_decode(ax, x_dec, title="AE: decode (sample z → D(z))")

        ax = _make_ax(0, 2)
        _plot_projection(ax, cache, title="AE: projection (x → D(E(x)))")
    else:
        # keep layout consistent
        for c in range(3):
            ax = _make_ax(0, c)
            ax.set_axis_off()
            ax.set_title("AE: (not run)")

    # --- Row 1: VAE ---
    if vae_pack is not None:
        cache, z_eval, x_dec, z_samp = vae_pack
        ax = _make_ax(1, 0)
        _plot_latent(ax, z_eval, cache["y_true"], z_samp, title="VAE: latent (mu(x))")

        ax = _make_ax(1, 1)
        _plot_decode(ax, x_dec, title="VAE: decode (z~N(0,1) → D(z))")

        ax = _make_ax(1, 2)
        _plot_projection(ax, cache, title="VAE: projection (x → D(mu(x)))")
    else:
        for c in range(3):
            ax = _make_ax(1, c)
            ax.set_axis_off()
            ax.set_title("VAE: (not run)")

    fig.suptitle(f"Dataset: {ds.name} | dim={ds.dim} | latent_dim={latent_dim}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_conditional_decodes(
    ds,
    x_on: np.ndarray,
    model: VAE,
    device: torch.device,
    n_points: int = 3,
    n_samples: int = 30,
    seed: int = 123,
) -> None:
    """
    Pick a few ON-manifold points, sample z ~ q(z|x), decode, and plot the decoded clouds.
    """
    if x_on.shape[0] == 0:
        return

    rng = np.random.default_rng(seed)
    idx = rng.choice(x_on.shape[0], size=min(n_points, x_on.shape[0]), replace=False)
    x_sel = x_on[idx]

    with torch.no_grad():
        xt = to_tensor(x_sel, device)
        mu, logvar = model.encode(xt)
        std = torch.exp(0.5 * logvar)

        all_decoded = []
        for i in range(x_sel.shape[0]):
            eps = torch.randn(n_samples, mu.shape[1], device=device)
            z = mu[i].unsqueeze(0) + eps * std[i].unsqueeze(0)
            x_dec = model.decode(z).cpu().numpy()
            all_decoded.append(x_dec)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_decoded)))

    if ds.dim == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x_on[:, 0], x_on[:, 1], s=8, alpha=0.18, label="GT ON")
        for i, x_dec in enumerate(all_decoded):
            ax.scatter(
                x_dec[:, 0],
                x_dec[:, 1],
                s=18,
                alpha=0.9,
                color=colors[i],
                label=f"x#{i+1}",
            )
            ax.scatter(
                x_sel[i, 0],
                x_sel[i, 1],
                s=120,
                marker="X",
                edgecolor="black",
                linewidth=1.0,
                color=colors[i],
                label=f"x#{i+1} (orig)",
            )
        ax.set_aspect("equal")
        ax.set_title("VAE: q(z|x) samples decoded")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_on[:, 0], x_on[:, 1], x_on[:, 2], s=8, alpha=0.15, label="GT ON")
        for i, x_dec in enumerate(all_decoded):
            ax.scatter(
                x_dec[:, 0],
                x_dec[:, 1],
                x_dec[:, 2],
                s=22,
                alpha=0.9,
                color=colors[i],
                label=f"x#{i+1}",
            )
            ax.scatter(
                x_sel[i, 0],
                x_sel[i, 1],
                x_sel[i, 2],
                s=140,
                marker="X",
                edgecolor="black",
                linewidth=1.0,
                color=colors[i],
                label=f"x#{i+1} (orig)",
            )
        ax.set_title("VAE: q(z|x) samples decoded")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spiral3d", choices=list(build_datasets().keys()))
    parser.add_argument("--models", type=str, default="both", choices=["ae", "vae", "both"])
    parser.add_argument("--latent_dim", type=int, default=-1, help="If -1, use dataset default.")
    parser.add_argument("--train_n", type=int, default=600)
    parser.add_argument("--eval_on_n", type=int, default=500)
    parser.add_argument("--eval_off_n", type=int, default=500)
    parser.add_argument("--sample_latent_n", type=int, default=150)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_final", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=440)
    parser.add_argument(
        "--threshold_q",
        type=float,
        default=95.0,
        help="Percentile of ON errors to set threshold.",
    )
    parser.add_argument("--outdir", type=str, default="outputs_manifold_bench")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto: use cuda if available else cpu",
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ds = build_datasets()[args.dataset]

    latent_dim = ds.latent_dim_default if args.latent_dim < 0 else args.latent_dim

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Check torch install and nvidia-smi.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[Device] Using {device}")

    # ---------------- Train data (ON only) ----------------
    rng_train = np.random.default_rng(args.train_seed)
    x_train = ds.train_on_sampler(args.train_n, rng_train)
    x_train_t = to_tensor(x_train, device)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_final=args.beta_final,
        warmup_epochs=args.warmup,
    )

    # Validation data
    x_val_on = ds.train_on_sampler(args.train_n, rng_train)

    # ---------------- Eval data (GT ON + OFF) with fixed seed ----------------
    rng_eval = np.random.default_rng(args.eval_seed)
    x_on = ds.eval_on_sampler(args.eval_on_n, rng_eval)
    x_off = ds.eval_off_sampler(args.eval_off_n, rng_eval)
    x_eval = np.concatenate([x_on, x_off], axis=0).astype(np.float32)
    y_true = np.concatenate([np.ones(len(x_on)), np.zeros(len(x_off))], axis=0).astype(int)

    # ---------------- Train & Eval models ----------------
    results = {}

    def run_ae():
        model = AutoEncoder(in_dim=ds.dim, latent_dim=latent_dim).to(device)
        train_ae(model, x_train_t, cfg, device)

        # after training AE model ...
        def proj(xt: torch.Tensor) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                return model(xt.to(device))

        thr = estimate_threshold(
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_val_on=x_val_on,
            threshold_method="percentile",
            threshold_q=args.threshold_q,
            device=device,
        )

        res, cache = eval_with_threshold(
            model_name=f"AE/{ds.name}/z{latent_dim}",
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_test_on=x_on,
            x_test_off=x_off,
            thr=thr,
            device=device,
        )

        results["ae"] = (res, cache, model)

        # latent scatter
        model.eval()
        with torch.no_grad():
            z = model.encode(to_tensor(x_eval, device)).cpu().numpy()

        # latent sampling -> decode
        # sample around encoded distribution (mean/cov) to avoid wild regions
        z_train = model.encode(to_tensor(x_train, device)).detach().cpu().numpy()
        if latent_dim == 1:
            mu = float(np.mean(z_train[:, 0]))
            sd = float(np.std(z_train[:, 0]) + 1e-6)
            z_samp = np.random.default_rng(args.eval_seed + 7).normal(
                mu, sd, size=(args.eval_on_n, 1)
            ).astype(np.float32)
        else:
            mu = np.mean(z_train, axis=0)
            cov = np.cov(z_train.T) + 1e-6 * np.eye(latent_dim)
            z_samp = np.random.default_rng(args.eval_seed + 7).multivariate_normal(
                mu, cov, size=args.eval_on_n
            ).astype(np.float32)

        with torch.no_grad():
            x_dec = model.decode(to_tensor(z_samp, device)).cpu().numpy()

    def run_vae():
        model = VAE(in_dim=ds.dim, latent_dim=latent_dim).to(device)
        train_vae(model, x_train_t, cfg, device)

        # projection function uses mu (deterministic)
        def proj(x: torch.Tensor) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                mu, logvar = model.encode(x.to(device))
                x_hat = model.decode(mu)
                return x_hat

        # --- NEW: estimate threshold from val_on (ONLY on-manifold) ---
        thr = estimate_threshold(
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_val_on=x_val_on,
            threshold_method="percentile",
            threshold_q=args.threshold_q,
            device=device,
        )

        # --- NEW: evaluate on test_on + test_off using FIXED thr ---
        res, cache = eval_with_threshold(
            model_name=f"VAE/{ds.name}/z{latent_dim}",
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_test_on=x_on,  # 你这里的 x_on / x_off 看起来就是 test set
            x_test_off=x_off,
            thr=thr,
            device=device,
        )

        results["vae"] = (res, cache, model)

        # latent scatter (use mu)
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(to_tensor(x_eval, device))
            z = mu.cpu().numpy()

        # latent sampling -> decode
        # For VAE we can sample from standard normal (prior)
        rng = np.random.default_rng(args.eval_seed + 9)
        z_samp = rng.normal(size=(args.eval_on_n, latent_dim)).astype(np.float32)

        with torch.no_grad():
            x_dec = model.decode(to_tensor(z_samp, device)).cpu().numpy()

    if args.models in ("ae", "both"):
        run_ae()
    if args.models in ("vae", "both"):
        run_vae()

    ae_pack = None
    vae_pack = None

    # --- AE ---
    if "ae" in results:
        res, cache, model = results["ae"]
        with torch.no_grad():
            z_eval = model.encode(to_tensor(cache["x"], device)).detach().cpu().numpy()
            # latent sample decode（用之前生成的）
            z_train = model.encode(x_train_t).detach().cpu().numpy()
            mu = np.mean(z_train, axis=0)
            cov = np.cov(z_train.T) + 1e-6 * np.eye(latent_dim)
            z_samp = np.random.multivariate_normal(mu, cov, size=args.sample_latent_n).astype(np.float32)
            z_samp_t = to_tensor(z_samp, device)
            x_dec = model.decode(z_samp_t).detach().cpu().numpy()
        ae_pack = (cache, z_eval, x_dec, z_samp)

    # --- VAE ---
    if "vae" in results:
        res, cache, model = results["vae"]
        with torch.no_grad():
            mu, _ = model.encode(torch.tensor(cache["x"], dtype=torch.float32, device=device))
            z_eval = mu.detach().cpu().numpy()
            z_samp = np.random.randn(args.sample_latent_n, latent_dim).astype(np.float32)
            x_dec = model.decode(to_tensor(z_samp, device)).detach().cpu().numpy()
        vae_pack = (cache, z_eval, x_dec, z_samp)

    visualize_all(ds, x_train, ae_pack, vae_pack, latent_dim)
    if "vae" in results:
        _, _, vae_model = results["vae"]
        plot_conditional_decodes(
            ds=ds,
            x_on=x_on,
            model=vae_model,
            device=device,
            n_points=3,
            n_samples=30,
            seed=args.eval_seed + 17,
        )

    print("Done.")


if __name__ == "__main__":
    main()
