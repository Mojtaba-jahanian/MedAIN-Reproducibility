from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


ROOT = Path.home() / "MedAIN_V10"
STAGE9_ROOT = Path.home() / "MedAIN_V19_STAGE9"
REPAIR_ROOT = Path.home() / "MedAIN_V13_IDENTITY_REPAIR"
CACHE_MANIFEST = (
    REPAIR_ROOT / "manifests/riga_cache256_manifest_identitysafe.csv"
)
SPLIT_DIR = REPAIR_ROOT / "manifests"

N_CLASSES = 3
IGNORE_INDEX = 255


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_u01(text: str) -> float:
    h = hashlib.sha256(text.encode()).hexdigest()
    return int(h[:16], 16) / 2**64


class RigaDataset(Dataset):
    def __init__(
        self,
        rows,
        augment=False,
        seed=1103,
        epoch=0,
    ):
        self.rows = rows
        self.augment = augment
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _augment(self, x, y, sample_id):
        key = f"{self.seed}:{self.epoch}:{sample_id}"

        flip = stable_u01(key + ":flip") < 0.5
        angle = (
            stable_u01(key + ":angle") * 20.0 - 10.0
        )

        if flip:
            x = TF.hflip(x)
            y = TF.hflip(y)

        x = TF.rotate(
            x,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        y = TF.rotate(
            y.unsqueeze(0).float(),
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=float(IGNORE_INDEX),
        ).squeeze(0).long()

        return x, y

    def __getitem__(self, idx):
        r = self.rows[idx]

        ip = ROOT / r["cached_image"]
        mp = ROOT / r["cached_hard_mask"]

        with Image.open(ip) as im:
            x = TF.to_tensor(im.convert("RGB"))

        with Image.open(mp) as im:
            y = torch.from_numpy(
                np.array(im, dtype=np.uint8, copy=True)
            ).long()

        if self.augment:
            x, y = self._augment(
                x, y, r["sample_uid"]
            )

        return x, y, r["sample_uid"]


def load_rows(target_domain):
    split_file = SPLIT_DIR / f"riga_lodo_{target_domain}_identitysafe.csv"

    if not split_file.exists():
        raise RuntimeError(f"Missing identity-safe split manifest: {split_file}")

    with split_file.open(encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    if len(split_rows) != 744:
        raise RuntimeError(f"Expected 744 rows, got {len(split_rows)}")

    uids = [r["sample_uid"] for r in split_rows]
    if len(set(uids)) != 744:
        raise RuntimeError(f"sample_uid not unique: {len(set(uids))}/744")

    out = {s:[r for r in split_rows if r["split"]==s]
           for s in ("train","val","test")}

    sets={s:{r["sample_uid"] for r in out[s]}
          for s in ("train","val","test")}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("Identity-safe split overlap detected")

    for r in split_rows:
        if not (ROOT/r["cached_image"]).exists():
            raise RuntimeError(f"Missing corrected image: {r['cached_image']}")
        if not (ROOT/r["cached_hard_mask"]).exists():
            raise RuntimeError(f"Missing corrected mask: {r['cached_hard_mask']}")

    return out


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                cin, cout, 3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                cout, cout, 3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.conv = ConvBlock(
            cin + skip,
            cout,
        )

    def forward(self, x, s):
        x = F.interpolate(
            x,
            size=s.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(
            torch.cat([x, s], dim=1)
        )


class MedAINLayer(nn.Module):
    def __init__(
        self,
        channels,
        rank=4,
        alpha=0.75,
    ):
        super().__init__()

        self.channels = channels
        self.rank = rank
        self.alpha = float(alpha)

        self.register_buffer(
            "U",
            torch.zeros(channels, rank)
        )

        self.register_buffer(
            "initialized",
            torch.tensor(False)
        )

    @torch.no_grad()
    def set_basis(self, U):
        if U.shape != self.U.shape:
            raise ValueError(
                f"Basis shape {U.shape} "
                f"!= {self.U.shape}"
            )

        self.U.copy_(U)
        self.initialized.fill_(True)

    def forward(self, z):
        if not bool(self.initialized):
            return z

        n, c, h, w = z.shape

        flat = z.reshape(n, c, -1)

        coeff = torch.einsum(
            "cr,ncs->nrs",
            self.U,
            flat,
        )

        proj = torch.einsum(
            "cr,nrs->ncs",
            self.U,
            coeff,
        ).reshape_as(z)

        return z - self.alpha * proj


class ResNet18UNet(nn.Module):
    def __init__(
        self,
        method="erm",
        rank=4,
        alpha=0.75,
    ):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        self.method = method

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.med1 = MedAINLayer(
            64, rank, alpha
        )

        self.med2 = MedAINLayer(
            128, rank, alpha
        )

        self.up3 = UpBlock(
            512, 256, 256
        )

        self.up2 = UpBlock(
            256, 128, 128
        )

        self.up1 = UpBlock(
            128, 64, 64
        )

        self.up0 = UpBlock(
            64, 64, 64
        )

        self.final_conv = ConvBlock(
            64, 32
        )

        self.head = nn.Conv2d(
            32, N_CLASSES, 1
        )

        self.register_buffer(
            "img_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "img_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        return (
            (x - self.img_mean)
            / self.img_std
        )

    def raw_early_features(self, x):
        x = self.normalize(x)

        s0 = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.maxpool(s0)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)

        return s0, f1, f2

    def forward(
        self,
        x,
        apply_medain=True,
        return_raw_features=False,
    ):
        x = self.normalize(x)

        s0 = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        z = self.maxpool(s0)

        raw1 = self.layer1(z)

        f1 = raw1
        if (
            self.method == "medain"
            and apply_medain
        ):
            f1 = self.med1(f1)

        raw2 = self.layer2(f1)

        f2 = raw2
        if (
            self.method == "medain"
            and apply_medain
        ):
            f2 = self.med2(f2)

        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        d0 = self.up0(d1, s0)

        d = F.interpolate(
            d0,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        d = self.final_conv(d)
        logits = self.head(d)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if return_raw_features:
            return logits, raw1, raw2

        return logits


def foreground_dice_loss(
    logits,
    target,
):
    valid = target != IGNORE_INDEX

    safe = target.clone()
    safe[~valid] = 0

    probs = torch.softmax(logits, dim=1)

    onehot = F.one_hot(
        safe,
        num_classes=N_CLASSES
    ).permute(0, 3, 1, 2).float()

    valid4 = valid.unsqueeze(1).float()

    probs = probs * valid4
    onehot = onehot * valid4

    dices = []

    for c in range(1, N_CLASSES):
        p = probs[:, c]
        y = onehot[:, c]

        inter = (p * y).sum()

        den = p.sum() + y.sum()

        dice = (
            2.0 * inter + 1.0
        ) / (
            den + 1.0
        )

        dices.append(dice)

    return 1.0 - torch.stack(dices).mean()


def compute_train_weights(rows):
    counts = np.zeros(
        N_CLASSES,
        dtype=np.float64,
    )

    for r in rows:
        path = ROOT / r["cached_hard_mask"]

        with Image.open(path) as im:
            a = np.asarray(im)

        for c in range(N_CLASSES):
            counts[c] += np.sum(a == c)

    freq = counts / counts.sum()

    w = 1.0 / np.sqrt(
        np.maximum(freq, 1e-10)
    )

    w = w / w.mean()

    w = np.clip(w, 0.2, 3.0)

    print("Train class counts:", counts.tolist())
    print("Train class frequencies:", freq.tolist())
    print("CE weights:", w.tolist())

    return torch.tensor(
        w,
        dtype=torch.float32,
    )


def loss_fn(
    logits,
    target,
    class_weights,
):
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )

    dl = foreground_dice_loss(
        logits,
        target,
    )

    return ce + dl, ce.detach(), dl.detach()


class MetricAccumulator:
    def __init__(self):
        self.inter = np.zeros(
            N_CLASSES,
            dtype=np.float64,
        )

        self.pred = np.zeros(
            N_CLASSES,
            dtype=np.float64,
        )

        self.true = np.zeros(
            N_CLASSES,
            dtype=np.float64,
        )

    def update(
        self,
        logits,
        target,
    ):
        pred = logits.argmax(1)

        valid = (
            target != IGNORE_INDEX
        )

        for c in range(N_CLASSES):
            p = (pred == c) & valid
            y = (target == c) & valid

            self.inter[c] += (
                p & y
            ).sum().item()

            self.pred[c] += p.sum().item()
            self.true[c] += y.sum().item()

    def compute(self):
        result = {}

        fg_dice = []
        fg_iou = []

        for c in range(N_CLASSES):
            inter = self.inter[c]

            dice = (
                2 * inter + 1e-8
            ) / (
                self.pred[c]
                + self.true[c]
                + 1e-8
            )

            union = (
                self.pred[c]
                + self.true[c]
                - inter
            )

            iou = (
                inter + 1e-8
            ) / (
                union + 1e-8
            )

            result[f"dice_c{c}"] = float(dice)
            result[f"iou_c{c}"] = float(iou)

            if c > 0:
                fg_dice.append(dice)
                fg_iou.append(iou)

        result["mean_fg_dice"] = float(
            np.mean(fg_dice)
        )

        result["mean_fg_iou"] = float(
            np.mean(fg_iou)
        )

        return result


def make_loader(
    dataset,
    batch_size,
    shuffle,
    seed,
):
    gen = torch.Generator()
    gen.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def perturb_pair(
    x,
    kind,
    eps=0.15,
):
    if kind == "brightness":
        xp = torch.clamp(
            x * (1 + eps), 0, 1
        )

        xm = torch.clamp(
            x * (1 - eps), 0, 1
        )

    elif kind == "contrast":
        m = x.mean(
            dim=(2, 3),
            keepdim=True
        )

        xp = torch.clamp(
            (x - m) * (1 + eps) + m,
            0, 1
        )

        xm = torch.clamp(
            (x - m) * (1 - eps) + m,
            0, 1
        )

    elif kind == "gamma":
        xp = torch.clamp(
            x, 1e-6, 1
        ) ** (1 + eps)

        xm = torch.clamp(
            x, 1e-6, 1
        ) ** (1 - eps)

    elif kind == "sharpness":
        xp = TF.adjust_sharpness(
            x,
            1 + eps,
        )

        xm = TF.adjust_sharpness(
            x,
            max(0.0, 1 - eps),
        )

    else:
        raise ValueError(kind)

    return xp, xm


def covariance_normalize(C):
    C = 0.5 * (C + C.T)

    tr = np.trace(C)

    if tr <= 1e-12:
        return C

    return C / tr


def solve_basis(
    CA,
    CT,
    rank,
    beta,
):
    A = covariance_normalize(CA)
    T = covariance_normalize(CT)

    B = T + beta * np.eye(
        T.shape[0],
        dtype=np.float64,
    )

    eigvals, eigvecs = (
        scipy.linalg.eigh(
            A,
            B,
            check_finite=True,
        )
    )

    order = np.argsort(
        eigvals
    )[::-1]

    vals = eigvals[order[:rank]]
    U = eigvecs[:, order[:rank]]

    # Euclidean orthonormalization
    U, _ = np.linalg.qr(U)

    capture = float(
        np.trace(
            U.T @ A @ U
        )
        / max(
            np.trace(A),
            1e-12,
        )
    )

    leakage = float(
        np.trace(
            U.T @ T @ U
        )
        / max(
            np.trace(T),
            1e-12,
        )
    )

    orth = float(
        np.linalg.norm(
            U.T @ U
            - np.eye(rank),
            ord="fro",
        )
    )

    return (
        U.astype(np.float32),
        vals.astype(np.float64),
        capture,
        leakage,
        orth,
    )


def estimate_medain_basis(
    model,
    dataset,
    class_weights,
    device,
    epoch,
    rank=4,
    beta=1e-3,
    basis_samples=24,
):
    was_training = model.training
    model.eval()

    n = min(
        basis_samples,
        len(dataset),
    )

    rng = np.random.default_rng(
        900000 + epoch
    )

    idx = rng.choice(
        len(dataset),
        size=n,
        replace=False,
    ).tolist()

    subset = Subset(dataset, idx)

    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    CA1 = np.zeros(
        (64, 64),
        dtype=np.float64,
    )

    CA2 = np.zeros(
        (128, 128),
        dtype=np.float64,
    )

    CT1 = np.zeros(
        (64, 64),
        dtype=np.float64,
    )

    CT2 = np.zeros(
        (128, 128),
        dtype=np.float64,
    )

    nA = 0
    nT = 0

    perturbations = [
        "brightness",
        "contrast",
        "gamma",
        "sharpness",
    ]

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        model.zero_grad(
            set_to_none=True
        )

        logits, f1, f2 = model(
            x,
            apply_medain=False,
            return_raw_features=True,
        )

        loss, _, _ = loss_fn(
            logits,
            y,
            class_weights,
        )

        g1, g2 = torch.autograd.grad(
            loss,
            [f1, f2],
            retain_graph=False,
            create_graph=False,
        )

        t1 = (
            g1.mean(dim=(2, 3))
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        t2 = (
            g2.mean(dim=(2, 3))
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        CT1 += t1.T @ t1
        CT2 += t2.T @ t2
        nT += t1.shape[0]

        for kind in perturbations:
            xp, xm = perturb_pair(
                x,
                kind,
                eps=0.15,
            )

            with torch.no_grad():
                _, p1, p2 = (
                    model.raw_early_features(
                        xp
                    )
                )

                _, m1, m2 = (
                    model.raw_early_features(
                        xm
                    )
                )

                p1 = p1.mean(
                    dim=(2, 3)
                )

                m1 = m1.mean(
                    dim=(2, 3)
                )

                p2 = p2.mean(
                    dim=(2, 3)
                )

                m2 = m2.mean(
                    dim=(2, 3)
                )

                d1 = (
                    (p1 - m1)
                    / (2 * 0.15)
                )

                d2 = (
                    (p2 - m2)
                    / (2 * 0.15)
                )

            d1 = (
                d1.cpu()
                .double()
                .numpy()
            )

            d2 = (
                d2.cpu()
                .double()
                .numpy()
            )

            CA1 += d1.T @ d1
            CA2 += d2.T @ d2
            nA += d1.shape[0]

    CA1 /= max(nA, 1)
    CA2 /= max(nA, 1)

    CT1 /= max(nT, 1)
    CT2 /= max(nT, 1)

    (
        U1, ev1,
        cap1, leak1, orth1,
    ) = solve_basis(
        CA1, CT1, rank, beta
    )

    (
        U2, ev2,
        cap2, leak2, orth2,
    ) = solve_basis(
        CA2, CT2, rank, beta
    )

    model.med1.set_basis(
        torch.from_numpy(U1).to(device)
    )

    model.med2.set_basis(
        torch.from_numpy(U2).to(device)
    )

    if was_training:
        model.train()

    return {
        "epoch": epoch,
        "rank": rank,
        "beta": beta,
        "basis_samples": n,
        "stage1": {
            "top_generalized_eigenvalues":
                ev1.tolist(),
            "acquisition_capture_ratio":
                cap1,
            "task_leakage_ratio":
                leak1,
            "orthogonality_error":
                orth1,
        },
        "stage2": {
            "top_generalized_eigenvalues":
                ev2.tolist(),
            "acquisition_capture_ratio":
                cap2,
            "task_leakage_ratio":
                leak2,
            "orthogonality_error":
                orth2,
        },
    }


@torch.no_grad()
def evaluate(
    model,
    dataset,
    class_weights,
    device,
    batch_size=8,
):
    loader = make_loader(
        dataset,
        batch_size,
        shuffle=False,
        seed=0,
    )

    model.eval()

    acc = MetricAccumulator()

    loss_sum = 0.0
    ce_sum = 0.0
    dice_loss_sum = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss, ce, dl = loss_fn(
            logits,
            y,
            class_weights,
        )

        bs = x.shape[0]

        loss_sum += loss.item() * bs
        ce_sum += ce.item() * bs
        dice_loss_sum += dl.item() * bs
        n += bs

        acc.update(
            logits,
            y,
        )

    m = acc.compute()

    m.update({
        "loss": loss_sum / n,
        "ce_loss": ce_sum / n,
        "dice_loss": dice_loss_sum / n,
        "n_samples": n,
    })

    return m


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--method",
        choices=["erm"],
        required=True,
    )

    ap.add_argument(
        "--target-domain",
        required=True,
        choices=[
            "BinRushed",
            "Magrabia",
            "MESSIDOR_Base1",
            "MESSIDOR_Base2",
            "MESSIDOR_Base3",
        ],
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1103,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--threads",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--rank",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.75,
    )

    ap.add_argument(
        "--beta",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--basis-update-every",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--basis-samples",
        type=int,
        default=24,
    )

    ap.add_argument(
        "--preflight",
        action="store_true",
    )

    ap.add_argument(
        "--skip-test",
        action="store_true",
    )

    args = ap.parse_args()

    if not args.preflight:
        if args.seed != 1103:
            raise RuntimeError("Stage9A pilot anchor seed is locked to 1103")
        if args.rank != 2 or abs(args.alpha - 0.5) > 1e-12:
            raise RuntimeError("Stage9A pilot anchor rank/alpha locked to r=2, alpha=0.5")
        if not args.skip_test:
            raise RuntimeError("Stage9A pilot forbids held-out target test evaluation")

    torch.set_num_threads(
        args.threads
    )

    torch.set_num_interop_threads(1)

    seed_everything(args.seed)

    device = torch.device("cpu")

    rows = load_rows(args.target_domain)

    print(
        "Split sizes:",
        {
            k: len(v)
            for k, v in rows.items()
        }
    )

    train_ds = RigaDataset(
        rows["train"],
        augment=True,
        seed=args.seed,
    )

    basis_ds = RigaDataset(
        rows["train"],
        augment=False,
        seed=args.seed,
    )

    val_ds = RigaDataset(
        rows["val"],
        augment=False,
        seed=args.seed,
    )

    test_ds = RigaDataset(
        rows["test"],
        augment=False,
        seed=args.seed,
    )

    class_weights = (
        compute_train_weights(
            rows["train"]
        ).to(device)
    )

    seed_everything(args.seed)

    model = ResNet18UNet(
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
    ).to(device)

    x0, y0, _ = next(
        iter(
            DataLoader(
                train_ds,
                batch_size=2,
                shuffle=False,
            )
        )
    )

    with torch.no_grad():
        z0 = model(
            x0.to(device)
        )

    print(
        "Preflight tensor shapes:",
        tuple(x0.shape),
        tuple(y0.shape),
        tuple(z0.shape),
    )

    if z0.shape != (
        2, N_CLASSES, 256, 256
    ):
        raise RuntimeError(
            f"Unexpected logits shape {z0.shape}"
        )

    if args.preflight:
        print("PREFLIGHT: PASS")
        return

    alpha_tag = str(args.alpha).replace(".", "p")

    run_name = (
        f"stage9a_anchor_{args.method}_"
        f"{args.target_domain}_"
        f"seed{args.seed}_"
        f"r{args.rank}_"
        f"a{alpha_tag}"
    )

    out_dir = (
        STAGE9_ROOT
        / "anchors/results"
        / run_name
    )

    ckpt_dir = (
        STAGE9_ROOT
        / "anchors/checkpoints"
        / run_name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(args).copy()

    train_uid_order = [r["sample_uid"] for r in rows["train"]]
    train_uid_order_sha256 = hashlib.sha256(
        ("\n".join(train_uid_order) + "\n").encode("utf-8")
    ).hexdigest()

    config.update({
        "stage": "MedAIN_Stage9A_Prospective_Pilot_Anchor",
        "target_domain": args.target_domain,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "device": str(device),
        "torch_version": torch.__version__,
        "target_test_evaluated": False,
        "external_data_used": False,
        "train_uid_order_sha256": train_uid_order_sha256,
        "protocol_lock": "STAGE9A_PILOT_PROTOCOL_LOCK_v1_0.json",
    })

    save_json(
        out_dir / "config.json",
        config,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    history = []
    basis_history = []

    last_state_path = ckpt_dir / "last_state.pt"
    start_epoch = 0

    if last_state_path.exists():
        state = torch.load(
            last_state_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            state["model_state"]
        )

        optimizer.load_state_dict(
            state["optimizer_state"]
        )

        scheduler.load_state_dict(
            state["scheduler_state"]
        )

        best_score = float(
            state["best_score"]
        )

        best_epoch = int(
            state["best_epoch"]
        )

        bad_epochs = int(
            state["bad_epochs"]
        )

        history = state.get(
            "history", []
        )

        basis_history = state.get(
            "basis_history", []
        )

        start_epoch = int(
            state["epoch"]
        ) + 1

        print(
            f"RESUME: epoch {start_epoch}, "
            f"best_epoch={best_epoch}, "
            f"best_val={best_score:.6f}"
        )

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        if (
            args.method == "medain"
            and (
                epoch == 0
                or epoch
                % args.basis_update_every
                == 0
            )
        ):
            print(
                f"[Epoch {epoch:03d}] "
                "updating MedAIN basis ..."
            )

            diag = estimate_medain_basis(
                model,
                basis_ds,
                class_weights,
                device,
                epoch,
                rank=args.rank,
                beta=args.beta,
                basis_samples=args.basis_samples,
            )

            basis_history.append(diag)

            with (
                out_dir
                / "basis_history.jsonl"
            ).open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(diag)
                    + "\n"
                )

            print(
                "Basis stage1:",
                diag["stage1"],
            )

            print(
                "Basis stage2:",
                diag["stage2"],
            )

        train_ds.set_epoch(epoch)

        train_loader = make_loader(
            train_ds,
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        )

        model.train()

        train_loss = 0.0
        train_n = 0

        train_metrics = MetricAccumulator()

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss, _, _ = loss_fn(
                logits,
                y,
                class_weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            bs = x.shape[0]

            train_loss += (
                loss.item() * bs
            )

            train_n += bs

            train_metrics.update(
                logits.detach(),
                y,
            )

        scheduler.step()

        trm = train_metrics.compute()

        valm = evaluate(
            model,
            val_ds,
            class_weights,
            device,
            batch_size=args.batch_size,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss":
                train_loss / train_n,
            "train_mean_fg_dice":
                trm["mean_fg_dice"],
            "train_mean_fg_iou":
                trm["mean_fg_iou"],
            "val_loss":
                valm["loss"],
            "val_mean_fg_dice":
                valm["mean_fg_dice"],
            "val_mean_fg_iou":
                valm["mean_fg_iou"],
            "val_dice_c1":
                valm["dice_c1"],
            "val_dice_c2":
                valm["dice_c2"],
            "seconds":
                time.time() - epoch_start,
        }

        history.append(row)

        print(
            json.dumps(row)
        )

        hist_path = (
            out_dir
            / "training_history.csv"
        )

        with hist_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    history[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(history)

        score = (
            valm["mean_fg_dice"]
        )

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0

            torch.save({
                "model_state":
                    model.state_dict(),
                "epoch":
                    epoch,
                "best_val_mean_fg_dice":
                    best_score,
                "config":
                    config,
            },
                ckpt_dir
                / "best_model.pt"
            )

        else:
            bad_epochs += 1

        # Atomic full-state checkpoint after every completed epoch.
        tmp_state_path = ckpt_dir / "last_state.tmp"

        torch.save({
            "model_state":
                model.state_dict(),
            "optimizer_state":
                optimizer.state_dict(),
            "scheduler_state":
                scheduler.state_dict(),
            "epoch":
                epoch,
            "best_score":
                best_score,
            "best_epoch":
                best_epoch,
            "bad_epochs":
                bad_epochs,
            "history":
                history,
            "basis_history":
                basis_history,
            "config":
                config,
        }, tmp_state_path)

        os.replace(
            tmp_state_path,
            last_state_path,
        )

        print(
            f"CHECKPOINT: epoch {epoch} saved",
            flush=True,
        )

        if bad_epochs >= args.patience:
            print(
                f"Early stopping at "
                f"epoch {epoch}; "
                f"best epoch={best_epoch}"
            )
            break

    checkpoint = torch.load(
        ckpt_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    val_best = evaluate(
        model,
        val_ds,
        class_weights,
        device,
        batch_size=args.batch_size,
    )

    # Stage 6B development runs are validation-only.
    # Target test is deliberately not touched when --skip-test.
    if args.skip_test:
        test_metrics = None
    else:
        test_metrics = evaluate(
            model,
            test_ds,
            class_weights,
            device,
            batch_size=args.batch_size,
        )

    total_seconds = (
        time.time() - start_time
    )

    summary = {
        "status": "PASS",
        "run_name": run_name,
        "method": args.method,
        "seed": args.seed,
        "target_domain": args.target_domain,
        "best_epoch": best_epoch,
        "best_val_mean_fg_dice":
            best_score,
        "validation_at_best":
            val_best,
        "test":
            test_metrics,
        "training_seconds":
            total_seconds,
        "train_n":
            len(train_ds),
        "val_n":
            len(val_ds),
        "test_n":
            len(test_ds),
        "basis_updates":
            len(basis_history),
    }

    save_json(
        out_dir / "summary.json",
        summary,
    )

    print()
    print(
        "============== FINAL SUMMARY =============="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "==========================================="
    )


if __name__ == "__main__":
    main()
