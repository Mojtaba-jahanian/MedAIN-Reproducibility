"""Real-image one-epoch ERM/MedAIN smoke test without framework dependencies.

This is a numerical pipeline gate, not a benchmark result. It trains a tiny
two-layer pixel classifier on consensus RIGA+ masks and verifies that the ERM
and MedAIN paths both complete with finite losses on actual images.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from medain_numpy import apply_medain, nuisance_basis


def load_case(row, dataset_root, size=64):
    root = Path(dataset_root)
    img = np.asarray(Image.open(root / row["image_relpath"]).convert("RGB").resize((size, size)), dtype=np.float64) / 255.0
    masks = []
    for p in row["mask_relpaths"].split(";"):
        masks.append(np.asarray(Image.open(root / p).resize((size, size), Image.Resampling.NEAREST)))
    stack = np.stack(masks)
    labels = np.stack([(stack == v).sum(0) for v in (0, 128, 255)]).argmax(0)
    return img.reshape(-1, 3), labels.reshape(-1)


def forward(x, params, basis=None, alpha=0.75):
    pre = x @ params["w1"] + params["b1"]
    h = np.maximum(pre, 0)
    if basis is not None:
        h = apply_medain(h, basis, alpha)
    logits = h @ params["w2"] + params["b2"]
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
    return pre, h, prob


def step(x, y, params, lr, basis=None, alpha=0.75):
    pre, h, prob = forward(x, params, basis, alpha)
    n = len(y)
    loss = -np.log(prob[np.arange(n), y] + 1e-12).mean()
    dlog = prob; dlog[np.arange(n), y] -= 1; dlog /= n
    dw2, db2 = h.T @ dlog, dlog.sum(0)
    dh = dlog @ params["w2"].T
    if basis is not None:
        dh = apply_medain(dh, basis, alpha)
    dpre = dh * (pre > 0)
    dw1, db1 = x.T @ dpre, dpre.sum(0)
    for key, grad in (("w1", dw1), ("b1", db1), ("w2", dw2), ("b2", db2)):
        params[key] -= lr * grad
    return float(loss)


def descriptor(x, params):
    pre, h, _ = forward(x, params)
    return h.mean(0)


def task_gradient(x, y, params):
    pre, _, prob = forward(x, params)
    dlog = prob; dlog[np.arange(len(y)), y] -= 1; dlog /= len(y)
    return ((dlog @ params["w2"].T) * (pre > 0)).mean(0)


def main(manifest_csv, dataset_root, output_json):
    rng = np.random.default_rng(1103)
    rows = list(csv.DictReader(open(manifest_csv, encoding="utf-8")))
    train = []
    for domain in ("Magrabia", "MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"):
        train += [r for r in rows if r["domain"] == domain][:8]
    test = [r for r in rows if r["domain"] == "BinRushed"][:8]
    data = [(load_case(r, dataset_root), r) for r in train]

    init = {
        "w1": rng.normal(0, 0.25, (3, 16)), "b1": np.zeros(16),
        "w2": rng.normal(0, 0.20, (16, 3)), "b2": np.zeros(3),
    }
    warm = {k: v.copy() for k, v in init.items()}
    warm_losses = []
    for (x, y), _ in data:
        idx = rng.choice(len(y), 2048, replace=False)
        warm_losses.append(step(x[idx], y[idx], warm, 0.08))

    acquisition, task = [], []
    for (x, y), _ in data:
        xp, xm = np.clip(x ** 0.8, 0, 1), np.clip(x ** 1.2, 0, 1)
        acquisition.append((descriptor(xp, warm) - descriptor(xm, warm)) / 0.4)
        idx = rng.choice(len(y), 2048, replace=False)
        task.append(task_gradient(x[idx], y[idx], warm))
    basis, eig, _, _ = nuisance_basis(np.stack(acquisition), np.stack(task), rank=2, beta=1e-3)

    erm = {k: v.copy() for k, v in warm.items()}
    med = {k: v.copy() for k, v in warm.items()}
    erm_losses, med_losses = [], []
    for (x, y), _ in data:
        idx = rng.choice(len(y), 2048, replace=False)
        erm_losses.append(step(x[idx], y[idx], erm, 0.05))
        med_losses.append(step(x[idx], y[idx], med, 0.05, basis))

    def evaluate(params, use_basis):
        losses, agreement = [], []
        for r in test:
            x, y = load_case(r, dataset_root)
            _, _, p = forward(x, params, basis if use_basis else None)
            losses.append(float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()))
            agreement.append(float((p.argmax(1) == y).mean()))
        return float(np.mean(losses)), float(np.mean(agreement))

    erm_test, erm_acc = evaluate(erm, False)
    med_test, med_acc = evaluate(med, True)
    projection_norm = float(np.mean([np.linalg.norm(descriptor(x, warm) @ basis) for (x, _), _ in data]))
    checks = {
        "real_images_loaded": len(data) == 32 and len(test) == 8,
        "six_masks_per_image": all(int(r["n_annotators"]) == 6 for _, r in data),
        "finite_erm_loss": bool(np.isfinite(erm_losses + [erm_test]).all()),
        "finite_medain_loss": bool(np.isfinite(med_losses + [med_test]).all()),
        "basis_orthonormal": bool(np.linalg.norm(basis.T @ basis - np.eye(2)) < 1e-10),
        "nonzero_projection": projection_norm > 1e-8,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "pipeline smoke test; not a benchmark result",
        "seed": 1103,
        "train_images": len(data), "held_out_real_images": len(test),
        "warmup_mean_loss": float(np.mean(warm_losses)),
        "erm_epoch_mean_loss": float(np.mean(erm_losses)),
        "medain_epoch_mean_loss": float(np.mean(med_losses)),
        "erm_heldout_loss": erm_test, "medain_heldout_loss": med_test,
        "erm_heldout_pixel_accuracy": erm_acc, "medain_heldout_pixel_accuracy": med_acc,
        "projection_norm": projection_norm,
        "top_generalized_eigenvalues": eig.tolist(),
        "checks": checks,
    }
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
