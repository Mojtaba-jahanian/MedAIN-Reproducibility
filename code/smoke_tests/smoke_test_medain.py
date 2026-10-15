"""Deterministic mechanism-level smoke test for MedAIN."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medain_numpy import apply_medain, nuisance_basis


def main() -> None:
    rng = np.random.default_rng(1103)
    channels, samples, rank, alpha = 32, 512, 4, 0.75
    nuisance_truth, _ = np.linalg.qr(rng.normal(size=(channels, rank)))
    task_raw = rng.normal(size=(channels, rank))
    task_raw -= nuisance_truth @ (nuisance_truth.T @ task_raw)
    task_truth, _ = np.linalg.qr(task_raw)

    acquisition = 3.0 * rng.normal(size=(samples, rank)) @ nuisance_truth.T
    acquisition += 0.10 * rng.normal(size=(samples, channels))
    task = 3.0 * rng.normal(size=(samples, rank)) @ task_truth.T
    task += 0.10 * rng.normal(size=(samples, channels))

    basis, eigenvalues, ca, ct = nuisance_basis(acquisition, task, rank, beta=1e-3)
    projector = basis @ basis.T
    overlap = np.linalg.norm(nuisance_truth.T @ basis, ord="fro") ** 2 / rank
    orth_error = np.linalg.norm(basis.T @ basis - np.eye(rank), ord="fro")

    nuisance_probe = rng.normal(size=(128, rank)) @ nuisance_truth.T
    task_probe = rng.normal(size=(128, rank)) @ task_truth.T
    nuisance_after = apply_medain(nuisance_probe, basis, alpha)
    task_after = apply_medain(task_probe, basis, alpha)
    nuisance_ratio = np.linalg.norm(nuisance_after) / np.linalg.norm(nuisance_probe)
    task_ratio = np.linalg.norm(task_after) / np.linalg.norm(task_probe)
    idempotence_error = np.linalg.norm(projector @ projector - projector, ord="fro")

    checks = {
        "finite_eigenvalues": bool(np.isfinite(eigenvalues).all()),
        "orthonormal_basis": bool(orth_error < 1e-10),
        "projector_idempotent": bool(idempotence_error < 1e-10),
        "recovers_nuisance_span": bool(overlap > 0.95),
        "suppresses_nuisance": bool(nuisance_ratio < 0.35),
        "preserves_task": bool(task_ratio > 0.95),
        "covariances_unit_trace": bool(abs(np.trace(ca)-1) < 1e-10 and abs(np.trace(ct)-1) < 1e-10),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": 1103,
        "channels": channels,
        "rank": rank,
        "alpha": alpha,
        "metrics": {
            "subspace_overlap": float(overlap),
            "nuisance_norm_ratio_after": float(nuisance_ratio),
            "task_norm_ratio_after": float(task_ratio),
            "orthonormality_error": float(orth_error),
            "projector_idempotence_error": float(idempotence_error),
            "top_generalized_eigenvalues": eigenvalues.tolist(),
        },
        "checks": checks,
    }
    out = Path(__file__).resolve().parents[1] / "audit" / "medain_mechanism_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

