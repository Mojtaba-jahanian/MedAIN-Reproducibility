"""Reference NumPy/SciPy implementation of the frozen MedAIN operator."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def trace_normalize(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    return cov / (np.trace(cov) + eps)


def covariance(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [samples, channels]")
    return vectors.T @ vectors / vectors.shape[0]


def nuisance_basis(
    acquisition_vectors: np.ndarray,
    task_vectors: np.ndarray,
    rank: int,
    beta: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ca = trace_normalize(covariance(acquisition_vectors))
    ct = trace_normalize(covariance(task_vectors))
    channels = ca.shape[0]
    if not 1 <= rank < channels:
        raise ValueError("rank must be in [1, channels-1]")
    metric = 0.5 * (ct + ct.T) + beta * np.eye(channels)
    eigenvalues, eigenvectors = eigh(0.5 * (ca + ca.T), metric)
    order = np.argsort(eigenvalues)[::-1][:rank]
    q, _ = np.linalg.qr(eigenvectors[:, order], mode="reduced")
    return q, eigenvalues[order], ca, ct


def apply_medain(features: np.ndarray, basis: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return features - alpha * (features @ basis) @ basis.T

