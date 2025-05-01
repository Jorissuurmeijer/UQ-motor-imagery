import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_uncertainty(p, mode="uncertainty"):
    if mode == "uncertainty":          # 1 − max‑probability, not 1 - for confidence
        return 1. - p.max(1)
    if mode == "confidence":
        return p.max(1)
    raise ValueError(f"unknown mode {mode}")


def accuracy_coverage_curve(y_true, p, u, step=0.01, min_points=5):
    N = len(u)
    order = np.argsort(u)              # 0 … N-1, low-to-high uncertainty

    coverages, accuracies, thresholds = [], [], []

    for k in range(0, N+1, max(1, int(step * N))):
        keep_idx = order[:N - k]       # the most-certain N-k samples
        n_kept   = len(keep_idx)
        cov      = n_kept / N

        if n_kept == 0:                # we have rejected every datapoint
            break

        # if n_kept < min_points:        # unreliable accuracy estimate
        #     continue

        acc = accuracy_score(
            y_true[keep_idx],
            p[keep_idx].argmax(axis=1)
        )

        thr = u[order[N - k - 1]] if n_kept > 0 else 0.0

        coverages.append(cov)
        accuracies.append(acc)
        thresholds.append(thr)


    return np.asarray(coverages), np.asarray(accuracies), np.asarray(thresholds)

