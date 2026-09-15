"""Temporal-gap feature transform for candidate-infector pairs.

A candidate edge is a (potential source i -> target j) pair. The signed serial
gap is ``dt = onset(j) - onset(i)``; a positive gap means the candidate source's
symptoms preceded the target's, which is the plausible direction of transmission.

The feature set mirrors the locked prior described in the paper:
signed gap, squared gap, log-abs gap, a negative-gap indicator, and a small set
of interval-window one-hot indicators over the admissible [w_min, w_max] window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DEFAULT_WINDOW_EDGES: tuple[float, ...] = (1.0, 10.0, 20.0, 30.0, 45.0, 60.0)


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for the gap feature transform.

    Parameters
    ----------
    window_edges:
        Monotonically increasing bin edges (in days) used to build one-hot
        window indicators. Consecutive edges define half-open bins ``[e_k, e_{k+1})``.
    include_window_indicators:
        If False, only the four core features (dt, dt^2, log|dt|, neg-indicator)
        are produced.
    """

    window_edges: tuple[float, ...] = DEFAULT_WINDOW_EDGES
    include_window_indicators: bool = True
    names: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        names = ["dt", "dt_sq", "log_abs_dt", "neg_gap"]
        if self.include_window_indicators:
            edges = self.window_edges
            names += [f"win_{edges[k]:.0f}_{edges[k+1]:.0f}" for k in range(len(edges) - 1)]
        object.__setattr__(self, "names", tuple(names))

    @property
    def n_features(self) -> int:
        return len(self.names)


def transform(dt: Sequence[float] | np.ndarray, spec: FeatureSpec | None = None) -> np.ndarray:
    """Map an array of signed gaps to the feature matrix ``X`` (n_pairs x n_features)."""
    spec = spec or FeatureSpec()
    dt = np.asarray(dt, dtype=float).reshape(-1)
    cols = [dt, dt ** 2, np.log(np.abs(dt) + 1.0), (dt < 0).astype(float)]
    if spec.include_window_indicators:
        edges = spec.window_edges
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            cols.append(((dt >= lo) & (dt < hi)).astype(float))
    return np.column_stack(cols)
