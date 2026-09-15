"""The learned temporal prior.

A logistic ranker over gap features. The central discipline of the paper is the
*locking protocol*: the prior is fit once on source data, then its coefficients
are frozen and never refit to a target outbreak. ``TemporalPrior.save`` /
``TemporalPrior.load`` persist the frozen coefficients as JSON so a saved prior
can be shipped and audited.

Scoring requires only numpy, so a locked prior can be evaluated anywhere without
scikit-learn installed.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FeatureSpec, transform


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class TemporalPrior:
    """Logistic temporal prior over candidate-infector gap features.

    Attributes
    ----------
    coef_ : np.ndarray
        Feature coefficients (length ``spec.n_features``).
    intercept_ : float
        Bias term.
    spec : FeatureSpec
        Feature configuration used at fit time. Must match at score time.
    locked : bool
        Set True once :meth:`lock` (or :meth:`load`) has been called; a locked
        prior refuses to be refit, enforcing the zero-shot protocol.
    """

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.locked: bool = False

    # -- training ---------------------------------------------------------
    def fit(self, dt: Sequence[float], y: Sequence[int], C: float = 1.0) -> "TemporalPrior":
        """Fit on source pairs. ``dt`` signed gaps, ``y`` in {0,1} (1 = true parent)."""
        if self.locked:
            raise RuntimeError(
                "This prior is locked. Fitting a locked prior would violate the "
                "zero-shot transfer protocol. Create a new TemporalPrior to refit."
            )
        from sklearn.linear_model import LogisticRegression  # local import: inference stays numpy-only

        X = transform(dt, self.spec)
        y = np.asarray(y, dtype=int).reshape(-1)
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X, y)
        self.coef_ = clf.coef_.reshape(-1).astype(float)
        self.intercept_ = float(clf.intercept_[0])
        return self

    def lock(self) -> "TemporalPrior":
        """Freeze the prior. Irreversible within this object."""
        if self.coef_ is None:
            raise RuntimeError("Cannot lock an unfitted prior.")
        self.locked = True
        return self

    # -- scoring ----------------------------------------------------------
    def score(self, dt: Sequence[float]) -> np.ndarray:
        """Plausibility score sigma(theta . x) for each signed gap."""
        if self.coef_ is None:
            raise RuntimeError("Prior is not fitted or loaded.")
        X = transform(dt, self.spec)
        return _sigmoid(X @ self.coef_ + self.intercept_)

    def curve(self, dt_grid: Sequence[float] | None = None, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(grid, score)`` for plotting the prior. Peak-normalized by default."""
        grid = np.asarray(dt_grid) if dt_grid is not None else np.linspace(-10, 60, 561)
        s = self.score(grid)
        if normalize and s.max() > 0:
            s = s / s.max()
        return grid, s

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.coef_ is None:
            raise RuntimeError("Nothing to save: prior is not fitted.")
        payload = {
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "feature_names": list(self.spec.names),
            "spec": {
                "window_edges": list(self.spec.window_edges),
                "include_window_indicators": self.spec.include_window_indicators,
            },
            "locked": True,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TemporalPrior":
        payload = json.loads(Path(path).read_text())
        spec = FeatureSpec(
            window_edges=tuple(payload["spec"]["window_edges"]),
            include_window_indicators=payload["spec"]["include_window_indicators"],
        )
        prior = cls(spec)
        prior.coef_ = np.asarray(payload["coef"], dtype=float)
        prior.intercept_ = float(payload["intercept"])
        prior.locked = bool(payload.get("locked", True))
        if len(prior.coef_) != spec.n_features:
            raise ValueError("Saved coefficients do not match feature spec length.")
        return prior
