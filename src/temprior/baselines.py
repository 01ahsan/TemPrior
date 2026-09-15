"""Fair, source-trained parametric baselines.

Each baseline is fit only on source serial gaps and never touches target data,
matching the comparison protocol in the paper. They all expose ``score(dt)`` so
they are drop-in interchangeable with :class:`TemporalPrior` in evaluation.

- Gaussian: normal density fit to all source gaps.
- KDE: Gaussian KDE (Scott's rule) fit to all source gaps.
- Gamma: gamma MLE on positive-only source gaps; 0 for dt <= 0.
- Lognormal: lognormal MLE on positive-only source gaps; 0 for dt <= 0.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


class GaussianBaseline:
    name = "Gaussian"

    def fit(self, gaps: Sequence[float]) -> "GaussianBaseline":
        g = np.asarray(gaps, float)
        self.mu, self.sd = float(g.mean()), float(g.std(ddof=1) or 1.0)
        return self

    def score(self, dt: Sequence[float]) -> np.ndarray:
        return stats.norm.pdf(np.asarray(dt, float), self.mu, self.sd)


class KDEBaseline:
    name = "KDE"

    def fit(self, gaps: Sequence[float]) -> "KDEBaseline":
        self.kde = stats.gaussian_kde(np.asarray(gaps, float))  # Scott's rule
        return self

    def score(self, dt: Sequence[float]) -> np.ndarray:
        return self.kde(np.asarray(dt, float))


class GammaBaseline:
    name = "Gamma"

    def fit(self, gaps: Sequence[float]) -> "GammaBaseline":
        pos = np.asarray(gaps, float)
        pos = pos[pos > 0]
        self.params = stats.gamma.fit(pos, floc=0)
        return self

    def score(self, dt: Sequence[float]) -> np.ndarray:
        dt = np.asarray(dt, float)
        s = stats.gamma.pdf(dt, *self.params)
        return np.where(dt > 0, s, 0.0)


class LognormalBaseline:
    name = "Lognormal"

    def fit(self, gaps: Sequence[float]) -> "LognormalBaseline":
        pos = np.asarray(gaps, float)
        pos = pos[pos > 0]
        self.params = stats.lognorm.fit(pos, floc=0)
        return self

    def score(self, dt: Sequence[float]) -> np.ndarray:
        dt = np.asarray(dt, float)
        s = stats.lognorm.pdf(dt, *self.params)
        return np.where(dt > 0, s, 0.0)


ALL_BASELINES = (GaussianBaseline, KDEBaseline, GammaBaseline, LognormalBaseline)


def fit_baselines(source_gaps: Sequence[float]) -> dict[str, object]:
    """Fit every baseline on the same source gaps; return name -> fitted scorer."""
    return {b.name: b().fit(source_gaps) for b in ALL_BASELINES}
