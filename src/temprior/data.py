"""Bundled default prior and a synthetic example outbreak.

The bundled ``default_prior.json`` is an APPROXIMATION calibrated to reproduce
the published prior *shape* (peak near ~20.5 days, broad 80% support ~3-38 days).
It is meant for demos and smoke tests. For exact reproduction of the paper's
numbers, replace it with the locked coefficients from the study repository:

    prior = TemporalPrior.load("path/to/your_locked_prior.json")
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureSpec
from .prior import TemporalPrior


def default_prior() -> TemporalPrior:
    """Load the bundled approximate prior."""
    with resources.as_file(resources.files("temprior") / "default_prior.json") as p:
        return TemporalPrior.load(p)


def make_calibrated_prior(
    peak: float = 20.5,
    spread: float = 13.5,
    neg_gap_penalty: float = -0.04,
) -> TemporalPrior:
    """Construct an approximate prior whose peak-normalized plausibility curve
    reproduces the published shape: a bump peaking at ``peak`` days with an 80%
    support window of roughly [3.25, 37.75] and strong suppression of negative gaps.

    The logit is set analytically to ``-(dt - peak)^2 / (2 * spread^2)`` plus a
    negative-gap penalty, which a logistic model represents exactly through the
    ``dt`` and ``dt^2`` features. This is a documented *approximation* of the
    published prior, not the locked weights; load the study's saved prior for
    exact reproduction.
    """
    spec = FeatureSpec(include_window_indicators=False)  # names: dt, dt_sq, log_abs_dt, neg_gap
    var = spread ** 2
    coef_dt = peak / var
    coef_dt2 = -1.0 / (2.0 * var)
    intercept = -(peak ** 2) / (2.0 * var)
    prior = TemporalPrior(spec)
    prior.coef_ = np.array([coef_dt, coef_dt2, 0.0, neg_gap_penalty], dtype=float)
    prior.intercept_ = float(intercept)
    return prior.lock()


def make_example_outbreak(n_cases: int = 40, seed: int = 7):
    """Return ``(linelist, edges)`` DataFrames for a small synthetic chain."""
    rng = np.random.default_rng(seed)
    onset = {0: 0.0}
    parent = {}
    order = [0]
    for cid in range(1, n_cases):
        src = int(rng.choice(order[-min(len(order), 12):]))  # attach to a recent case
        gap = float(np.clip(rng.normal(22, 8), 4, 45))
        onset[cid] = round(onset[src] + gap, 1)
        parent[cid] = src
        order.append(cid)
    linelist = pd.DataFrame({"case_id": list(onset), "onset": [onset[c] for c in onset]})
    edges = pd.DataFrame(
        {"parent_id": list(parent.values()), "child_id": list(parent.keys())}
    )
    edges["high_confidence"] = True
    # inject a few uncertain edges with a confidence column for the uncertainty demo
    conf = rng.uniform(0.2, 0.99, len(edges))
    edges["confidence"] = conf.round(3)
    return linelist, edges
