"""TemPrior - a transferable learned temporal prior for outbreak
transmission reconstruction with decision-relevant uncertainty.

Reference implementation of the method described in the accompanying paper.
"""
from .features import FeatureSpec, transform
from .prior import TemporalPrior
from .baselines import (
    GaussianBaseline,
    KDEBaseline,
    GammaBaseline,
    LognormalBaseline,
    fit_baselines,
)
from .benchmark import RankTask, build_benchmark
from .evaluate import EvalResult, evaluate, rank_candidates
from .uncertainty import (
    categorize_labels,
    expand_edges,
    offspring_counts,
    decision_instability,
)
from .data import default_prior, make_calibrated_prior, make_example_outbreak

__version__ = "0.1.0"

__all__ = [
    "FeatureSpec",
    "transform",
    "TemporalPrior",
    "GaussianBaseline",
    "KDEBaseline",
    "GammaBaseline",
    "LognormalBaseline",
    "fit_baselines",
    "RankTask",
    "build_benchmark",
    "EvalResult",
    "evaluate",
    "rank_candidates",
    "categorize_labels",
    "expand_edges",
    "offspring_counts",
    "decision_instability",
    "default_prior",
    "make_calibrated_prior",
    "make_example_outbreak",
]
