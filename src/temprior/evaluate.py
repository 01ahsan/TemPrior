"""Algorithm 2 (Phase 2) - zero-shot ranking evaluation.

Given any scorer exposing ``score(gaps) -> array`` and a benchmark of
:class:`~temprior.benchmark.RankTask`, compute MRR, Top-k accuracy,
NDCG, and mean true-parent rank. Ties are broken by average rank so a scorer
that returns constant scores gets the expected chance value rather than a lucky
best-case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .benchmark import RankTask


@dataclass
class EvalResult:
    mrr: float
    top1: float
    top3: float
    top5: float
    ndcg: float
    mean_rank: float
    n_tasks: int
    ranks: list  # per-task rank of the true parent (1-indexed)

    def as_dict(self) -> dict:
        return {
            "MRR": round(self.mrr, 4),
            "Top-1": round(self.top1, 4),
            "Top-3": round(self.top3, 4),
            "Top-5": round(self.top5, 4),
            "NDCG": round(self.ndcg, 4),
            "MeanRank": round(self.mean_rank, 3),
            "n_tasks": self.n_tasks,
        }


def _true_rank(scores: np.ndarray, true_idx: int) -> float:
    """Average-rank of the true parent (1-indexed), tie-aware."""
    s_true = scores[true_idx]
    greater = int(np.sum(scores > s_true))
    equal = int(np.sum(scores == s_true))  # includes the true one
    return greater + (equal + 1) / 2.0


def evaluate(scorer, benchmark: list[RankTask]) -> EvalResult:
    ranks: list[float] = []
    for task in benchmark:
        gaps = np.asarray(task.gaps, float)
        scores = np.asarray(scorer.score(gaps), float)
        true_idx = task.candidates.index(task.true_parent)
        ranks.append(_true_rank(scores, true_idx))
    r = np.asarray(ranks, float)
    n = len(r)
    return EvalResult(
        mrr=float(np.mean(1.0 / r)),
        top1=float(np.mean(r <= 1)),
        top3=float(np.mean(r <= 3)),
        top5=float(np.mean(r <= 5)),
        ndcg=float(np.mean(1.0 / np.log2(r + 1.0))),
        mean_rank=float(np.mean(r)),
        n_tasks=n,
        ranks=ranks,
    )


def rank_candidates(scorer, gaps, candidates) -> list[tuple]:
    """Return ``[(candidate, score, rank), ...]`` sorted best-first for one task."""
    scores = np.asarray(scorer.score(np.asarray(gaps, float)), float)
    order = np.argsort(-scores, kind="mergesort")
    out = []
    for rank, idx in enumerate(order, start=1):
        out.append((candidates[idx], float(scores[idx]), rank))
    return out
