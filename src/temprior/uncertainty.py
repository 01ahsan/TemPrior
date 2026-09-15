"""Uncertainty layer.

Three capabilities matching the paper's uncertainty analysis:

1. ``categorize_labels`` - bucket edges into strict / preferred / plausible /
   unresolved by a confidence value, and summarize the unresolved fraction.
2. ``expand_edges`` - build strict vs uncertainty-aware edge sets and offspring
   (burden) counts per source, with an offspring Gini coefficient.
3. ``decision_instability`` - compare the strict top-k source shortlist against
   the uncertainty-aware one: Jaccard overlap, decision regret (missed burden),
   and which sources are newly elevated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- 1. label categories --------------------------------------------------
def categorize_labels(
    confidence: list[float],
    strict: float = 0.80,
    preferred: float = 0.60,
    plausible: float = 0.40,
) -> dict:
    """Bucket per-edge confidence values and report the unresolved fraction."""
    c = np.asarray(confidence, float)
    counts = {
        "strict": int(np.sum(c >= strict)),
        "preferred": int(np.sum((c >= preferred) & (c < strict))),
        "plausible": int(np.sum((c >= plausible) & (c < preferred))),
        "unresolved": int(np.sum(c < plausible)),
    }
    n = len(c)
    unresolved_frac = counts["unresolved"] / n if n else 0.0
    return {"counts": counts, "n": n, "unresolved_fraction": unresolved_frac}


# --- 2. edge expansion ----------------------------------------------------
def _gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * v.sum()) / (n * v.sum()))


def offspring_counts(edges: list[tuple]) -> dict:
    """Count offspring per source from ``(source, child)`` edges."""
    counts: dict = {}
    for src, _child in edges:
        counts[src] = counts.get(src, 0) + 1
    return counts


@dataclass
class ExpansionResult:
    strict_offspring: dict
    aware_offspring: dict
    gini_strict: float
    gini_aware: float
    newly_active_sources: list


def expand_edges(strict_edges: list[tuple], aware_edges: list[tuple]) -> ExpansionResult:
    """Compare a strict edge set to an uncertainty-aware (superset) edge set."""
    so = offspring_counts(strict_edges)
    ao = offspring_counts(aware_edges)
    newly = sorted(set(ao) - set(so), key=lambda k: (-ao[k], str(k)))
    return ExpansionResult(
        strict_offspring=so,
        aware_offspring=ao,
        gini_strict=_gini(np.array(list(so.values()) or [0.0])),
        gini_aware=_gini(np.array(list(ao.values()) or [0.0])),
        newly_active_sources=newly,
    )


# --- 3. decision instability ---------------------------------------------
def _topk(offspring: dict, k: int) -> list:
    return [s for s, _ in sorted(offspring.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]]


@dataclass
class InstabilityResult:
    strict_topk: list
    aware_topk: list
    jaccard: float
    decision_regret: float  # fraction of aware-top-k burden missed by strict-top-k
    missed_sources: list
    newly_elevated: list

    def as_dict(self) -> dict:
        return {
            "strict_topk": list(self.strict_topk),
            "aware_topk": list(self.aware_topk),
            "jaccard": round(self.jaccard, 3),
            "decision_regret": round(self.decision_regret, 4),
            "missed_sources": list(self.missed_sources),
            "newly_elevated": list(self.newly_elevated),
        }


def decision_instability(strict_offspring: dict, aware_offspring: dict, k: int = 5) -> InstabilityResult:
    """Compare fixed-capacity top-k shortlists under strict vs aware burden."""
    s_top = _topk(strict_offspring, k)
    a_top = _topk(aware_offspring, k)
    s_set, a_set = set(s_top), set(a_top)
    union = s_set | a_set
    jaccard = len(s_set & a_set) / len(union) if union else 1.0

    aware_burden = sum(aware_offspring.get(x, 0) for x in a_top)
    covered = sum(aware_offspring.get(x, 0) for x in a_top if x in s_set)
    regret = 1.0 - (covered / aware_burden) if aware_burden else 0.0

    missed = [x for x in a_top if x not in s_set]
    newly = [x for x in a_top if x not in s_set]
    return InstabilityResult(s_top, a_top, jaccard, regret, missed, newly)
