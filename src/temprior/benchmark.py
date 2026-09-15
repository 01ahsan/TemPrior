"""Algorithm 1 - candidate-infector ranking benchmark construction.

Turns a line list plus a documented (high-confidence) transmission tree into a
set of ranking tasks. For every child with a documented parent, the candidate
set is all cases whose onset falls inside the admissible window before the child.
Tasks with fewer than two candidates, or without exactly one true parent in the
candidate set, are dropped - matching the strict unique-parent criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RankTask:
    target: object
    candidates: list  # candidate source ids
    gaps: list  # signed gap for each candidate (onset[target] - onset[cand])
    true_parent: object

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def build_benchmark(
    linelist: pd.DataFrame,
    edges: pd.DataFrame,
    w_min: float = 1.0,
    w_max: float = 60.0,
    id_col: str = "case_id",
    onset_col: str = "onset",
    parent_col: str = "parent_id",
    child_col: str = "child_id",
    confidence_col: str | None = "high_confidence",
) -> list[RankTask]:
    """Construct ranking tasks.

    Parameters
    ----------
    linelist:
        Columns: ``id_col`` (unique case id), ``onset_col`` (numeric day or
        date-parseable). Missing onsets are allowed and handled by skipping.
    edges:
        Documented parent->child edges. Columns: ``parent_col``, ``child_col``,
        and optionally ``confidence_col`` (truthy = high-confidence).
    """
    onset = _onset_map(linelist, id_col, onset_col)
    parent_of = _true_parents(edges, parent_col, child_col, confidence_col)

    tasks: list[RankTask] = []
    for child, parent in parent_of.items():
        dc, dp = onset.get(child), onset.get(parent)
        if dc is None or dp is None:  # missing onset for child or documented parent
            continue
        gp = dc - dp
        if not (w_min <= gp <= w_max):  # timing criterion on the true edge
            continue
        cands, gaps = [], []
        for cid, d in onset.items():
            if cid == child or d is None:
                continue
            g = dc - d
            if w_min <= g <= w_max:
                cands.append(cid)
                gaps.append(g)
        if len(cands) < 2:  # ranking needs at least two candidates
            continue
        if cands.count(parent) != 1:  # exactly one true parent must be present
            continue
        tasks.append(RankTask(child, cands, gaps, parent))
    return tasks


def _onset_map(linelist: pd.DataFrame, id_col: str, onset_col: str) -> dict:
    s = linelist[onset_col]
    if not pd.api.types.is_numeric_dtype(s):
        parsed = pd.to_datetime(s, errors="coerce")
        base = parsed.min()
        s = (parsed - base).dt.days
    out = {}
    for cid, val in zip(linelist[id_col], s):
        out[cid] = None if pd.isna(val) else float(val)
    return out


def _true_parents(edges: pd.DataFrame, parent_col: str, child_col: str, confidence_col: str | None) -> dict:
    df = edges
    if confidence_col and confidence_col in edges.columns:
        df = edges[edges[confidence_col].astype(bool)]
    parent_of: dict = {}
    for _, row in df.iterrows():
        child = row[child_col]
        if child in parent_of:  # keep strict unique-parent: drop children with >1 hi-conf parent
            parent_of[child] = None
        else:
            parent_of[child] = row[parent_col]
    return {c: p for c, p in parent_of.items() if p is not None}
