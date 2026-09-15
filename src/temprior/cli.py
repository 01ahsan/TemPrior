"""Command-line interface: ``temprior``.

Subcommands
-----------
rank      Rank candidate infectors for each case in a line list.
evaluate  Score the prior and all baselines on a benchmark built from the data.
uncertainty  Summarize label categories and top-k decision instability.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .baselines import fit_baselines
from .benchmark import build_benchmark
from .data import default_prior
from .evaluate import evaluate, rank_candidates
from .prior import TemporalPrior
from .uncertainty import categorize_labels, decision_instability, expand_edges, offspring_counts


def _load_prior(path: str | None) -> TemporalPrior:
    return TemporalPrior.load(path) if path else default_prior()


def _cmd_rank(args) -> None:
    linelist = pd.read_csv(args.linelist)
    edges = pd.read_csv(args.edges) if args.edges else None
    prior = _load_prior(args.prior)
    tasks = build_benchmark(linelist, edges, w_min=args.w_min, w_max=args.w_max) if edges is not None else []
    if edges is None:
        # No documented edges: rank every case against all admissible earlier cases.
        tasks = _open_tasks(linelist, args.w_min, args.w_max)
    rows = []
    for t in tasks:
        for cand, score, rank in rank_candidates(prior, t.gaps, t.candidates):
            rows.append({"target": t.target, "candidate": cand, "score": round(score, 4), "rank": rank})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} ranked rows for {len(tasks)} tasks -> {args.out}")


def _open_tasks(linelist, w_min, w_max):
    from .benchmark import RankTask

    onset = dict(zip(linelist["case_id"], pd.to_numeric(linelist["onset"], errors="coerce")))
    tasks = []
    for child, dc in onset.items():
        if pd.isna(dc):
            continue
        cands, gaps = [], []
        for cid, d in onset.items():
            if cid == child or pd.isna(d):
                continue
            g = dc - d
            if w_min <= g <= w_max:
                cands.append(cid)
                gaps.append(g)
        if cands:
            tasks.append(RankTask(child, cands, gaps, None))
    return tasks


def _cmd_evaluate(args) -> None:
    linelist = pd.read_csv(args.linelist)
    edges = pd.read_csv(args.edges)
    bench = build_benchmark(linelist, edges, w_min=args.w_min, w_max=args.w_max)
    prior = _load_prior(args.prior)
    src_gaps = [g for t in bench for g, c in zip(t.gaps, t.candidates) if c == t.true_parent]
    scorers = {"LearnedPrior": prior, **fit_baselines(src_gaps)}
    print(f"Benchmark: {len(bench)} tasks\n")
    print(f"{'method':<16}{'MRR':>8}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'NDCG':>8}{'MeanRank':>10}")
    for name, sc in scorers.items():
        r = evaluate(sc, bench).as_dict()
        print(f"{name:<16}{r['MRR']:>8}{r['Top-1']:>8}{r['Top-3']:>8}{r['Top-5']:>8}{r['NDCG']:>8}{r['MeanRank']:>10}")


def _cmd_uncertainty(args) -> None:
    edges = pd.read_csv(args.edges)
    if "confidence" in edges.columns:
        cats = categorize_labels(edges["confidence"].tolist())
        print("Label categories:", cats["counts"])
        print(f"Unresolved fraction: {cats['unresolved_fraction']:.4f}")
        strict = list(zip(edges.loc[edges["confidence"] >= 0.80, "parent_id"], edges.loc[edges["confidence"] >= 0.80, "child_id"]))
        aware = list(zip(edges.loc[edges["confidence"] >= 0.40, "parent_id"], edges.loc[edges["confidence"] >= 0.40, "child_id"]))
    else:
        col = "high_confidence"
        strict = list(zip(edges.loc[edges[col], "parent_id"], edges.loc[edges[col], "child_id"]))
        aware = list(zip(edges["parent_id"], edges["child_id"]))
    exp = expand_edges(strict, aware)
    inst = decision_instability(exp.strict_offspring, exp.aware_offspring, k=args.k)
    print(f"Offspring Gini: strict {exp.gini_strict:.3f} -> aware {exp.gini_aware:.3f}")
    print("Top-k instability:", inst.as_dict())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="temprior", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--w-min", type=float, default=1.0)
    common.add_argument("--w-max", type=float, default=60.0)
    common.add_argument("--prior", default=None, help="Path to locked prior JSON (default: bundled approx prior)")

    r = sub.add_parser("rank", parents=[common], help="Rank candidate infectors")
    r.add_argument("--linelist", required=True)
    r.add_argument("--edges", default=None)
    r.add_argument("--out", default="ranked.csv")
    r.set_defaults(func=_cmd_rank)

    e = sub.add_parser("evaluate", parents=[common], help="Evaluate prior vs baselines")
    e.add_argument("--linelist", required=True)
    e.add_argument("--edges", required=True)
    e.set_defaults(func=_cmd_evaluate)

    u = sub.add_parser("uncertainty", help="Label categories + decision instability")
    u.add_argument("--edges", required=True)
    u.add_argument("--k", type=int, default=5)
    u.set_defaults(func=_cmd_uncertainty)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
