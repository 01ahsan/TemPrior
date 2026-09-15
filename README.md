<div align="center">

# TemPrior

**Temporal prior for outbreak transmission reconstruction — zero-shot, uncertainty-aware, instantly deployable**

[![Tests](https://github.com/01ahsan/temprior/actions/workflows/ci.yml/badge.svg)](https://github.com/01ahsan/temprior/actions)
[![PyPI version](https://img.shields.io/pypi/v/temprior.svg)](https://pypi.org/project/temprior/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/temprior.svg)](https://pypi.org/project/temprior/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.30842-b31b1b.svg)](https://arxiv.org/abs/2606.30842)
[![Try it live »](https://img.shields.io/badge/demo-Try%20it%20live-brightgreen)](https://01ahsan.github.io/temprior)

</div>

---

When an outbreak begins, genomic sequencing takes days and phylodynamic pipelines need multiple high-quality sequences per case. **TemPrior answers *"who infected whom?"* from symptom-onset timing alone — in seconds, on any machine, with no sequences required.**

It learns a flexible temporal prior from multi-disease outbreak data, **locks it before seeing any target outbreak**, and ranks each case's most likely infectors zero-shot. It also makes explicit what most transmission-reconstruction tools ignore: real epidemiological labels are frequently uncertain, and that uncertainty changes which cases public-health teams would prioritize for isolation and contact tracing.

> **Reference paper:** Karim, M. A. (2026). *A Transferable Learned Temporal Prior for Transmission Reconstruction and Decision-Relevant Uncertainty in Real Outbreak Labels.* arXiv:2606.30842 · [PDF](https://arxiv.org/pdf/2606.30842)

---

## Why TemPrior

| | Standard parametric methods | TemPrior |
|---|---|---|
| Distribution assumption | Hand-specified (Gaussian / Gamma / Lognormal) | Learned from multi-disease data |
| Refits to target disease? | No | No — locked before any target data is seen |
| Handles label uncertainty? | No — labels treated as ground truth | Yes — strict / preferred / plausible / unresolved tiers |
| Decision-instability analysis | No | Yes — shows which top-*k* priorities shift under uncertainty |
| Sequences required? | No | No |

On real Andes virus (ANDV) outbreak data, TemPrior achieves **MRR 0.571 vs 0.274 for Gaussian** (permutation *p* ≤ 0.0002), identifying the true infector in the top 3 candidates for 76% of cases. The advantage is concentrated in the 16–40 day serial-gap regime where parametric models systematically underperform.

---

## Install

```bash
pip install temprior
```

Requires Python ≥ 3.9. No C extensions; pure Python with NumPy / SciPy / pandas / scikit-learn.

---

## Quickstart — five minutes to first result

### 1. Rank candidate infectors from a line list

```python
import temprior as tp

# Load your data (or use the built-in synthetic example)
linelist, edges = tp.make_example_outbreak()

# Build ranking tasks: every case ranked against admissible earlier cases
benchmark = tp.build_benchmark(linelist, edges, w_min=1, w_max=60)

# Rank with the learned prior
prior = tp.default_prior()
for task in benchmark[:3]:
    ranked = tp.rank_candidates(prior, task.gaps, task.candidates)
    top = ranked[0]
    print(f"Case {task.target}: top candidate = {top[0]}  score = {top[1]:.3f}")
```

### 2. Benchmark the prior against fair parametric baselines

```python
source_gaps = [g for t in benchmark for g, c
               in zip(t.gaps, t.candidates) if c == t.true_parent]
scorers = {"TemPrior": prior, **tp.fit_baselines(source_gaps)}

for name, scorer in scorers.items():
    r = tp.evaluate(scorer, benchmark).as_dict()
    print(f"{name:12s}  MRR={r['MRR']}  Top-1={r['Top-1']}  Top-3={r['Top-3']}")
```

### 3. Quantify label uncertainty and decision instability

```python
# Categorize transmission labels by confidence
cats = tp.categorize_labels(edges["confidence"].tolist())
print(f"Unresolved fraction: {cats['unresolved_fraction']:.1%}")

# Compare strict vs uncertainty-aware top-5 isolation shortlist
strict = list(zip(edges.loc[edges.confidence >= .8, "parent_id"],
                  edges.loc[edges.confidence >= .8, "child_id"]))
aware  = list(zip(edges.loc[edges.confidence >= .4, "parent_id"],
                  edges.loc[edges.confidence >= .4, "child_id"]))

exp  = tp.expand_edges(strict, aware)
inst = tp.decision_instability(exp.strict_offspring, exp.aware_offspring, k=5)
print(f"Top-5 Jaccard:       {inst.jaccard:.3f}")
print(f"Decision regret:     {inst.decision_regret:.1%}")
print(f"Newly elevated:      {inst.newly_elevated}")
```

### 4. CLI — run from the terminal on your own CSV files

```bash
# Rank candidate infectors and write results
temprior rank --linelist cases.csv --edges edges.csv --out ranked.csv

# Benchmark prior vs all baselines, print table
temprior evaluate --linelist cases.csv --edges edges.csv

# Label uncertainty + top-k decision instability
temprior uncertainty --edges edges.csv --k 5
```

---

## Input format

**Line list** (`cases.csv`):

| `case_id` | `onset` |
|-----------|---------|
| P1 | 2024-01-03 |
| P2 | 2024-01-26 |

`onset` can be a date string or a numeric day. Missing onsets are skipped.

**Edges** (`edges.csv`):

| `parent_id` | `child_id` | `high_confidence` | `confidence` |
|-------------|------------|-------------------|--------------|
| P1 | P2 | True | 0.91 |

`high_confidence` and `confidence` are optional. `temprior rank` works with no edges file at all — it ranks every case against all admissible earlier cases.

---

## When to use TemPrior

**Use TemPrior when:** genomic data is absent, delayed, or inconclusive (common for hantaviruses and other pathogens with low within-host diversity); an immediate ranking is needed in the first 24–48 hours of a response; or you want to quantify how label uncertainty affects intervention prioritization.

**Use genomic methods (outbreaker2, JUNIPER, BREATH) when:** multiple high-quality sequences per case are available with sufficient within-host diversity and there is time to run the pipeline. These are complementary regimes, not competing tools.

**TemPrior's advantage is concentrated in the 16–40 day serial-gap regime.** In short-gap settings (< 10 days, e.g. COVID-19 household clusters), parametric baselines remain competitive.

---

## Bringing your own locked prior

The bundled `default_prior()` approximates the published prior shape (peak ~20.5 days, 80% support ~3–38 days). To use your own trained weights:

```python
# Train on your source data
prior = tp.TemporalPrior()
prior.fit(source_gaps, labels)
prior.lock()            # enforces zero-shot protocol in code — fit() will raise after this
prior.save("my_prior.json")

# Load anywhere, no scikit-learn needed at inference time
prior = tp.TemporalPrior.load("my_prior.json")
```

---

## Correspondence to the paper

| Module | Paper element |
|--------|--------------|
| `benchmark.py` | Algorithm 1 — candidate-infector benchmark construction |
| `prior.py` | Learned temporal prior; locking protocol |
| `evaluate.py` | Algorithm 2 (Phase 2) — MRR / Top-k / NDCG evaluation |
| `baselines.py` | Fair source-trained Gaussian / KDE / Gamma / Lognormal |
| `uncertainty.py` | Label categories, edge expansion, top-k decision instability |

---

## Development

```bash
git clone https://github.com/01ahsan/temprior
cd temprior
pip install -e ".[test]"
pytest -q          # 8 tests, < 5 seconds
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Please open an issue before large changes.

---

## Citation

If you use TemPrior in your research, please cite:

```bibtex
@misc{karim2026transferablelearnedtemporalprior,
      title={A Transferable Learned Temporal Prior for Transmission Reconstruction and Decision-Relevant Uncertainty in Real Outbreak Labels}, 
      author={Md Ahsan Karim},
      year={2026},
      eprint={2606.30842},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.30842}, 
}
```

*Update to the peer-reviewed reference on journal publication.*

---
