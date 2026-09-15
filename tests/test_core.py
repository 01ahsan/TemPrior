import numpy as np
import pandas as pd
import pytest

import temprior as tp
from temprior.features import FeatureSpec, transform


def test_feature_shape():
    spec = FeatureSpec()
    X = transform([1, 5, -3, 20], spec)
    assert X.shape == (4, spec.n_features)
    # negative-gap indicator column (index 3) fires only for dt < 0
    assert list(X[:, 3]) == [0, 0, 1, 0]


def test_prior_calibrated_shape():
    p = tp.make_calibrated_prior()
    grid, s = p.curve()
    assert abs(grid[int(np.argmax(s))] - 20.5) < 1.0  # peak near 20.5 days


def test_prior_save_load_roundtrip(tmp_path):
    p = tp.make_calibrated_prior()
    f = tmp_path / "prior.json"
    p.save(f)
    q = tp.TemporalPrior.load(f)
    a = p.score([2, 10, 23, 40])
    b = q.score([2, 10, 23, 40])
    assert np.allclose(a, b)
    assert q.locked


def test_locked_prior_refuses_refit():
    p = tp.make_calibrated_prior()  # already locked
    with pytest.raises(RuntimeError):
        p.fit([1, 2, 3], [1, 0, 0])


def test_benchmark_unique_parent_and_window():
    ll = pd.DataFrame({"case_id": [0, 1, 2, 3], "onset": [0, 20, 25, 200]})
    ed = pd.DataFrame(
        {"parent_id": [0, 0, 2], "child_id": [1, 2, 3], "high_confidence": [True, True, True]}
    )
    bench = tp.build_benchmark(ll, ed, w_min=1, w_max=60)
    targets = {t.target for t in bench}
    assert 3 not in targets  # gap 175 > 60 -> excluded
    for t in bench:
        assert t.candidates.count(t.true_parent) == 1
        assert t.n_candidates >= 2


def test_evaluate_metrics_bounds():
    ll, ed = tp.make_example_outbreak()
    bench = tp.build_benchmark(ll, ed)
    r = tp.evaluate(tp.default_prior(), bench)
    assert 0.0 <= r.mrr <= 1.0
    assert 0.0 <= r.top1 <= r.top3 <= r.top5 <= 1.0
    assert r.n_tasks == len(bench)


def test_constant_scorer_is_chance():
    class Const:
        def score(self, dt):
            return np.ones(len(np.asarray(dt)))

    ll, ed = tp.make_example_outbreak()
    bench = tp.build_benchmark(ll, ed)
    r = tp.evaluate(Const(), bench)
    # average-rank tie handling => MRR strictly between 0 and 1, no lucky perfect score
    assert 0.0 < r.mrr < 1.0
    assert r.top1 < 1.0


def test_uncertainty_bounds():
    edges = [("a", "x"), ("a", "y"), ("b", "z")]
    aware = edges + [("a", "w"), ("c", "q")]
    exp = tp.expand_edges(edges, aware)
    inst = tp.decision_instability(exp.strict_offspring, exp.aware_offspring, k=2)
    assert 0.0 <= inst.jaccard <= 1.0
    assert 0.0 <= inst.decision_regret <= 1.0
    assert "c" in exp.newly_active_sources
