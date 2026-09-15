# Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/01ahsan/TemPrior.git
cd TemPrior
pip install -e ".[test]"
pytest -q
```

- Open an issue before large changes so we can discuss the approach.
- Keep public code free of any real outbreak datasets or fitted study weights.
- Add or update tests for any behavior change; CI must pass on Python 3.9–3.12.
- Keep the locking protocol intact: a locked prior must never be refit.
