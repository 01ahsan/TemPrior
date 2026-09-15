# Publishing TemPrior

Production and test publishing use separate Trusted Publishers:

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| Project | temprior | temprior |
| Owner | 01ahsan | 01ahsan |
| Repository | TemPrior | TemPrior |
| Workflow filename | release.yml | testpypi.yml |
| GitHub environment | pypi | testpypi |

Register the production pending publisher at https://pypi.org/manage/account/publishing/
and create the `pypi` environment in GitHub before pushing a stable tag.
No stored API token is needed. A pending publisher does not reserve a project name.

Set matching versions in `pyproject.toml`, `src/temprior/__init__.py`, and
`CITATION.cff`. Confirm the author is Md Ahsan Karim in metadata and LICENSE.
Commit and push main, then create the release tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The production workflow checks the tag against the package version, builds and
checks both distributions, and publishes through the `pypi` environment. It then
creates a draft GitHub Release with both distributions attached.

After publication, verify in a clean environment:

```bash
python -m pip install temprior==0.1.0
temprior --help
```

Keep real outbreak data and locked study weights in the private reproduction
repository. Only synthetic examples and the analytic approximation belong here.
If authentication fails before any upload, correct the publisher and rerun the
failed job; a new version is not necessary. Never replace an existing release tag.
