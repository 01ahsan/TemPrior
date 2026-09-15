# Private research software policy

This repository is the general-purpose tool component of a two-part research
workflow.

## Allowed here

- General package implementation
- Synthetic example data
- Unit tests and local development configuration
- Documentation of the method at an implementation level

## Never commit here

- Real outbreak line lists, transmission edges, or derived records
- Locked study coefficients or private model artifacts
- Manuscript table and figure reproduction scripts
- Private statistical results or reviewer materials
- Personal, device, filesystem, credential, or institutional identifiers

The restricted reproduction repository remains the source of truth for the
study-specific analysis. This repository must remain private until the owner
deliberately approves a separate public release and performs a fresh audit.

Do not configure PyPI Trusted Publishing, public GitHub Pages, or a public
release workflow for this repository.
