# Synthetic data notice

To respect data-governance requirements and rights-holder restrictions, all numerical recordings distributed in this repository are independently generated synthetic surrogates.

File organization, table schemas, anonymized subject identifiers, phase codes, trial identifiers, and modality organization are retained solely to preserve the functionality and reproducibility of the workshop. The distributed values are not experimental observations, do not correspond to individual animals, and must not be used for scientific inference or presented as research results. Original research data are not distributed by this repository.

The synthetic data are reproducible from [`scripts/generate_surrogate_data.py`](scripts/generate_surrogate_data.py). The generator uses a fixed random seed and does not read, transform, sample, perturb, or otherwise depend on the original recordings.

Earlier workshop copies may contain a previous data release. Attendees are asked not to redistribute those copies and to replace or delete them. Updating this repository cannot technically revoke copies, forks, caches, or files already saved elsewhere.

Questions about access to the original research data should be directed to the relevant KFO5001 investigators and institutional data stewards.
