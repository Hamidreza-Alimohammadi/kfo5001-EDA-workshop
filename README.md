# KFO5001 EDA Workshop

Companion repository for **A Hands-On Session on Exploratory Data Analysis: From Patterns to Insight**.

> **Data status:** Every distributed numerical recording is an independently generated synthetic surrogate. The files preserve the workshop schema and anonymous identifiers, but contain no experimental observations and must not be used for scientific inference. See [DATA_NOTICE.md](DATA_NOTICE.md).

## Open in Colab

[Open the workshop notebook in Google Colab](https://colab.research.google.com/github/Hamidreza-Alimohammadi/kfo5001-EDA-workshop/blob/main/KFO5001_EDA_Workshop.ipynb)

Save a personal copy in Drive before editing. The notebook installs or imports the included helper modules and loads the synthetic teaching data automatically.

## Repository structure

- `KFO5001_EDA_Workshop.ipynb`: the five-section hands-on workshop.
- `workshop_tools/`: reusable loading, visualization, feature, trajectory, and pattern helpers.
- `data/reduced/`: synthetic session tables and prepared extinction features.
- `scripts/generate_surrogate_data.py`: deterministic generator for every distributed data value.
- `tests/`: scientific-rule, interface, and notebook smoke tests.
- `environment.yml`: minimal local Conda environment.

## Run locally

```bash
conda env create -f environment.yml
conda activate kfo5001-eda
jupyter lab KFO5001_EDA_Workshop.ipynb
```

Regenerate the synthetic data with the published seed:

```bash
python scripts/generate_surrogate_data.py
```

## Rights

Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders. All rights reserved. Limited workshop and personal educational evaluation permission is described in [COPYRIGHT.md](COPYRIGHT.md); reuse beyond those terms requires prior written permission.
