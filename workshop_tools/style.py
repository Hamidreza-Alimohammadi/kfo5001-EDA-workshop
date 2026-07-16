# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import pandas as pd


SIGNAL_COLORS = {
    "freezing": "#3f3f3f",
    "motion": "#3274a1",
    "speed": "#4c956c",
    "hr": "#c44e52",
}

PHASE_COLORS = {
    "ext_1": "#3b78a8",
    "ext_2": "#3f8f68",
}
PHASE_LABELS = {
    "ext_1": "Extinction 1",
    "ext_2": "Extinction 2",
}

EPOCH_COLORS = ["0.88", "#f5d76e", "#e9a23b", "#b8d8c0"]
EPOCH_LABELS = ["Pre-Tone", "Early-Tone", "Late-Tone", "Post-Tone"]


def despine(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def std_limits(values, n_std=1.5, lower_bound=None):
    if n_std is None:
        return None
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return None
    mean = values.mean()
    std = values.std()
    if pd.isna(std) or std == 0:
        return None
    lower = mean - n_std * std
    upper = mean + n_std * std
    if lower_bound is not None:
        lower = max(lower, lower_bound)
    if lower >= upper:
        return None
    return lower, upper
