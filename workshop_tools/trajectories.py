# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd

from workshop_tools import features, style


PHASE_ORDER = ["ext_1", "ext_2"]
FEATURE_LABELS = {
    "freezing_fraction": "Freezing fraction",
    "longest_freezing_bout_s": "Longest freezing bout (s)",
    "latency_to_freezing_s": "Latency to freezing (s)",
    "motion_median": "Median motion (a.u.)",
    "motion_iqr": "Motion IQR (a.u.)",
    "motion_p90": "Motion P90 (a.u.)",
    "speed_median_px_s": "Median speed (px/s)",
    "distance_px": "Distance (px)",
    "speed_p90_px_s": "Speed P90 (px/s)",
    "hr_median_bpm": "Median HR (bpm)",
    "hr_p10_bpm": "HR P10 (bpm)",
    "hr_p90_bpm": "HR P90 (bpm)",
    "hr_p90_minus_p10_bpm": "HR P90 - P10 (bpm)",
}


def _select_trajectory(epoch_features, feature, epoch):
    if feature not in epoch_features.columns:
        raise ValueError(f"Unknown trajectory feature: {feature!r}.")
    if epoch not in features.EPOCH_ORDER:
        raise ValueError(f"epoch must be one of {features.EPOCH_ORDER}.")

    selected = epoch_features.loc[
        epoch_features["epoch"] == epoch,
        ["subject_id", "phase", "trial", feature],
    ].copy()
    selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    return selected


def _feature_label(feature):
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _color_limits(values, feature):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if feature == "freezing_fraction":
        return 0.0, 1.0
    if feature == "latency_to_freezing_s":
        return 0.0, 10.0
    if values.empty:
        return 0.0, 1.0

    lower, upper = values.quantile([0.02, 0.98])
    if lower == upper:
        padding = abs(lower) * 0.05 or 1.0
        return lower - padding, upper + padding
    return float(lower), float(upper)


def plot_trial_heatmap(
    epoch_features,
    feature="freezing_fraction",
    epoch="Late-Tone",
    figsize=(12, 5.5),
):
    """Show subject-by-trial values for both extinction days."""
    from copy import copy

    import matplotlib.pyplot as plt

    selected = _select_trajectory(epoch_features, feature, epoch)
    phases = [phase for phase in PHASE_ORDER if phase in selected["phase"].unique()]
    if not phases:
        raise ValueError("No extinction phases are available for the heatmap.")

    subjects = sorted(selected["subject_id"].unique())
    vmin, vmax = _color_limits(selected[feature], feature)
    cmap = copy(plt.get_cmap("viridis"))
    cmap.set_bad("0.88")

    fig, axes = plt.subplots(
        1,
        len(phases),
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    image = None
    for ax, phase in zip(axes, phases):
        phase_values = selected.loc[selected["phase"] == phase]
        matrix = phase_values.pivot(
            index="subject_id", columns="trial", values=feature
        ).reindex(index=subjects, columns=range(1, 17))
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(style.PHASE_LABELS[phase])
        ax.set_xlabel("Trial")
        ax.set_xticks([0, 3, 7, 11, 15])
        ax.set_xticklabels([1, 4, 8, 12, 16])
        ax.set_yticks(np.arange(len(subjects)))
        ax.set_yticklabels(subjects, fontsize=7)

    axes[0].set_ylabel("Subject")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label(_feature_label(feature))
    fig.suptitle(f"{_feature_label(feature)} across trials - {epoch}")
    return fig, axes


def plot_subject_trajectory(
    epoch_features,
    subject_id,
    feature="freezing_fraction",
    epoch="Late-Tone",
    figsize=(10, 3.8),
):
    """Plot one subject continuously across both 16-trial extinction days."""
    import matplotlib.pyplot as plt

    selected = _select_trajectory(epoch_features, feature, epoch)
    selected = selected.loc[selected["subject_id"] == subject_id]
    if selected.empty:
        raise ValueError(f"No trajectory found for subject {subject_id!r}.")

    fig, ax = plt.subplots(figsize=figsize)
    data_found = False
    for phase_i, phase in enumerate(PHASE_ORDER):
        values = selected.loc[selected["phase"] == phase].sort_values("trial")
        if values.empty or values[feature].notna().sum() == 0:
            continue
        x = values["trial"].to_numpy() + phase_i * 16
        ax.plot(
            x,
            values[feature],
            color=style.PHASE_COLORS[phase],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=style.PHASE_LABELS[phase],
        )
        data_found = True

    if not data_found:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(16.5, color="0.45", linestyle="--", linewidth=0.9)
    ax.set_xlim(0.5, 32.5)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_xlabel("Extinction trial across both days")
    ax.set_ylabel(_feature_label(feature))
    ax.set_title(f"{subject_id} - {epoch}")
    if data_found:
        ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_subject_trajectories(
    epoch_features,
    feature="freezing_fraction",
    epoch="Late-Tone",
    ncols=4,
    figsize=(12, 9),
):
    """Show the same trajectory view in small multiples for every subject."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = _select_trajectory(epoch_features, feature, epoch)
    subjects = sorted(selected["subject_id"].unique())
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, subject_id in zip(axes.flat, subjects):
        subject_values = selected.loc[selected["subject_id"] == subject_id]
        data_found = False
        for phase_i, phase in enumerate(PHASE_ORDER):
            values = subject_values.loc[subject_values["phase"] == phase].sort_values(
                "trial"
            )
            if values.empty or values[feature].notna().sum() == 0:
                continue
            x = values["trial"].to_numpy() + phase_i * 16
            ax.plot(
                x,
                values[feature],
                color=style.PHASE_COLORS[phase],
                marker="o",
                markersize=2.1,
                linewidth=0.85,
            )
            data_found = True
        if not data_found:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.axvline(16.5, color="0.65", linestyle="--", linewidth=0.7)
        ax.set_title(subject_id, fontsize=9)
        ax.grid(axis="y", color="0.92", linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes.flat[len(subjects) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xticks([1, 8, 16, 24, 32])

    fig.legend(
        handles=[
            Line2D([0], [0], color=style.PHASE_COLORS[phase], label=style.PHASE_LABELS[phase])
            for phase in PHASE_ORDER
        ],
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"Individual {_feature_label(feature).lower()} trajectories - {epoch}",
        y=0.995,
    )
    fig.text(0.5, 0.01, "Extinction trial across both days", ha="center")
    fig.text(0.01, 0.5, _feature_label(feature), va="center", rotation="vertical")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.92))
    return fig, axes


def summarize_trajectories(
    epoch_features,
    feature="freezing_fraction",
    epoch="Late-Tone",
    early_trials=range(1, 5),
    late_trials=range(13, 17),
    min_block_trials=3,
    min_trend_trials=12,
):
    """Summarize each subject's initial, late, and across-trial response."""
    selected = _select_trajectory(epoch_features, feature, epoch)
    rows = []

    for (subject_id, phase), values in selected.groupby(["subject_id", "phase"]):
        values = values.sort_values("trial")
        valid = values.dropna(subset=[feature])
        early_values = values.loc[
            values["trial"].isin(early_trials), feature
        ].dropna()
        late_values = values.loc[values["trial"].isin(late_trials), feature].dropna()
        early = early_values.median() if len(early_values) >= min_block_trials else np.nan
        late = late_values.median() if len(late_values) >= min_block_trials else np.nan

        if len(valid) >= min_trend_trials:
            linear_trend = np.polyfit(valid["trial"], valid[feature], deg=1)[0]
            x = valid["trial"].to_numpy(dtype=float)
            y = valid[feature].to_numpy(dtype=float)
            trial_steps = np.diff(x)
            consecutive = trial_steps == 1
            if consecutive.any():
                interval_areas = trial_steps * (y[:-1] + y[1:]) / 2
                average_curve_level = interval_areas[consecutive].sum() / trial_steps[
                    consecutive
                ].sum()
                median_trial_change = np.median(np.abs(np.diff(y)[consecutive]))
            else:
                average_curve_level = np.nan
                median_trial_change = np.nan
        else:
            linear_trend = np.nan
            average_curve_level = np.nan
            median_trial_change = np.nan

        rows.append(
            {
                "subject_id": subject_id,
                "phase": phase,
                "feature": feature,
                "epoch": epoch,
                "early_block_median": early,
                "late_block_median": late,
                "late_minus_early": late - early,
                "linear_trend_per_trial": linear_trend,
                "average_curve_level": average_curve_level,
                "median_absolute_trial_change": median_trial_change,
                "valid_trial_count": int(valid[feature].notna().sum()),
                "early_block_valid_count": int(len(early_values)),
                "late_block_valid_count": int(len(late_values)),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["subject_id", "phase"]).reset_index(
        drop=True
    )
    early_day_2 = summary.loc[
        summary["phase"] == "ext_2", ["subject_id", "early_block_median"]
    ].set_index("subject_id")["early_block_median"]
    late_day_1 = summary.loc[
        summary["phase"] == "ext_1", ["subject_id", "late_block_median"]
    ].set_index("subject_id")["late_block_median"]
    between_day_shift = early_day_2 - late_day_1
    summary["day2_early_minus_day1_late"] = summary["subject_id"].map(
        between_day_shift
    )
    return summary
