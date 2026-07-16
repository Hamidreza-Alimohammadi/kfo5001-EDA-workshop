# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd

from workshop_tools import style
from workshop_tools.session_data import load_session_files, validate_phases
from workshop_tools.signals import freezing_bouts


EPOCH_ORDER = ["Pre-Tone", "Early-Tone", "Late-Tone", "Post-Tone"]
CONTRAST_SPECS = {
    "Early-Tone minus Pre-Tone": ("Early-Tone", "Pre-Tone"),
    "Late-Tone minus Early-Tone": ("Late-Tone", "Early-Tone"),
    "Post-Tone minus Late-Tone": ("Post-Tone", "Late-Tone"),
    "Post-Tone minus Pre-Tone": ("Post-Tone", "Pre-Tone"),
}
CONTRAST_FEATURES = [
    "freezing_fraction",
    "motion_median",
    "speed_median_px_s",
    "hr_median_bpm",
]
OVERVIEW_FEATURES = [
    ("freezing_fraction", "Freezing fraction"),
    ("motion_median", "Median motion (a.u.)"),
    ("speed_median_px_s", "Median speed (px/s)"),
    ("hr_median_bpm", "Median HR (bpm)"),
]


def load_prepared_features(path="data/reduced/extinction_epoch_features.csv.gz"):
    """Load the workshop's cached epoch table with its intended data types."""
    features = pd.read_csv(path)
    validate_phases(features["phase"].dropna().unique())
    unknown_epochs = sorted(set(features["epoch"].dropna()) - set(EPOCH_ORDER))
    if unknown_epochs:
        raise ValueError(f"Unknown epochs: {unknown_epochs}. Choose from {EPOCH_ORDER}.")
    features["epoch"] = pd.Categorical(
        features["epoch"], categories=EPOCH_ORDER, ordered=True
    )
    if "tone_freezing_observed" in features:
        observed = features["tone_freezing_observed"].map(
            {"True": True, "False": False, True: True, False: False}
        )
        features["tone_freezing_observed"] = observed.astype("boolean")
    return features


def _percentile(values, q):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def _add_speed(tracking):
    tracking = tracking.copy()
    dt = tracking["time_s"].diff()
    step_distance = np.hypot(tracking["x"].diff(), tracking["y"].diff())
    valid_step = dt.gt(0) & step_distance.notna()
    tracking["step_distance_px"] = step_distance.where(valid_step)
    tracking["speed_px_s"] = (step_distance / dt).where(valid_step)
    tracking["previous_time_s"] = tracking["time_s"].shift()
    return tracking


def _epoch_specs(tone, before_s, after_s):
    tone_start = float(tone["start_s"])
    tone_end = float(tone["end_s"])
    tone_mid = tone_start + (tone_end - tone_start) / 2
    return [
        ("Pre-Tone", tone_start - before_s, tone_start),
        ("Early-Tone", tone_start, tone_mid),
        ("Late-Tone", tone_mid, tone_end),
        ("Post-Tone", tone_end, tone_end + after_s),
    ]


def _freezing_latency(bouts, tone_start, tone_end, valid_fraction, min_valid_fraction):
    if pd.isna(valid_fraction) or valid_fraction < min_valid_fraction:
        return np.nan, pd.NA

    for bout_start, bout_end in bouts:
        if bout_end <= tone_start:
            continue
        if bout_start >= tone_end:
            break
        latency = max(0.0, bout_start - tone_start)
        return latency, True

    return tone_end - tone_start, False


def _summarize_epoch(
    tracking,
    heart_rate,
    bouts,
    start_s,
    end_s,
    min_valid_fraction,
    min_hr_samples,
):
    tracking_epoch = tracking[
        (tracking["time_s"] >= start_s) & (tracking["time_s"] < end_s)
    ]
    heart_epoch = heart_rate[
        (heart_rate["time_s"] >= start_s) & (heart_rate["time_s"] < end_s)
    ]

    freezing = tracking_epoch["freezing"].dropna()
    motion = tracking_epoch["motion"]
    speed = tracking_epoch["speed_px_s"]
    complete_steps = tracking_epoch[tracking_epoch["previous_time_s"] >= start_s]
    hr = heart_epoch["hr_bpm"]

    bout_durations = [
        max(0.0, min(bout_end, end_s) - max(bout_start, start_s))
        for bout_start, bout_end in bouts
        if bout_end > start_s and bout_start < end_s
    ]
    bout_durations = [duration for duration in bout_durations if duration > 0]

    expected_frames = (end_s - start_s) / tracking["time_s"].diff().median()
    hr_p10 = _percentile(hr, 0.10)
    hr_p90 = _percentile(hr, 0.90)

    motion_valid_fraction = (
        float(tracking_epoch["motion"].notna().mean())
        if not tracking_epoch.empty
        else np.nan
    )
    position_valid_fraction = (
        float(tracking_epoch[["x", "y"]].notna().all(axis=1).mean())
        if not tracking_epoch.empty
        else np.nan
    )
    freezing_valid_fraction = (
        float(tracking_epoch["freezing"].notna().mean())
        if not tracking_epoch.empty
        else np.nan
    )
    hr_valid_count = int(hr.notna().sum())
    hr_valid_fraction = float(hr.notna().mean()) if not heart_epoch.empty else np.nan

    summary = {
        "freezing_fraction": float(freezing.mean()) if not freezing.empty else np.nan,
        "longest_freezing_bout_s": max(bout_durations, default=0.0)
        if not freezing.empty
        else np.nan,
        "motion_median": _percentile(motion, 0.50),
        "motion_iqr": _percentile(motion, 0.75) - _percentile(motion, 0.25),
        "motion_p90": _percentile(motion, 0.90),
        "speed_median_px_s": _percentile(speed, 0.50),
        "distance_px": float(complete_steps["step_distance_px"].sum(min_count=1)),
        "speed_p90_px_s": _percentile(speed, 0.90),
        "hr_median_bpm": _percentile(hr, 0.50),
        "hr_p10_bpm": hr_p10,
        "hr_p90_bpm": hr_p90,
        "hr_p90_minus_p10_bpm": hr_p90 - hr_p10,
        "tracking_coverage": min(1.0, len(tracking_epoch) / expected_frames)
        if expected_frames > 0
        else np.nan,
        "motion_valid_fraction": motion_valid_fraction,
        "position_valid_fraction": position_valid_fraction,
        "freezing_valid_fraction": freezing_valid_fraction,
        "hr_valid_fraction": hr_valid_fraction,
        "tracking_sample_count": int(len(tracking_epoch)),
        "motion_valid_count": int(tracking_epoch["motion"].notna().sum()),
        "position_valid_count": int(
            tracking_epoch[["x", "y"]].notna().all(axis=1).sum()
        ),
        "freezing_valid_count": int(tracking_epoch["freezing"].notna().sum()),
        "hr_sample_count": int(len(heart_epoch)),
        "hr_valid_count": hr_valid_count,
    }

    if pd.isna(freezing_valid_fraction) or freezing_valid_fraction < min_valid_fraction:
        summary["freezing_fraction"] = np.nan
        summary["longest_freezing_bout_s"] = np.nan
    if pd.isna(motion_valid_fraction) or motion_valid_fraction < min_valid_fraction:
        for feature in ["motion_median", "motion_iqr", "motion_p90"]:
            summary[feature] = np.nan
    if pd.isna(position_valid_fraction) or position_valid_fraction < min_valid_fraction:
        for feature in ["speed_median_px_s", "distance_px", "speed_p90_px_s"]:
            summary[feature] = np.nan
    if (
        pd.isna(hr_valid_fraction)
        or hr_valid_fraction < min_valid_fraction
        or hr_valid_count < min_hr_samples
    ):
        for feature in [
            "hr_median_bpm",
            "hr_p10_bpm",
            "hr_p90_bpm",
            "hr_p90_minus_p10_bpm",
        ]:
            summary[feature] = np.nan
    return summary


def extract_epoch_features(
    sessions,
    phases=("ext_1", "ext_2"),
    before_s=10,
    after_s=10,
    min_valid_fraction=0.80,
    min_hr_samples=10,
):
    """Extract interpretable features from four epochs around each extinction tone."""
    phases = validate_phases(phases)
    selected = sessions[sessions["phase"].isin(phases)].copy()
    selected = selected.sort_values(["subject_id", "phase"])
    rows = []

    for _, session in selected.iterrows():
        tracking, heart_rate, events = load_session_files(session)
        tracking = _add_speed(tracking)
        bouts = freezing_bouts(tracking)
        tones = events[events["event"] == "tone"].sort_values("trial")

        for _, tone in tones.iterrows():
            tone_tracking = tracking[
                (tracking["time_s"] >= float(tone["start_s"]))
                & (tracking["time_s"] < float(tone["end_s"]))
            ]
            tone_freezing_coverage = (
                float(tone_tracking["freezing"].notna().mean())
                if not tone_tracking.empty
                else np.nan
            )
            latency, freezing_observed = _freezing_latency(
                bouts,
                float(tone["start_s"]),
                float(tone["end_s"]),
                tone_freezing_coverage,
                min_valid_fraction,
            )
            for epoch, start_s, end_s in _epoch_specs(tone, before_s, after_s):
                row = {
                    "subject_id": session["subject_id"],
                    "phase": session["phase"],
                    "trial": int(tone["trial"]),
                    "epoch": epoch,
                    "epoch_start_s": start_s,
                    "epoch_end_s": end_s,
                    "epoch_duration_s": end_s - start_s,
                    "latency_to_freezing_s": latency,
                    "tone_freezing_observed": freezing_observed,
                }
                row.update(
                    _summarize_epoch(
                        tracking,
                        heart_rate,
                        bouts,
                        start_s,
                        end_s,
                        min_valid_fraction,
                        min_hr_samples,
                    )
                )
                rows.append(row)

    features = pd.DataFrame(rows)
    if not features.empty:
        features["epoch"] = pd.Categorical(
            features["epoch"], categories=EPOCH_ORDER, ordered=True
        )
        features = features.sort_values(
            ["subject_id", "phase", "trial", "epoch"]
        ).reset_index(drop=True)
    return features


def build_epoch_contrasts(epoch_features, features=None):
    """Return tidy cross-epoch changes for comparable signal summaries."""
    features = CONTRAST_FEATURES if features is None else list(features)
    index = ["subject_id", "phase", "trial"]
    missing = set(features) - set(epoch_features.columns)
    if missing:
        raise ValueError(f"Unknown feature columns: {sorted(missing)}")

    long = epoch_features.melt(
        id_vars=index + ["epoch"],
        value_vars=list(features),
        var_name="feature",
        value_name="epoch_value",
    )
    wide = long.set_index(index + ["feature", "epoch"])["epoch_value"].unstack(
        "epoch"
    )

    rows = []
    for contrast, (left, right) in CONTRAST_SPECS.items():
        values = (wide[left] - wide[right]).rename("value").reset_index()
        values.insert(3, "contrast", contrast)
        rows.append(values)
    return pd.concat(rows, ignore_index=True)


def summarize_subject_epochs(epoch_features, features=None):
    """Collapse trials to one median value per subject, phase, and epoch."""
    if features is None:
        features = [feature for feature, _ in OVERVIEW_FEATURES]
    missing = set(features) - set(epoch_features.columns)
    if missing:
        raise ValueError(f"Unknown feature columns: {sorted(missing)}")

    return (
        epoch_features.groupby(
            ["subject_id", "phase", "epoch"],
            observed=True,
            as_index=False,
        )[list(features)]
        .median()
        .sort_values(["subject_id", "phase", "epoch"])
        .reset_index(drop=True)
    )


def plot_epoch_overview(epoch_features, feature_specs=None, figsize=(7, 4.5)):
    """Plot each subject-level epoch distribution in a separate figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    feature_specs = OVERVIEW_FEATURES if feature_specs is None else list(feature_specs)
    if not feature_specs:
        raise ValueError("Choose at least one (feature, label) pair.")
    if any(len(spec) != 2 for spec in feature_specs):
        raise ValueError("Each feature specification must be a (feature, label) pair.")
    selected_features = list(dict.fromkeys(feature for feature, _ in feature_specs))
    subject_epochs = summarize_subject_epochs(epoch_features, selected_features)
    phases = [
        phase
        for phase in ["ext_1", "ext_2"]
        if phase in subject_epochs["phase"].unique()
    ]
    if not phases:
        raise ValueError("No extinction phases are available for the overview.")

    x = np.arange(len(EPOCH_ORDER), dtype=float)
    offsets = np.linspace(-0.17, 0.17, len(phases)) if len(phases) > 1 else [0.0]
    rng = np.random.default_rng(7)

    figures = []
    axes = []
    for feature, label in feature_specs:
        fig, ax = plt.subplots(figsize=figsize)
        for phase, offset in zip(phases, offsets):
            color = style.PHASE_COLORS[phase]
            for epoch_i, epoch in enumerate(EPOCH_ORDER):
                values = subject_epochs.loc[
                    (subject_epochs["phase"] == phase)
                    & (subject_epochs["epoch"] == epoch),
                    feature,
                ].dropna()
                if values.empty:
                    continue

                position = x[epoch_i] + offset
                box = ax.boxplot(
                    values,
                    positions=[position],
                    widths=0.26,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.3},
                    whiskerprops={"color": color, "linewidth": 1},
                    capprops={"color": color, "linewidth": 1},
                )
                box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.72)
                jitter = rng.uniform(-0.045, 0.045, len(values))
                ax.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=13,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.7,
                    alpha=0.9,
                    zorder=3,
                )

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(style.EPOCH_LABELS)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        style.despine(ax)
        if feature == "freezing_fraction":
            ax.set_ylim(-0.03, 1.03)
        ax.legend(
            handles=[
                Patch(facecolor=style.PHASE_COLORS[phase], label=style.PHASE_LABELS[phase])
                for phase in phases
            ],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=len(phases),
        )
        ax.set_title("Overall epoch pattern - trial median per subject", pad=38)
        fig.tight_layout()
        figures.append(fig)
        axes.append(ax)
    return figures, axes


def plot_trial_epochs(
    session,
    trial=1,
    before_s=10,
    after_s=10,
    figsize=(11, 7),
):
    """Show native signals and the four feature-extraction epochs for one trial."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    tracking, heart_rate, events = load_session_files(session)
    tracking = _add_speed(tracking)
    tone = events[(events["event"] == "tone") & (events["trial"] == trial)]
    if tone.empty:
        raise ValueError(f"Tone trial {trial} was not found.")
    tone = tone.iloc[0]
    tone_start = float(tone["start_s"])
    epochs = _epoch_specs(tone, before_s, after_s)
    start_s = epochs[0][1]
    end_s = epochs[-1][2]

    tracking_window = tracking[tracking["time_s"].between(start_s, end_s)].copy()
    heart_window = heart_rate[heart_rate["time_s"].between(start_s, end_s)].copy()
    tracking_window["relative_time_s"] = tracking_window["time_s"] - tone_start
    heart_window["relative_time_s"] = heart_window["time_s"] - tone_start

    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.4, 0.55, 1.4]},
    )
    for ax in axes:
        for (epoch, epoch_start, epoch_end), color in zip(epochs, style.EPOCH_COLORS):
            ax.axvspan(
                epoch_start - tone_start,
                epoch_end - tone_start,
                color=color,
                alpha=0.35,
                linewidth=0,
            )
        ax.axvline(0, color="0.25", linestyle="--", linewidth=0.8)
        ax.axvline(float(tone["end_s"]) - tone_start, color="0.25", linestyle="--", linewidth=0.8)
        style.despine(ax)

    axes[0].plot(
        tracking_window["relative_time_s"],
        tracking_window["motion"],
        color=style.SIGNAL_COLORS["motion"],
        linewidth=0.8,
    )
    axes[0].set_ylabel("Motion\n(a.u.)")
    axes[1].plot(
        tracking_window["relative_time_s"],
        tracking_window["speed_px_s"],
        color=style.SIGNAL_COLORS["speed"],
        linewidth=0.8,
    )
    axes[1].set_ylabel("Speed\n(px/s)")
    freezing_cmap = ListedColormap([(1, 1, 1, 0), style.SIGNAL_COLORS["freezing"]]).with_extremes(bad="0.75")
    axes[2].imshow(
        tracking_window["freezing"].astype(float).to_numpy()[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=freezing_cmap,
        vmin=0,
        vmax=1,
        extent=[-before_s, float(tone["duration_s"]) + after_s, 0, 1],
        zorder=2,
    )
    axes[2].set_ylabel("Freezing")
    axes[2].set_yticks([])
    axes[3].plot(
        heart_window["relative_time_s"],
        heart_window["hr_bpm"],
        color=style.SIGNAL_COLORS["hr"],
        linewidth=0.9,
    )
    axes[3].set_ylabel("HR (bpm)")
    axes[3].set_xlabel("Seconds from tone onset")
    axes[0].set_title(
        f"{session['subject_id']} - {session['phase']} - tone {trial}"
    )
    axes[0].legend(
        handles=[
            Patch(facecolor=color, alpha=0.35, label=label)
            for color, label in zip(style.EPOCH_COLORS, style.EPOCH_LABELS)
        ],
        frameon=False,
        ncol=4,
        fontsize=8,
        loc="upper right",
    )
    axes[-1].set_xlim(-before_s, float(tone["duration_s"]) + after_s)
    fig.tight_layout()
    return fig, axes
