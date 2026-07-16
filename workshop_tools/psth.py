# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd

from workshop_tools import style
from workshop_tools.session_data import load_session_files, select_sessions
from workshop_tools.synchronization import event_aligned_bins


FEATURE_SPECS = [
    ("mean_motion", "Motion (a.u.)", "motion"),
    ("freezing_fraction", "Freezing", "freezing"),
    ("mean_hr_bpm", "HR (bpm)", "hr"),
]


def _resolve_color(color):
    return style.SIGNAL_COLORS.get(color, color)


def build_phase_psth(
    sessions,
    selection=("all", "cond"),
    phase=None,
    alignment_event="tone",
    window=(-30, 30),
    bin_size_s=1,
):
    rows = []
    selected = select_sessions(sessions, selection=selection, phase=phase)
    selected = selected[
        selected["has_tracking"] & selected["has_events"]
    ].reset_index(drop=True)
    if selected.empty:
        raise ValueError("No selected sessions have tracking and events.")

    for _, session in selected.iterrows():
        tracking, heart_rate, events = load_session_files(session)
        binned = event_aligned_bins(
            tracking,
            heart_rate,
            events,
            event=alignment_event,
            window=window,
            bin_size_s=bin_size_s,
        )
        binned.insert(0, "subject_id", session["subject_id"])
        binned.insert(1, "phase", session["phase"])
        rows.append(binned)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_event_windows(sessions, selection=("all", "cond"), alignment_event="tone"):
    selected = select_sessions(sessions, selection=selection)
    selected = selected[selected["has_events"]].reset_index(drop=True)
    if selected.empty:
        raise ValueError("No selected sessions have events.")
    if selected["phase"].nunique() != 1:
        raise ValueError("Event windows require sessions from one phase.")

    session = selected.iloc[0]
    _, _, events = load_session_files(session)
    anchor_events = events[events["event"] == alignment_event][
        ["trial", "start_s", "end_s"]
    ]
    rows = []

    for _, anchor in anchor_events.iterrows():
        anchor_start = anchor["start_s"]

        trial_events = events[events["trial"] == anchor["trial"]]
        for _, event in trial_events.iterrows():
            rows.append(
                {
                    "trial": int(anchor["trial"]),
                    "event": event["event"],
                    "start_s": float(event["start_s"] - anchor_start),
                    "end_s": float(event["end_s"] - anchor_start),
                }
            )

    return pd.DataFrame(rows)


def _trial_groups(trials):
    trials = sorted(trials)
    if len(trials) == 16:
        groups = []
        for start in range(0, 16, 4):
            block = trials[start : start + 4]
            groups.append([(trial, f"Trial {trial}") for trial in block])
            groups[-1].append((block, f"Mean {block[0]}-{block[-1]}"))
        return groups
    return [[(trial, f"Trial {trial}") for trial in trials]]


def _subject_values(psth, trials, feature):
    trial_list = trials if isinstance(trials, list) else [trials]
    values = (
        psth[psth["trial"].isin(trial_list)]
        .groupby(["subject_id", "bin_start_s"], as_index=False)[feature]
        .mean()
    )
    return values


def _summarize_time(values, feature, center, band):
    grouped = values.groupby("bin_start_s", as_index=False)
    center_values = grouped[feature].mean() if center == "mean" else grouped[feature].median()
    center_values = center_values.rename(columns={feature: "center"})

    if band is None:
        center_values["lower"] = np.nan
        center_values["upper"] = np.nan
        return center_values

    if band == "std":
        spread = grouped[feature].std().rename(columns={feature: "spread"})
    elif band == "sem":
        spread = grouped[feature].sem().rename(columns={feature: "spread"})
    else:
        spread = (
            values.groupby("bin_start_s")[feature]
            .agg(lambda x: (x.dropna() - x.dropna().median()).abs().median())
            .rename("spread")
            .reset_index()
        )

    summary = center_values.merge(spread, on="bin_start_s", how="left")
    summary["lower"] = summary["center"] - summary["spread"]
    summary["upper"] = summary["center"] + summary["spread"]
    return summary


def plot_phase_psth(
    psth,
    event_windows,
    center="median",
    band="mad",
    show_subject_traces=True,
    subject_trace_color=None,
    subject_trace_alpha=0.28,
    subject_trace_linewidth=0.75,
    average_column_facecolor="0.96",
    y_limit_n_std=None,
    freezing_label_fontsize=5,
    title=None,
    alignment_label="tone onset",
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if title is None:
        phases = sorted(psth["phase"].dropna().unique()) if "phase" in psth else []
        phase_text = phases[0] if len(phases) == 1 else "Selected sessions"
        title = f"{phase_text} PSTH"
    band_text = "no band" if band is None else f"+/- {band.upper()}"
    trial_groups = _trial_groups(psth["trial"].unique())
    row_blocks = len(trial_groups)
    col_n = max(len(group) for group in trial_groups)
    feature_specs = [(feature, label, _resolve_color(color)) for feature, label, color in FEATURE_SPECS]
    feature_row_n = len(feature_specs)
    spacer_row_n = max(row_blocks - 1, 0)
    row_n = row_blocks * feature_row_n + spacer_row_n
    height_ratios = []
    for block_i in range(row_blocks):
        height_ratios.extend([1.2, 0.9, 1.2])
        if block_i < row_blocks - 1:
            height_ratios.append(0.42)

    fig, axes = plt.subplots(
        row_n,
        col_n,
        figsize=(3.05 * col_n, 1.85 * row_blocks * feature_row_n),
        sharex=True,
        squeeze=False,
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.015,
            "wspace": 0.22,
        },
    )
    freezing_cmap = LinearSegmentedColormap.from_list(
        "psth_freezing_raster",
        ["white", dict((feature, color) for feature, _, color in feature_specs)["freezing_fraction"]],
    )
    freezing_cmap.set_bad("0.9")
    feature_limits = {
        feature_name: style.std_limits(
            psth[feature_name],
            y_limit_n_std,
            lower_bound=0 if feature_name == "mean_motion" else None,
        )
        for feature_name, _, _ in feature_specs
        if feature_name != "freezing_fraction"
    }

    for block_i, group in enumerate(trial_groups):
        block_row_start = block_i * (feature_row_n + 1)
        if block_i > 0:
            spacer_row = block_row_start - 1
            for col_i in range(col_n):
                axes[spacer_row, col_i].set_visible(False)

        for col_i in range(col_n):
            if col_i >= len(group):
                for feature_i in range(feature_row_n):
                    axes[block_row_start + feature_i, col_i].set_visible(False)
                continue

            trial_selector, panel_title = group[col_i]
            trial_list = trial_selector if isinstance(trial_selector, list) else [trial_selector]
            is_average_column = isinstance(trial_selector, list)
            trial_events = event_windows[event_windows["trial"] == trial_list[0]]

            for feature_i, (feature_name, label, color) in enumerate(feature_specs):
                ax = axes[block_row_start + feature_i, col_i]
                if is_average_column:
                    ax.set_facecolor(average_column_facecolor)
                for _, event in trial_events.iterrows():
                    shade = "0.82" if event["event"] == "tone" else "0.45"
                    alpha = 0.35 if event["event"] == "tone" else 0.28
                    ax.axvspan(
                        event["start_s"],
                        event["end_s"],
                        color=shade,
                        alpha=alpha,
                        linewidth=0,
                        zorder=0,
                    )

                values = _subject_values(psth, trial_list, feature_name)

                if feature_name == "freezing_fraction":
                    freezing = (
                        values.pivot(
                            index="subject_id",
                            columns="bin_start_s",
                            values=feature_name,
                        )
                        .sort_index()
                    )
                    if not freezing.empty:
                        ax.imshow(
                            freezing.to_numpy(dtype=float),
                            aspect="auto",
                            interpolation="nearest",
                            cmap=freezing_cmap,
                            vmin=0,
                            vmax=1,
                            extent=[
                                float(freezing.columns.min()),
                                float(freezing.columns.max()),
                                len(freezing),
                                0,
                            ],
                            zorder=1,
                            alpha=0.98,
                        )
                        ax.set_yticks(np.arange(len(freezing.index)) + 0.5)
                        ax.set_yticklabels(
                            freezing.index.tolist(),
                            fontsize=freezing_label_fontsize,
                        )
                        ax.tick_params(axis="y", length=0, pad=1)
                    else:
                        ax.set_yticks([])
                else:
                    summary = _summarize_time(values, feature_name, center, band)
                    y_limits = feature_limits[feature_name]
                    trace_color = subject_trace_color or color
                    if show_subject_traces:
                        subject_curves = (
                            values.pivot(
                                index="bin_start_s",
                                columns="subject_id",
                                values=feature_name,
                            )
                            .sort_index()
                        )
                        for subject_id in subject_curves.columns:
                            ax.plot(
                                subject_curves.index.to_numpy(dtype=float),
                                subject_curves[subject_id].to_numpy(dtype=float),
                                color=trace_color,
                                alpha=subject_trace_alpha,
                                linewidth=subject_trace_linewidth,
                                zorder=1,
                            )

                    x = summary["bin_start_s"].to_numpy(dtype=float)
                    y = summary["center"].to_numpy(dtype=float)
                    if band is not None:
                        lower = summary["lower"].to_numpy(dtype=float)
                        upper = summary["upper"].to_numpy(dtype=float)
                        ax.fill_between(
                            x,
                            lower,
                            upper,
                            color=color,
                            alpha=0.22 if is_average_column else 0.18,
                            linewidth=0,
                            zorder=2,
                        )
                    ax.plot(
                        x,
                        y,
                        color=color,
                        linewidth=2.4 if is_average_column else 1.7,
                        zorder=3,
                    )
                    if y_limits is not None:
                        ax.set_ylim(*y_limits)

                ax.axvline(
                    0,
                    color="0.2",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=4,
                )

                if is_average_column:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color("0.35")
                        spine.set_linewidth(1.1)
                else:
                    style.despine(ax)

                if feature_i == 0:
                    ax.set_title(
                        panel_title,
                        fontweight="bold" if is_average_column else "normal",
                    )
                if col_i == 0:
                    ax.set_ylabel(label)
                if feature_i < len(feature_specs) - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(f"Seconds from {alignment_label}")

    fig.suptitle(f"{title}: {center} {band_text}")
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.055,
        top=0.90,
        hspace=0.015,
        wspace=0.22,
    )
    return fig, axes
