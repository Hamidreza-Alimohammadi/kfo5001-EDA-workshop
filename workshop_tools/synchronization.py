# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd

from workshop_tools import style


def sampling_intervals(timestamps):
    """Calculate adjacent intervals without bridging missing timestamps."""
    timestamps = pd.to_numeric(timestamps, errors="coerce")
    return timestamps.diff().dropna()


def sampling_summary(session, tracking, heart_rate, events):
    def interval_text(values):
        values = values.dropna()
        if values.empty:
            return "not sampled", "not sampled"
        median = values.median()
        mad = (values - median).abs().median()
        mean = values.mean()
        std = values.std()
        return (
            f"{median:.4f} +/- {mad:.4f}",
            f"{mean:.4f} +/- {std:.4f}",
        )

    tracking_dt = sampling_intervals(tracking["time_s"])
    heart_dt = sampling_intervals(heart_rate["time_s"])
    tracking_median_mad, tracking_mean_std = interval_text(tracking_dt)
    heart_median_mad, heart_mean_std = interval_text(heart_dt)
    rows = [
        {
            "modality": "Tracking",
            "what_is_sampled": "video frames",
            "rows": len(tracking),
            "time_column": "time_s",
            "median_dt_s_mad": tracking_median_mad,
            "mean_dt_s_std": tracking_mean_std,
        },
        {
            "modality": "Heart rate",
            "what_is_sampled": "heartbeats",
            "rows": len(heart_rate),
            "time_column": "time_s",
            "median_dt_s_mad": heart_median_mad,
            "mean_dt_s_std": heart_mean_std,
        },
        {
            "modality": "Events",
            "what_is_sampled": "tone/shock intervals",
            "rows": len(events),
            "time_column": "start_s, end_s",
            "median_dt_s_mad": "not sampled",
            "mean_dt_s_std": "not sampled",
        },
    ]
    summary = pd.DataFrame(rows)
    subject = session["subject_id"] if "subject_id" in session else session["mouse_id"]
    summary.insert(0, "subject_id", subject)
    summary.insert(1, "phase", session["phase"])
    return summary


def plot_timestamp_geometry(
    tracking,
    heart_rate,
    events,
    trial=1,
    before_s=2,
    after_s=4,
    tracking_stride=5,
):
    import matplotlib.pyplot as plt

    tone = events[(events["event"] == "tone") & (events["trial"] == trial)].iloc[0]
    start_s = tone["start_s"] - before_s
    end_s = tone["end_s"] + after_s

    tracking_times = tracking.loc[
        tracking["time_s"].between(start_s, end_s), "time_s"
    ].iloc[::tracking_stride]
    heartbeat_times = heart_rate.loc[
        heart_rate["time_s"].between(start_s, end_s), "time_s"
    ]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.eventplot(
        [tracking_times, heartbeat_times],
        lineoffsets=[2, 1],
        linelengths=0.75,
        colors=[style.SIGNAL_COLORS["motion"], style.SIGNAL_COLORS["hr"]],
    )
    ax.axvspan(tone["start_s"], tone["end_s"], color="gold", alpha=0.35)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["tracking frames", "heartbeats", "tone interval"])
    ax.set_xlabel("Session time (s)")
    ax.set_title(f"Different timestamp patterns around tone {trial}")
    ax.set_xlim(start_s, end_s)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    return fig, ax


def plot_sampling_intervals(tracking, heart_rate, bins=30, yscale="linear"):
    import matplotlib.pyplot as plt

    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")

    tracking_dt = sampling_intervals(tracking["time_s"])
    heart_dt = sampling_intervals(heart_rate["time_s"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=False)
    for ax, values, color, title in [
        (axes[0], tracking_dt, style.SIGNAL_COLORS["motion"], "Tracking frame intervals"),
        (axes[1], heart_dt, style.SIGNAL_COLORS["hr"], "Heartbeat intervals"),
    ]:
        ax.hist(values, bins=bins, color=color, alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yscale(yscale)
        ax.set_title(title)
        ax.set_xlabel("Seconds between samples")

    axes[0].set_ylabel("Count")

    fig.tight_layout()
    return fig, axes


def event_aligned_bins(
    tracking,
    heart_rate,
    events,
    event="tone",
    window=(-30, 60),
    bin_size_s=1,
):
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive.")

    rows = []
    selected_events = events[events["event"] == event].sort_values("trial")
    left, right = window
    bin_starts = np.arange(left, right, bin_size_s)

    for _, event_row in selected_events.iterrows():
        tone_start = event_row["start_s"]
        for bin_start in bin_starts:
            bin_end = bin_start + bin_size_s
            absolute_start = tone_start + bin_start
            absolute_end = tone_start + bin_end

            tracking_bin = tracking[
                (tracking["time_s"] >= absolute_start)
                & (tracking["time_s"] < absolute_end)
            ]
            heart_bin = heart_rate[
                (heart_rate["time_s"] >= absolute_start)
                & (heart_rate["time_s"] < absolute_end)
            ]

            rows.append(
                {
                    "trial": int(event_row["trial"]),
                    "bin_start_s": round(float(bin_start), 6),
                    "bin_end_s": round(float(bin_end), 6),
                    "freezing_fraction": tracking_bin["freezing"].mean(),
                    "mean_motion": tracking_bin["motion"].mean(),
                    "mean_hr_bpm": heart_bin["hr_bpm"].mean(),
                    "tracking_samples": len(tracking_bin),
                    "heartbeats": len(heart_bin),
                }
            )

    return pd.DataFrame(rows)
