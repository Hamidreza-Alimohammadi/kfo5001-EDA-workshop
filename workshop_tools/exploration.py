# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import pandas as pd

from workshop_tools import style
from workshop_tools.session_data import load_session_files, select_sessions
from workshop_tools.signals import freezing_bouts


def _shade_events(axes, events, tone_color, shock_color):
    for ax in axes:
        for _, event in events.iterrows():
            if event["event"] == "tone":
                color, alpha = tone_color, 0.35
            elif event["event"] == "shock":
                color, alpha = shock_color, 0.30
            else:
                continue
            ax.axvspan(
                event["start_s"],
                event["end_s"],
                color=color,
                alpha=alpha,
                linewidth=0,
                zorder=0,
            )


def plot_session_trace(
    session,
    tone_color="0.82",
    shock_color="0.45",
    figsize=(12, 5.5),
    y_limit_n_std=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    tracking, heart_rate, events = load_session_files(session)
    title = f"{session['subject_id']} - {session['phase']}"
    freezing_cmap = LinearSegmentedColormap.from_list(
        "session_freezing_raster",
        ["white", style.SIGNAL_COLORS["freezing"]],
    )
    freezing_cmap.set_bad("0.9")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 0.7, 2.4]},
    )

    _shade_events(axes, events, tone_color, shock_color)

    axes[0].plot(
        tracking["time_s"],
        tracking["motion"],
        color=style.SIGNAL_COLORS["motion"],
        linewidth=0.8,
        zorder=2,
    )
    axes[0].set_ylabel("Motion (a.u.)")
    axes[0].set_title(title)
    motion_limits = style.std_limits(tracking["motion"], y_limit_n_std, lower_bound=0)
    if motion_limits is not None:
        axes[0].set_ylim(*motion_limits)

    axes[1].imshow(
        tracking["freezing"].astype(float).to_numpy()[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=freezing_cmap,
        vmin=0,
        vmax=1,
        extent=[tracking["time_s"].min(), tracking["time_s"].max(), 0, 1],
        alpha=0.95,
        zorder=2,
    )
    axes[1].set_ylabel("Freezing")
    axes[1].set_yticks([])

    if not heart_rate.empty:
        axes[2].plot(
            heart_rate["time_s"],
            heart_rate["hr_bpm"],
            color=style.SIGNAL_COLORS["hr"],
            linewidth=0.9,
            zorder=2,
        )
        hr_limits = style.std_limits(heart_rate["hr_bpm"], y_limit_n_std)
        if hr_limits is not None:
            axes[2].set_ylim(*hr_limits)
    axes[2].set_ylabel("HR (bpm)")
    axes[2].set_xlabel("Session time (s)")

    x_max = tracking["time_s"].max()
    if not heart_rate.empty:
        x_max = max(x_max, heart_rate["time_s"].max())
    for ax in axes:
        style.despine(ax)
        ax.set_xlim(0, x_max)

    fig.tight_layout()
    return fig, axes

def plot_session_overlay(
    sessions,
    tone_color="0.82",
    shock_color="0.45",
    figsize=(13, 7),
    y_limit_n_std=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if sessions["phase"].nunique() != 1:
        raise ValueError("Overlay plots require sessions from one phase.")

    session_files = []
    for _, session in sessions.iterrows():
        tracking, heart_rate, events = load_session_files(session)
        session_files.append((session, tracking, heart_rate, events))

    events = session_files[0][3]
    phase = sessions["phase"].iloc[0]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.2, 2.4]},
    )

    _shade_events(axes, events, tone_color, shock_color)

    x_max = 0
    motion_values = []
    hr_values = []
    for i, (session, tracking, heart_rate, _) in enumerate(session_files):
        x_max = max(x_max, tracking["time_s"].max())
        motion_values.append(tracking["motion"])
        if not heart_rate.empty:
            x_max = max(x_max, heart_rate["time_s"].max())
            hr_values.append(heart_rate["hr_bpm"])

        axes[0].plot(
            tracking["time_s"],
            tracking["motion"],
            color=style.SIGNAL_COLORS["motion"],
            linewidth=0.55,
            alpha=0.18,
            zorder=1,
        )
        if not heart_rate.empty:
            axes[2].plot(
                heart_rate["time_s"],
                heart_rate["hr_bpm"],
                color=style.SIGNAL_COLORS["hr"],
                linewidth=0.8,
                alpha=0.18,
                zorder=1,
            )
        bouts = [
            (start, end - start)
            for start, end in freezing_bouts(tracking)
        ]
        if bouts:
            axes[1].broken_barh(
                bouts,
                (i - 0.38, 0.76),
                facecolors=style.SIGNAL_COLORS["freezing"],
                edgecolors="none",
                alpha=0.95,
                zorder=2,
            )

    axes[0].set_title(f"Whole-session overlay - {phase}")
    axes[0].set_ylabel("Motion (a.u.)")
    motion_limits = style.std_limits(pd.concat(motion_values), y_limit_n_std, lower_bound=0)
    if motion_limits is not None:
        axes[0].set_ylim(*motion_limits)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=style.SIGNAL_COLORS["motion"], linewidth=1.5, label="Motion (a.u.)"),
            Line2D([0], [0], color=style.SIGNAL_COLORS["freezing"], linewidth=1.5, label="Freezing"),
            Line2D([0], [0], color=style.SIGNAL_COLORS["hr"], linewidth=1.5, label="HR"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=8,
        frameon=False,
    )
    axes[1].set_ylabel("Freezing")
    axes[1].set_yticks(range(len(session_files)))
    axes[1].set_yticklabels([session["subject_id"] for session, *_ in session_files])
    axes[1].set_ylim(len(session_files) - 0.5, -0.5)
    axes[2].set_ylabel("HR (bpm)")
    axes[2].set_xlabel("Session time (s)")
    if hr_values:
        hr_limits = style.std_limits(pd.concat(hr_values), y_limit_n_std)
        if hr_limits is not None:
            axes[2].set_ylim(*hr_limits)

    for ax in axes:
        style.despine(ax)
        ax.set_xlim(0, x_max)

    fig.tight_layout()
    return fig, axes


def plot_session_traces(
    sessions,
    selection="all",
    phase=None,
    max_sessions=None,
    **plot_kwargs,
):
    selected = select_sessions(sessions, selection=selection, phase=phase)
    selected = selected[
        selected["has_tracking"] & selected["has_events"]
    ].reset_index(drop=True)
    if selected.empty:
        raise ValueError("No selected sessions have tracking and events.")
    if max_sessions is not None:
        selected = selected.head(max_sessions)

    if len(selected) == 1:
        return [plot_session_trace(selected.iloc[0], **plot_kwargs)]
    return [plot_session_overlay(selected, **plot_kwargs)]
