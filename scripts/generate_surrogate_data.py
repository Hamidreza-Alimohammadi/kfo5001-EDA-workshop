# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

#!/usr/bin/env python3
"""Generate the public workshop's independent synthetic surrogate dataset.

The generator is self-contained and never reads experimental recordings. A fixed
seed makes the teaching release reproducible; changing the seed creates another
fully synthetic realization with the same schemas and session organization.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 5001
SUBJECTS = [f"subject_{number:02d}" for number in range(1, 15)]
PHASES = ("cond", "ext_1", "ext_2")
PHASE_TOKEN = {"cond": 1, "ext_1": 2, "ext_2": 3}
DURATION_S = {"cond": 1080.0, "ext_1": 1920.0, "ext_2": 1920.0}
MISSING_HR_SUBJECTS = {"subject_04", "subject_08", "subject_12"}
MISSING_FREEZING = {("subject_13", "cond"), ("subject_13", "ext_1")}


def event_table(phase):
    """Create generic tone and shock schedules for one synthetic session."""
    if phase == "cond":
        tone_starts = np.arange(180.0, 901.0, 180.0)
        rows = []
        for trial, start in enumerate(tone_starts, start=1):
            rows.append(("tone", trial, start, start + 10.0, 10.0))
            rows.append(("shock", trial, start + 9.0, start + 10.0, 1.0))
    else:
        tone_starts = 90.0 + np.arange(16) * 110.0
        rows = [("tone", trial, start, start + 10.0, 10.0) for trial, start in enumerate(tone_starts, start=1)]
    return pd.DataFrame(rows, columns=["event", "trial", "start_s", "end_s", "duration_s"]).sort_values("start_s").reset_index(drop=True)


def target_freezing(times, events, phase, subject_index, rng):
    """Define a continuous, heterogeneous synthetic freezing propensity."""
    subject_level = 0.07 + 0.20 * (subject_index / (len(SUBJECTS) - 1))
    subject_level += rng.normal(0.0, 0.018)
    target = np.full(times.size, np.clip(subject_level, 0.03, 0.42))
    if phase == "cond":
        for tone in events.loc[events["event"] == "tone"].itertuples():
            learned = 0.06 + 0.035 * tone.trial + 0.07 * subject_index / 13
            target[(times >= tone.start_s - 10) & (times < tone.start_s)] += 0.35 * learned
            target[(times >= tone.start_s) & (times < tone.end_s)] += learned
            target[(times >= tone.end_s) & (times < tone.end_s + 10)] += 0.65 * learned
    else:
        day_offset = 0 if phase == "ext_1" else 16
        learning_rate = 0.026 + 0.035 * (13 - subject_index) / 13
        fear_scale = 0.55 + 0.22 * subject_index / 13 + rng.normal(0.0, 0.025)
        for tone in events.itertuples():
            global_trial = day_offset + tone.trial - 1
            response = fear_scale * np.exp(-learning_rate * global_trial)
            response += 0.035 * np.sin((tone.trial + subject_index) / 2.7)
            target[(times >= tone.start_s - 10) & (times < tone.start_s)] += 0.30 * response
            target[(times >= tone.start_s) & (times < tone.start_s + 5)] += response
            target[(times >= tone.start_s + 5) & (times < tone.end_s)] += 0.78 * response
            target[(times >= tone.end_s) & (times < tone.end_s + 10)] += 0.48 * response
    return np.clip(target, 0.01, 0.92)


def freezing_states(target, rng, frame_rate=30.0):
    """Draw persistent binary bouts whose occupancy follows the target curve."""
    states = np.zeros(target.size, dtype=bool)
    states[0] = rng.random() < target[0]
    off_probability = 1.0 / (2.4 * frame_rate)
    random_values = rng.random(target.size)
    for index in range(1, target.size):
        if states[index - 1]:
            states[index] = random_values[index] >= off_probability
        else:
            on_probability = min(0.25, off_probability * target[index] / (1.0 - target[index]))
            states[index] = random_values[index] < on_probability
    return states


def tracking_table(duration_s, events, phase, subject_index, rng, freezing_available):
    """Generate bounded frame-level position, motion, and freezing signals."""
    frame_rate = 30.0
    times = np.arange(0.0, duration_s, 1.0 / frame_rate)
    freezing = freezing_states(target_freezing(times, events, phase, subject_index, rng), rng)
    movement_scale = np.where(freezing, 0.055, 0.78 + 0.13 * np.sin(times / 17.0))
    dx = rng.normal(0.0, movement_scale)
    dy = rng.normal(0.0, movement_scale)
    x = 320.0 + np.cumsum(dx)
    y = 240.0 + np.cumsum(dy)
    x = np.clip(x, 40.0, 600.0)
    y = np.clip(y, 35.0, 445.0)
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    motion = step * (1.0 + rng.lognormal(-0.15, 0.22, times.size))

    table = pd.DataFrame({"frame": np.arange(times.size), "time_s": times, "x": x, "y": y, "motion": motion})
    table["freezing"] = pd.Series(freezing, dtype="boolean")
    # Independently generated dropout blocks provide realistic missing-data examples.
    for _ in range(2):
        start = rng.uniform(60.0, duration_s - 30.0)
        mask = table["time_s"].between(start, start + rng.uniform(0.4, 1.6))
        table.loc[mask, ["x", "y", "motion"]] = np.nan
    if freezing_available:
        start = rng.uniform(80.0, duration_s - 30.0)
        table.loc[table["time_s"].between(start, start + rng.uniform(0.3, 1.0)), "freezing"] = pd.NA
    else:
        table["freezing"] = pd.NA
    return table


def heart_table(duration_s, events, phase, subject_index, rng, available):
    """Generate an irregular heartbeat timeline with synthetic artifact flags."""
    columns = ["beat", "time_s", "ibi_s", "hr_bpm", "in_artifact_window", "in_removed_window"]
    if not available:
        return pd.DataFrame(columns=columns)

    baseline = 565.0 + 7.5 * subject_index + rng.normal(0.0, 8.0)
    times = []
    ibis = []
    current = 0.0
    while current < duration_s:
        event_effect = 0.0
        for tone in events.loc[events["event"] == "tone"].itertuples():
            relative = current - tone.start_s
            if -5.0 <= relative < 15.0:
                event_effect += 30.0 * np.exp(-((relative - 3.0) / 6.0) ** 2)
        if phase == "cond":
            event_effect += 7.0
        instantaneous_hr = baseline + event_effect + 10.0 * np.sin(current / 31.0) + rng.normal(0.0, 10.0)
        ibi = np.clip(60.0 / instantaneous_hr + rng.normal(0.0, 0.0022), 0.065, 0.16)
        current += ibi
        if current < duration_s:
            times.append(current)
            ibis.append(ibi)

    times = np.asarray(times)
    ibis = np.asarray(ibis)
    hr = pd.Series(60.0 / ibis).rolling(7, center=True, min_periods=4).median().to_numpy()
    artifact = np.zeros(times.size, dtype=bool)
    removed = np.zeros(times.size, dtype=bool)
    for flag in (artifact, removed):
        start = rng.uniform(50.0, duration_s - 20.0)
        flag[(times >= start) & (times <= start + rng.uniform(1.0, 4.0))] = True
    hr[artifact | removed] = np.nan
    ibis[0] = np.nan
    return pd.DataFrame({"beat": np.arange(times.size), "time_s": times, "ibi_s": ibis, "hr_bpm": hr, "in_artifact_window": artifact, "in_removed_window": removed})


def write_table(table, path, float_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=float_format)


def generate(output_dir, seed=SEED):
    """Write all public session files, metadata, and prepared features."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data" / "reduced"
    rng = np.random.default_rng(seed)
    session_rows = []

    for subject_index, subject_id in enumerate(SUBJECTS):
        for phase in PHASES:
            events = event_table(phase)
            duration_s = DURATION_S[phase]
            has_freezing = (subject_id, phase) not in MISSING_FREEZING
            has_heartbeats = subject_id not in MISSING_HR_SUBJECTS
            tracking = tracking_table(duration_s, events, phase, subject_index, rng, has_freezing)
            heart_rate = heart_table(duration_s, events, phase, subject_index, rng, has_heartbeats)
            session_dir = data_dir / subject_id / phase
            write_table(tracking, session_dir / "Tracking.csv", "%.6f")
            write_table(heart_rate, session_dir / "HeartRate.csv", "%.6f")
            write_table(events, session_dir / "Events.csv", "%.6f")
            session_rows.append({
                "subject_id": subject_id,
                "phase": phase,
                "session_date_token": PHASE_TOKEN[phase],
                "reduced_session": str(Path("data/reduced") / subject_id / phase),
                "has_tracking": True,
                "has_freezing": has_freezing,
                "frame_n": len(tracking),
                "duration_s": float(tracking["time_s"].iloc[-1]),
                "tracking_median_hz": 30.0,
                "tracking_error": np.nan,
                "has_heartbeats": has_heartbeats,
                "heartbeat_n": len(heart_rate),
                "heartbeat_error": np.nan,
                "has_events": True,
                "event_range_n": len(events),
                "event_error": np.nan,
            })

    sessions = pd.DataFrame(session_rows)
    write_table(sessions, data_dir / "sessions.csv", "%.6f")

    import sys
    sys.path.insert(0, str(output_dir))
    from workshop_tools.features import extract_epoch_features

    previous = Path.cwd()
    try:
        import os
        os.chdir(output_dir)
        features = extract_epoch_features(sessions)
        features.to_csv(data_dir / "extinction_epoch_features.csv.gz", index=False, compression="gzip", float_format="%.6f")
    finally:
        os.chdir(previous)
    return sessions, features


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    sessions, features = generate(args.output, args.seed)
    print(f"Generated {len(sessions)} synthetic sessions and {len(features)} epoch rows with seed {args.seed}.")


if __name__ == "__main__":
    main()
