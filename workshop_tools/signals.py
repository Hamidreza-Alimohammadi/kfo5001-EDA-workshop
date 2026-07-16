# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd


def freezing_bouts(tracking):
    """Return freezing intervals while treating missing samples as bout breaks."""
    if tracking.empty:
        return []

    times = pd.to_numeric(tracking["time_s"], errors="coerce").to_numpy(dtype=float)
    freezing = tracking["freezing"]
    valid_time = np.isfinite(times)
    valid_freezing = freezing.notna().to_numpy()
    frozen = freezing.astype("boolean").fillna(False).to_numpy(dtype=bool) & valid_time

    time_steps = np.diff(times[valid_time])
    time_steps = time_steps[np.isfinite(time_steps) & (time_steps > 0)]
    frame_dt = float(np.median(time_steps)) if len(time_steps) else 0.0
    maximum_contiguous_step = 1.5 * frame_dt if frame_dt > 0 else np.inf

    bouts = []
    start_index = None
    for index in range(len(times)):
        contiguous = (
            index > 0
            and valid_time[index - 1]
            and valid_freezing[index - 1]
            and times[index] - times[index - 1] <= maximum_contiguous_step
        )
        continues_bout = frozen[index] and valid_freezing[index] and contiguous

        if frozen[index] and valid_freezing[index] and start_index is None:
            start_index = index
        elif start_index is not None and not continues_bout:
            bouts.append((float(times[start_index]), float(times[index - 1] + frame_dt)))
            start_index = index if frozen[index] and valid_freezing[index] else None

    if start_index is not None:
        bouts.append((float(times[start_index]), float(times[-1] + frame_dt)))
    return bouts
