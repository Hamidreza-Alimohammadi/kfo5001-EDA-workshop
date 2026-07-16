# Synthetic reduced workshop data

All values in this directory are independently generated synthetic surrogates, not experimental observations. See [`DATA_NOTICE.md`](../../DATA_NOTICE.md).

Each session has three CSV files:

- `Tracking.csv`: synthetic video-frame timeline at 30 Hz, with `frame`, `time_s`, `x`, `y`, `motion`, and `freezing`.
- `HeartRate.csv`: synthetic irregular heartbeat timeline, with `beat`, `time_s`, `ibi_s`, `hr_bpm`, `in_artifact_window`, and `in_removed_window`.
- `Events.csv`: generic tone/shock intervals, with `event`, `trial`, `start_s`, `end_s`, and `duration_s`.

Subjects use anonymous teaching identifiers `subject_01` through `subject_14`. Phases use exactly `cond`, `ext_1`, and `ext_2`. These labels preserve notebook functionality and do not identify experimental animals.

Run `python scripts/generate_surrogate_data.py` from the repository root to reproduce every file in this directory using the published fixed seed.
