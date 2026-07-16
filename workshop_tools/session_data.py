# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODALITY_COLUMNS = {
    "tracking": "has_tracking",
    "freezing": "has_freezing",
    "hr": "has_heartbeats",
    "events": "has_events",
}
PHASE_ORDER = ["cond", "ext_1", "ext_2"]


def load_sessions(
    data_dir="data/reduced",
    has_tracking=None,
    has_heartbeats=None,
    has_events=None,
    has_freezing=None,
):
    sessions = pd.read_csv(Path(data_dir) / "sessions.csv")
    validate_phases(sessions["phase"].dropna().unique())
    filters = {
        "has_tracking": has_tracking,
        "has_heartbeats": has_heartbeats,
        "has_events": has_events,
        "has_freezing": has_freezing,
    }
    for column, value in filters.items():
        if value is not None:
            sessions = sessions[sessions[column] == bool(value)]
    return sessions.reset_index(drop=True)


def choose_session(sessions, session_key=None, phase="ext_1", modalities=()):
    """Return one validated session, optionally chosen by subject and phase."""
    selection = ("all", phase) if session_key is None else session_key
    requested_modalities = (modalities,) if isinstance(modalities, str) else tuple(modalities)
    selected = select_sessions(sessions, selection=selection, modalities=requested_modalities)
    if selected.empty:
        key_text = f" for {session_key!r}" if session_key is not None else ""
        raise ValueError(f"No session{key_text} has modalities {requested_modalities!r}.")
    return selected.iloc[0]


def load_session_files(session):
    session_dir = Path(session["reduced_session"])
    tracking = pd.read_csv(session_dir / "Tracking.csv")
    heart_rate = pd.read_csv(session_dir / "HeartRate.csv")
    events = pd.read_csv(session_dir / "Events.csv")
    return tracking, heart_rate, events


def load_session(session):
    """Load one session into a named bundle of metadata and modality tables."""
    tracking, heart_rate, events = load_session_files(session)
    return SimpleNamespace(info=session.copy(), tracking=tracking, heart_rate=heart_rate, events=events)


def validate_phases(phases):
    """Require the canonical experimental phase codes."""
    phases = [phases] if isinstance(phases, str) else list(phases)
    unknown = sorted(set(phases) - set(PHASE_ORDER))
    if unknown:
        raise ValueError(f"Unknown phases: {unknown}. Choose from {PHASE_ORDER}.")
    return tuple(phases)


def _validate_phase(phase):
    if phase is None:
        return None
    validate_phases([phase])
    return phase


def _modality_columns(modalities):
    modalities = (modalities,) if isinstance(modalities, str) else tuple(modalities)
    unknown = set(modalities) - set(MODALITY_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}. Choose from {sorted(MODALITY_COLUMNS)}.")
    return [MODALITY_COLUMNS[modality] for modality in modalities]


def select_sessions(sessions, selection="all", phase=None, modalities=()):
    """Select sessions by subject and phase identifiers."""
    phase = _validate_phase(phase)
    if isinstance(selection, pd.Series):
        selected = selection.to_frame().T
    elif isinstance(selection, pd.DataFrame):
        selected = selection.copy()
    elif selection == "all":
        if phase is None:
            raise ValueError(
                "Use ('all', phase) or pass phase=...; 'all' alone is ambiguous."
            )
        selected = sessions[sessions["phase"] == phase].copy()
    else:
        items = selection if isinstance(selection, list) else [selection]
        selected_rows = []
        for item in items:
            if isinstance(item, dict):
                subject_id = item.get("subject_id")
                item_phase = _validate_phase(item.get("phase", phase))
            elif isinstance(item, tuple) and len(item) == 2:
                subject_id, item_phase = item
                item_phase = _validate_phase(item_phase)
            else:
                subject_id = item
                item_phase = phase

            if subject_id == "all":
                if item_phase is None:
                    raise ValueError("Use ('all', phase); 'all' needs a phase.")
                matches = sessions[sessions["phase"] == item_phase]
                selected_rows.append(matches)
                continue

            matches = sessions[sessions["subject_id"] == subject_id]
            if item_phase is not None:
                matches = matches[matches["phase"] == item_phase]
            if matches.empty:
                raise ValueError(f"No session found for {item!r}.")
            if item_phase is None and len(matches) > 1:
                raise ValueError(
                    f"{subject_id!r} has multiple phases; pass a phase as well."
                )
            selected_rows.append(matches)
        selected = pd.concat(selected_rows, ignore_index=True)

    if phase is not None:
        selected = selected[selected["phase"] == phase]
    for column in _modality_columns(modalities):
        selected = selected[selected[column]]

    return selected.reset_index(drop=True)


def select_subjects(table, selection="all"):
    """Filter any subject-level or trial-level table by subject identifiers."""
    if "subject_id" not in table.columns:
        raise ValueError("The table must contain a 'subject_id' column.")
    if selection == "all":
        return table.copy()
    subjects = [selection] if isinstance(selection, str) else list(selection)
    if not subjects:
        raise ValueError("Choose at least one subject or use 'all'.")
    available = set(table["subject_id"].dropna().unique())
    unknown = sorted(set(subjects) - available)
    if unknown:
        raise ValueError(f"Unknown subjects: {unknown}. Choose from {sorted(available)}.")
    return table.loc[table["subject_id"].isin(subjects)].copy()


def subject_overview(sessions):
    """Show which experimental phases are available for every subject."""
    overview = sessions.assign(available=True).pivot_table(
        index="subject_id", columns="phase", values="available", aggfunc="any", fill_value=False
    )
    columns = [phase for phase in PHASE_ORDER if phase in overview.columns]
    overview = overview.reindex(columns=columns)
    overview.columns.name = None
    return overview.reset_index()


def modality_overview(sessions):
    """Count available modality sessions within each experimental phase."""
    rows = []
    labels = {"tracking": "Tracking", "freezing": "Freezing", "hr": "HR", "events": "Events"}
    for modality, column in MODALITY_COLUMNS.items():
        row = {"modality": labels[modality]}
        for phase in PHASE_ORDER:
            phase_sessions = sessions.loc[sessions["phase"] == phase]
            row[phase] = f"{int(phase_sessions[column].sum())}/{len(phase_sessions)}"
        rows.append(row)
    return pd.DataFrame(rows)
