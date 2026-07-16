# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

from pathlib import Path

import pandas as pd


def check_setup(sessions=None, epoch_features=None):
    """Check the shared state required by every workshop section."""
    if sessions is None:
        sessions = pd.read_csv("data/reduced/sessions.csv")
    if epoch_features is None:
        from workshop_tools.features import load_prepared_features

        epoch_features = load_prepared_features()

    required_session_columns = {"subject_id", "phase", "reduced_session"}
    required_feature_columns = {
        "subject_id",
        "phase",
        "trial",
        "epoch",
        "freezing_fraction",
        "motion_median",
        "hr_median_bpm",
    }
    missing_session_columns = required_session_columns - set(sessions.columns)
    missing_feature_columns = required_feature_columns - set(epoch_features.columns)
    if missing_session_columns or missing_feature_columns:
        raise RuntimeError(
            "Setup check failed. Missing columns: "
            f"sessions={sorted(missing_session_columns)}, "
            f"epoch_features={sorted(missing_feature_columns)}"
        )

    referenced_files = [
        Path(session_dir) / filename
        for session_dir in sessions["reduced_session"]
        for filename in ["Tracking.csv", "Events.csv"]
    ]
    missing_files = [path for path in referenced_files if not path.exists()]
    if missing_files:
        raise RuntimeError(f"Setup check failed. Missing data file: {missing_files[0]}")

    extinction_subjects = sessions.loc[
        sessions["phase"].isin(["ext_1", "ext_2"]), "subject_id"
    ].nunique()

    print("Repository ready")
    print("Workshop tools ready")
    print(f"Reduced session table loaded ({len(sessions)} sessions)")
    print(
        "Prepared extinction feature table loaded "
        f"({len(epoch_features)} rows, {extinction_subjects} subjects)"
    )
    print("\nYOU ARE READY FOR THE WORKSHOP")


def missing_examples(tables, max_rows=5):
    rows = []
    for table_name, table in tables.items():
        missing_rows = table[table.isna().any(axis=1)].head(max_rows)
        for row_index, row in missing_rows.iterrows():
            missing_columns = row.index[row.isna()].tolist()
            rows.append(
                {
                    "table": table_name,
                    "row_index": row_index,
                    "missing_columns": ", ".join(missing_columns),
                    "row_values": row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def variable_summary(tables, transpose=False, max_categories=8):
    rows = []
    for table_name, table in tables.items():
        for column in table.columns:
            values = table[column]
            numeric_values = pd.to_numeric(values, errors="coerce")
            is_numeric = values.dtype.kind in "iufc"
            categories = None
            if not is_numeric:
                categories = values.dropna().unique()[:max_categories].tolist()
            rows.append(
                {
                    "table": table_name,
                    "variable": column,
                    "dtype": str(values.dtype),
                    "rows": len(values),
                    "missing": int(values.isna().sum()),
                    "missing_percent": 100 * values.isna().mean(),
                    "unique_values": values.nunique(dropna=True),
                    "categories": categories,
                    "min": numeric_values.min() if is_numeric else None,
                    "max": numeric_values.max() if is_numeric else None,
                }
            )
    summary = pd.DataFrame(rows)
    if transpose:
        return summary.set_index(["table", "variable"]).T
    return summary
