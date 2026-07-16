"""Execute workshop sections without a Jupyter frontend.

Usage:
    python tests/smoke_notebook.py independent
    python tests/smoke_notebook.py sequential
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workshop_tools import (
    diagnostics,
    exploration,
    features,
    patterns,
    psth,
    robustness,
    session_data,
    style,
    synchronization,
    trajectories,
)


NOTEBOOK = Path("KFO5001_EDA_Workshop.ipynb")
SECTION_CELLS = {
    "Section 1": [8, 11, 13, 14, 16, 18, 20, 21, 23, 24],
    "Section 2": [8, 29, 31],
    "Section 3": [8, 34, 36, 38, 40, 42],
    "Section 4": [8, 46, 48, 50, 52, 54],
    "Section 5": [8, 58, 59, 61, 63, 65, 67, 69],
}


def shared_namespace():
    return {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "display": lambda value: value,
        "np": np,
        "pd": pd,
        "plt": plt,
        "SimpleNamespace": SimpleNamespace,
        "diagnostics": diagnostics,
        "exploration": exploration,
        "features": features,
        "patterns": patterns,
        "psth": psth,
        "robustness": robustness,
        "session_data": session_data,
        "style": style,
        "synchronization": synchronization,
        "trajectories": trajectories,
        "sessions": session_data.load_sessions(has_tracking=True, has_events=True),
        "prepared_epoch_features": features.load_prepared_features(),
    }


def execute_cells(cells, namespace, sources, section_prefix):
    started = time.perf_counter()
    original_show = plt.show
    shared_ids = {
        name: id(namespace[name])
        for name in ["sessions", "prepared_epoch_features"]
    }
    shared_copies = {
        name: namespace[name].copy(deep=True)
        for name in ["sessions", "prepared_epoch_features"]
    }
    existing_names = set(namespace)
    plt.show = lambda *args, **kwargs: plt.close("all")
    try:
        for cell_index in cells:
            exec(
                compile(sources[cell_index], f"notebook_cell_{cell_index}", "exec"),
                namespace,
            )
            for name in ["sessions", "prepared_epoch_features"]:
                if id(namespace[name]) != shared_ids[name]:
                    raise AssertionError(f"Cell {cell_index} replaced shared {name}.")
                pd.testing.assert_frame_equal(namespace[name], shared_copies[name])

            new_names = set(namespace) - existing_names
            unexpected = {
                name
                for name in new_names
                if name != "check_setup" and not name.startswith(section_prefix)
            }
            if unexpected:
                raise AssertionError(
                    f"Cell {cell_index} leaked names outside {section_prefix}: "
                    f"{sorted(unexpected)}"
                )
    finally:
        plt.show = original_show
        plt.close("all")
    return time.perf_counter() - started


def main(mode):
    notebook = json.loads(NOTEBOOK.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    if mode == "independent":
        for section, cells in SECTION_CELLS.items():
            section_prefix = section.lower().replace(" ", "_")
            elapsed = execute_cells(
                cells,
                shared_namespace(),
                sources,
                section_prefix,
            )
            print(f"{section}: {elapsed:.1f} s")
    elif mode == "sequential":
        namespace = shared_namespace()
        total_elapsed = 0.0
        for section, cells in SECTION_CELLS.items():
            section_prefix = section.lower().replace(" ", "_")
            total_elapsed += execute_cells(
                cells,
                namespace,
                sources,
                section_prefix,
            )
        print(f"Sequential notebook: {total_elapsed:.1f} s")
    else:
        raise SystemExit("Choose 'independent' or 'sequential'.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "independent")
