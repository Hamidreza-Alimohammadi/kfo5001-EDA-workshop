# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import json
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_data_are_labeled_synthetic(self):
        notice = (REPO_ROOT / "DATA_NOTICE.md").read_text().lower()
        readme = (REPO_ROOT / "README.md").read_text().lower()
        self.assertIn("synthetic surrogate", notice)
        self.assertIn("not experimental observations", notice)
        self.assertIn("synthetic surrogate", readme)

    def test_generator_is_independent(self):
        source = (REPO_ROOT / "scripts" / "generate_surrogate_data.py").read_text()
        self.assertIn("SEED = 5001", source)
        self.assertNotIn("data/original", source)
        self.assertNotIn("read_csv", source)

    def test_session_structure_is_complete(self):
        sessions = pd.read_csv(REPO_ROOT / "data" / "reduced" / "sessions.csv")
        self.assertEqual(sessions["subject_id"].nunique(), 14)
        self.assertEqual(sessions["phase"].unique().tolist(), ["cond", "ext_1", "ext_2"])
        self.assertEqual(len(sessions), 42)
        for session_dir in sessions["reduced_session"]:
            for filename in ["Tracking.csv", "HeartRate.csv", "Events.csv"]:
                self.assertTrue((REPO_ROOT / session_dir / filename).is_file())

    def test_notebook_has_no_stored_execution(self):
        notebook = json.loads((REPO_ROOT / "KFO5001_EDA_Workshop.ipynb").read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(cell["execution_count"] is None for cell in code_cells))
        self.assertTrue(all(not cell["outputs"] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
