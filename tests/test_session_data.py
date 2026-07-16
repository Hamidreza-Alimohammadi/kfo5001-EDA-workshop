import unittest

import pandas as pd

from workshop_tools.session_data import choose_session, modality_overview, select_sessions, select_subjects, subject_overview


class SessionSelectionTests(unittest.TestCase):
    @staticmethod
    def _sessions():
        return pd.DataFrame(
            {
                "subject_id": ["subject_01", "subject_01", "subject_02", "subject_02"],
                "phase": ["ext_1", "ext_2", "ext_1", "ext_2"],
                "has_tracking": [True, True, True, True],
                "has_freezing": [True, True, False, True],
                "has_heartbeats": [True, False, True, True],
                "has_events": [True, True, True, True],
            }
        )

    def test_session_key_uses_canonical_phase(self):
        selected = choose_session(self._sessions(), ("subject_01", "ext_1"), modalities=("hr",))

        self.assertEqual(selected["phase"], "ext_1")

    def test_noncanonical_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown phases"):
            choose_session(self._sessions(), ("subject_01", "extinction_1"))

    def test_required_modality_rejects_unavailable_session(self):
        with self.assertRaisesRegex(ValueError, "modalities"):
            choose_session(self._sessions(), ("subject_01", "ext_2"), modalities=("hr",))

    def test_all_phase_selection_can_require_modalities(self):
        selected = select_sessions(self._sessions(), ("all", "ext_2"), modalities="hr")

        self.assertEqual(selected["subject_id"].tolist(), ["subject_02"])

    def test_subject_selection_accepts_one_or_many(self):
        table = pd.DataFrame({"subject_id": ["subject_01", "subject_02"], "value": [1, 2]})

        one = select_subjects(table, "subject_01")
        many = select_subjects(table, ["subject_02", "subject_01"])

        self.assertEqual(one["subject_id"].tolist(), ["subject_01"])
        self.assertEqual(set(many["subject_id"]), {"subject_01", "subject_02"})

    def test_overviews_expose_subjects_and_modalities(self):
        sessions = self._sessions()

        subjects = subject_overview(sessions)
        modalities = modality_overview(sessions)

        self.assertEqual(subjects["subject_id"].tolist(), ["subject_01", "subject_02"])
        self.assertEqual(modalities.loc[modalities["modality"] == "HR", "ext_2"].iloc[0], "1/2")


if __name__ == "__main__":
    unittest.main()
