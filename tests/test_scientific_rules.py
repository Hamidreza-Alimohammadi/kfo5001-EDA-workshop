import unittest

import numpy as np
import pandas as pd

from workshop_tools import patterns, session_data, trajectories
from workshop_tools.features import (
    EPOCH_ORDER,
    _freezing_latency,
    _summarize_epoch,
    load_prepared_features,
    plot_epoch_overview,
)
from workshop_tools.signals import freezing_bouts
from workshop_tools.style import EPOCH_LABELS
from workshop_tools.synchronization import sampling_intervals
from workshop_tools.trajectories import summarize_trajectories
from workshop_tools.patterns import build_state_matrix, build_trial_matrix, fit_state_umap, plot_state_umap, plot_trial_patterns, standardize_pattern_matrix, top_feature_correlations


class FreezingBoutTests(unittest.TestCase):
    def test_missing_sample_splits_freezing_bouts(self):
        tracking = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0, 3.0],
                "freezing": [True, np.nan, np.nan, True],
            }
        )

        self.assertEqual(freezing_bouts(tracking), [(0.0, 1.0), (3.0, 4.0)])

    def test_latency_requires_tone_level_coverage(self):
        latency, observed = _freezing_latency(
            [(2.0, 3.0)], 0.0, 4.0, valid_fraction=0.50, min_valid_fraction=0.80
        )

        self.assertTrue(np.isnan(latency))
        self.assertIs(observed, pd.NA)


class EpochTerminologyTests(unittest.TestCase):
    def test_epoch_names_are_exact_and_consistent(self):
        expected = ["Pre-Tone", "Early-Tone", "Late-Tone", "Post-Tone"]

        self.assertEqual(EPOCH_ORDER, expected)
        self.assertEqual(EPOCH_LABELS, expected)
        self.assertEqual(
            load_prepared_features()["epoch"].cat.categories.tolist(),
            expected,
        )


class PhaseTerminologyTests(unittest.TestCase):
    def test_phase_codes_are_exact_and_consistent(self):
        expected = ["cond", "ext_1", "ext_2"]

        self.assertEqual(session_data.PHASE_ORDER, expected)
        self.assertEqual(trajectories.PHASE_ORDER, expected[1:])
        self.assertEqual(patterns.TRIAL_PHASES, expected[1:])
        self.assertEqual(session_data.load_sessions()["phase"].unique().tolist(), expected)
        self.assertEqual(load_prepared_features()["phase"].unique().tolist(), expected[1:])
        self.assertFalse(hasattr(session_data, "PHASE_ALIASES"))


class SamplingIntervalTests(unittest.TestCase):
    def test_missing_timestamp_is_not_bridged(self):
        timestamps = pd.Series([0.0, 1.0, np.nan, 3.0, 4.0])

        intervals = sampling_intervals(timestamps)

        self.assertEqual(intervals.tolist(), [1.0, 1.0])


class CorrelationRankingTests(unittest.TestCase):
    def test_top_correlations_are_unique_and_ranked_by_magnitude(self):
        matrix = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "feature_b": [2.0, 4.0, 6.0, 8.0],
                "feature_c": [1.0, 1.0, 2.0, 3.0],
            }
        )

        ranked = top_feature_correlations(matrix, top_n=2)

        self.assertEqual(len(ranked), 2)
        self.assertTrue((ranked["feature_1"] != ranked["feature_2"]).all())
        self.assertTrue(ranked["absolute_r"].is_monotonic_decreasing)
        self.assertAlmostEqual(ranked.iloc[0]["pearson_r"], 1.0)


class TrialPatternTests(unittest.TestCase):
    def test_trial_matrix_preserves_32_trials_per_signal(self):
        import matplotlib.pyplot as plt

        trial_matrix = build_trial_matrix(load_prepared_features(), signals=("freezing", "motion"), epoch="Late-Tone")
        complete = trial_matrix.dropna()
        standardized = standardize_pattern_matrix(complete)
        figure, axes, subject_order = plot_trial_patterns(standardized)

        self.assertEqual(trial_matrix.shape, (14, 64))
        self.assertGreaterEqual(len(complete), 2)
        self.assertLessEqual(len(complete), 14)
        self.assertEqual(len(subject_order), len(complete))
        self.assertEqual(len(axes), 2)
        plt.close(figure)


class PooledStateTests(unittest.TestCase):
    def test_pooled_state_embedding_retains_metadata_alignment(self):
        import matplotlib.pyplot as plt

        state_matrix, state_metadata = build_state_matrix(load_prepared_features())
        complete = ~state_matrix.isna().any(axis=1)
        test_matrix = standardize_pattern_matrix(state_matrix.loc[complete].iloc[:120])
        test_metadata = state_metadata.loc[test_matrix.index]
        embedding, diagnostics = fit_state_umap(test_matrix, n_neighbors=15)
        figure, axes = plot_state_umap(embedding, test_metadata)

        self.assertEqual(state_matrix.shape, (1792, 4))
        self.assertGreater(complete.sum(), 1000)
        self.assertEqual(state_metadata["subject_id"].nunique(), 14)
        self.assertTrue(embedding.index.equals(test_metadata.index))
        self.assertTrue(0 <= diagnostics["trustworthiness"] <= 1)
        self.assertEqual(axes.shape, (2, 2))
        plt.close(figure)


class EpochCoverageTests(unittest.TestCase):
    def test_low_motion_coverage_only_masks_motion_features(self):
        tracking = pd.DataFrame(
            {
                "time_s": np.arange(10, dtype=float),
                "freezing": [False] * 10,
                "motion": [1.0] * 7 + [np.nan] * 3,
                "x": np.arange(10, dtype=float),
                "y": np.arange(10, dtype=float),
                "speed_px_s": [np.nan] + [np.sqrt(2)] * 9,
                "step_distance_px": [np.nan] + [np.sqrt(2)] * 9,
                "previous_time_s": [np.nan] + list(np.arange(9, dtype=float)),
            }
        )
        heart_rate = pd.DataFrame(
            {"time_s": np.linspace(0.1, 9.9, 20), "hr_bpm": [500.0] * 20}
        )

        summary = _summarize_epoch(
            tracking,
            heart_rate,
            freezing_bouts(tracking),
            0.0,
            10.0,
            min_valid_fraction=0.80,
            min_hr_samples=10,
        )

        self.assertTrue(np.isnan(summary["motion_median"]))
        self.assertEqual(summary["motion_valid_count"], 7)
        self.assertEqual(summary["freezing_fraction"], 0.0)
        self.assertEqual(summary["hr_median_bpm"], 500.0)


class EpochOverviewTests(unittest.TestCase):
    def test_custom_feature_specs_create_separate_figures(self):
        import matplotlib.pyplot as plt

        feature_specs = [("motion_median", "Motion"), ("hr_median_bpm", "HR")]
        figures, axes = plot_epoch_overview(load_prepared_features(), feature_specs=feature_specs)

        self.assertEqual(len(figures), 2)
        self.assertEqual([axis.get_ylabel() for axis in axes], ["Motion", "HR"])
        for figure in figures:
            plt.close(figure)

    def test_repeated_feature_does_not_break_boxplot(self):
        import matplotlib.pyplot as plt

        feature_specs = [("motion_median", "Motion"), ("motion_median", "Motion again")]
        figures, axes = plot_epoch_overview(load_prepared_features(), feature_specs=feature_specs)

        self.assertEqual(len(figures), 2)
        self.assertEqual([axis.get_ylabel() for axis in axes], ["Motion", "Motion again"])
        for figure in figures:
            plt.close(figure)


class TrajectorySufficiencyTests(unittest.TestCase):
    @staticmethod
    def _features(valid_trials):
        return pd.DataFrame(
            {
                "subject_id": ["subject_01"] * 16,
                "phase": ["ext_1"] * 16,
                "trial": np.arange(1, 17),
                "epoch": ["Late-Tone"] * 16,
                "freezing_fraction": [
                    float(trial) if trial in valid_trials else np.nan
                    for trial in range(1, 17)
                ],
            }
        )

    def test_two_trials_do_not_define_a_trajectory(self):
        summary = summarize_trajectories(self._features({1, 16})).iloc[0]

        self.assertTrue(np.isnan(summary["linear_trend_per_trial"]))
        self.assertTrue(np.isnan(summary["early_block_median"]))
        self.assertEqual(summary["valid_trial_count"], 2)

    def test_twelve_trials_are_enough_for_a_known_linear_trend(self):
        summary = summarize_trajectories(
            self._features(set(range(1, 13)))
        ).iloc[0]

        self.assertAlmostEqual(summary["linear_trend_per_trial"], 1.0)
        self.assertAlmostEqual(summary["early_block_median"], 2.5)
        self.assertTrue(np.isnan(summary["late_block_median"]))


if __name__ == "__main__":
    unittest.main()
