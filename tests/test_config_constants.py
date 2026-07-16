import ast
import unittest
from pathlib import Path

from workshop_tools import features, session_data, style


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigConstantTests(unittest.TestCase):
    def test_capitalized_defaults_are_resolved_at_call_time(self):
        captured_defaults = []
        for path in sorted((REPO_ROOT / "workshop_tools").glob("*.py")):
            tree = ast.parse(path.read_text())
            constants = {
                node.targets[0].id
                for node in tree.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.isupper()
            }
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                defaults = [default for default in node.args.defaults + node.args.kw_defaults if default]
                names = {name.id for default in defaults for name in ast.walk(default) if isinstance(name, ast.Name)}
                for constant in sorted(names & constants):
                    captured_defaults.append(f"{path.name}:{node.name}:{constant}")

        self.assertEqual(captured_defaults, [])

    def test_cross_module_constants_use_module_lookup(self):
        direct_imports = []
        module_lookups = [
            ("features.py", "style"),
            ("exploration.py", "style"),
            ("psth.py", "style"),
            ("synchronization.py", "style"),
            ("trajectories.py", "style"),
            ("trajectories.py", "features"),
            ("robustness.py", "trajectories"),
        ]
        for filename, module_name in module_lookups:
            source = (REPO_ROOT / "workshop_tools" / filename).read_text()
            if f"from workshop_tools.{module_name} import" in source:
                direct_imports.append(filename)
        self.assertEqual(direct_imports, [])

    def test_reassigned_feature_constants_change_default_behavior(self):
        epoch_features = features.load_prepared_features()
        original_overview = features.OVERVIEW_FEATURES
        original_contrasts = features.CONTRAST_FEATURES
        try:
            features.OVERVIEW_FEATURES = [("motion_median", "Custom motion")]
            figure, axes = features.plot_epoch_overview(epoch_features)
            self.assertEqual(len(axes), 1)
            self.assertEqual(axes[0].get_ylabel(), "Custom motion")

            features.CONTRAST_FEATURES = ["freezing_fraction"]
            contrasts = features.build_epoch_contrasts(epoch_features)
            self.assertEqual(contrasts["feature"].unique().tolist(), ["freezing_fraction"])
        finally:
            features.OVERVIEW_FEATURES = original_overview
            features.CONTRAST_FEATURES = original_contrasts
            import matplotlib.pyplot as plt

            plt.close("all")

    def test_freezing_layer_is_solid_above_epoch_shading(self):
        sessions = session_data.load_sessions(has_tracking=True, has_events=True)
        session = session_data.choose_session(sessions, ("subject_01", "ext_1"))
        original_colors = style.SIGNAL_COLORS
        try:
            style.SIGNAL_COLORS = {**original_colors, "freezing": "#123456"}
            figure, axes = features.plot_trial_epochs(session, trial=1)
            freezing_image = axes[2].images[0]
            self.assertEqual(freezing_image.cmap(0.0)[-1], 0.0)
            self.assertEqual(freezing_image.cmap(1.0)[-1], 1.0)
            self.assertGreater(freezing_image.get_zorder(), max(patch.get_zorder() for patch in axes[2].patches))
        finally:
            style.SIGNAL_COLORS = original_colors
            import matplotlib.pyplot as plt

            plt.close("all")


if __name__ == "__main__":
    unittest.main()
