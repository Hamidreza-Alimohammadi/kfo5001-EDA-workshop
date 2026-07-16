import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ApiNameTests(unittest.TestCase):
    def test_function_names_have_at_most_three_words(self):
        long_names = []
        for directory in ["workshop_tools", "preprocessing"]:
            for path in sorted((REPO_ROOT / directory).glob("*.py")):
                for node in ast.parse(path.read_text()).body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        words = [word for word in node.name.split("_") if word]
                        if len(words) > 3:
                            long_names.append(f"{path.name}:{node.lineno}:{node.name}")

        self.assertEqual(long_names, [])

    def test_functions_follow_module_scopes(self):
        expected_modules = {
            "load_sessions": "session_data.py",
            "load_session": "session_data.py",
            "load_session_files": "session_data.py",
            "choose_session": "session_data.py",
            "select_sessions": "session_data.py",
            "select_subjects": "session_data.py",
            "missing_examples": "diagnostics.py",
            "variable_summary": "diagnostics.py",
            "event_aligned_bins": "synchronization.py",
            "sampling_intervals": "synchronization.py",
            "plot_session_traces": "exploration.py",
            "build_phase_psth": "psth.py",
            "plot_phase_psth": "psth.py",
        }
        found_modules = {}
        for path in sorted((REPO_ROOT / "workshop_tools").glob("*.py")):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found_modules[node.name] = path.name

        for function_name, module_name in expected_modules.items():
            self.assertEqual(found_modules.get(function_name), module_name)


if __name__ == "__main__":
    unittest.main()
