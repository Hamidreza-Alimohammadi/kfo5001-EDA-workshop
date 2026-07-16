import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "KFO5001_EDA_Workshop.ipynb"
SECTION_CELLS = {
    1: [11, 13, 14, 16, 18, 20, 21, 23, 24],
    2: [29, 31],
    3: [34, 36, 38, 40, 42],
    4: [46, 48, 50, 52, 54],
    5: [58, 59, 61, 63, 65, 67, 69],
}


def assigned_names(source):
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


class NotebookNamespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())

    def test_section_assignments_use_their_own_prefix(self):
        for section_number, cell_indices in SECTION_CELLS.items():
            prefix = f"section_{section_number}"
            names = set()
            for cell_index in cell_indices:
                source = "".join(self.notebook["cells"][cell_index]["source"])
                names.update(assigned_names(source))

            unexpected = sorted(name for name in names if not name.startswith(prefix))
            self.assertEqual(unexpected, [], f"Section {section_number} leaked names")

    def test_every_section_initializes_one_state_object(self):
        for section_number, cell_indices in SECTION_CELLS.items():
            entry_source = "".join(
                self.notebook["cells"][cell_indices[0]]["source"]
            )
            expected = f"section_{section_number} = SimpleNamespace()"
            self.assertIn(expected, entry_source)

    def test_prepared_and_live_feature_tables_have_distinct_names(self):
        setup_source = "".join(self.notebook["cells"][6]["source"])
        extraction_source = "".join(self.notebook["cells"][36]["source"])

        self.assertIn(
            "prepared_epoch_features = features.load_prepared_features()",
            setup_source,
        )
        self.assertIn(
            "section_3.extracted_epoch_features =",
            extraction_source,
        )
        self.assertNotIn("section_3.epoch_features", extraction_source)

    def test_modules_use_one_canonical_name(self):
        all_sources = "\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
        )
        for alias in [
            "section_1_tools",
            "section_2_tools",
            "section_3_tools",
            "section_4_tools",
            "section_5_tools",
            "section_6_pattern_tools",
        ]:
            self.assertNotIn(alias, all_sources)

    def test_sections_use_consistent_selection_names(self):
        sources = {
            section_number: "\n".join(
                "".join(self.notebook["cells"][cell_index]["source"])
                for cell_index in cell_indices
            )
            for section_number, cell_indices in SECTION_CELLS.items()
        }
        for section_number in [1, 2, 3]:
            self.assertIn(f"section_{section_number}.session_key", sources[section_number])
        for section_number in [4, 5]:
            self.assertIn(f"section_{section_number}.subject_selection", sources[section_number])


if __name__ == "__main__":
    unittest.main()
