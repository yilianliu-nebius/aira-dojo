"""Behavioral checks for the bounded optimizer and immutable evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from dflash2_integration.aira_runner import ASSETS, candidate_from_code
from dflash2_integration.candidate import architecture_fingerprint, estimated_parameters, validate_candidate
from dflash2_integration.evaluate import teacher_matching, verify_protected


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "kind": "recipe", "hypothesis": "Improve teacher alignment",
            "predicted_metric": "Reduce selector error", "selector_rank": 256,
            "learning_rate": 0.00003, "selector_loss_weight": 0.1, "seed": 42,
            "architecture": {"convolutions": [], "correctors": []},
        }
        self.schema = Draft202012Validator(json.loads((ASSETS / "candidate.schema.json").read_text()))

    def test_generated_calls_cannot_write_files(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            code = f"candidate = __import__('pathlib').Path({str(marker)!r}).write_text('unsafe')"
            with self.assertRaises(ValueError):
                candidate_from_code(code, self.schema, "recipe")
            self.assertFalse(marker.exists())

    def test_duplicate_keys_cannot_change_the_proposal(self):
        entries = [f"{key!r}: {value!r}" for key, value in self.candidate.items()]
        entries.append("'selector_rank': 384")
        with self.assertRaises(ValueError):
            candidate_from_code("candidate = {" + ", ".join(entries) + "}", self.schema, "recipe")

    def test_nonfinite_and_boolean_numbers_are_rejected(self):
        cases = (("seed", True), ("selector_rank", True), ("learning_rate", True),
                 ("learning_rate", float("nan")), ("selector_loss_weight", float("inf")))
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_candidate({**self.candidate, key: value})

    def test_architecture_and_recipe_changes_remain_separate(self):
        cases = (
            {"kind": "architecture", "selector_rank": 384, "learning_rate": 0.0001},
            {"kind": "recipe", "selector_rank": 384},
            {"kind": "architecture", "selector_rank": 384},
            {"kind": "architecture", "selector_rank": 512},
            {"kind": "architecture", "selector_rank": 320},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_candidate({**self.candidate, **changes})

    def test_optimizer_cannot_add_an_evaluator_control(self):
        with self.assertRaises(ValueError):
            validate_candidate({**self.candidate, "temperature": 0})

    def test_composed_literal_preserves_the_physical_parameter_gate(self):
        candidate = {
            **self.candidate, "kind": "architecture",
            "architecture": {
                "convolutions": [{"layer": 2, "sublayer": "attention", "taps": [0, 1, 3], "group_size": 8}],
                "correctors": [{"layer": 4, "rank": 32, "gated": True}],
            },
        }
        parsed = candidate_from_code("candidate = " + repr(candidate), self.schema, "architecture")
        self.assertEqual(estimated_parameters(parsed), 1944418560)
        with self.assertRaises(ValueError):
            validate_candidate({
                **candidate, "selector_rank": 384,
                "architecture": {
                    "convolutions": [{"layer": 2, "sublayer": "attention", "taps": [0, 1, 7], "group_size": 8}],
                    "correctors": candidate["architecture"]["correctors"],
                },
            })

    def test_duplicate_placements_and_invalid_sparse_stencils_are_rejected(self):
        entry = {"layer": 1, "sublayer": "mlp", "taps": [0, 1, 3], "group_size": 16}
        plans = (
            {"convolutions": [entry, entry], "correctors": []},
            {"convolutions": [{**entry, "taps": [0, 3, 1]}], "correctors": []},
            {"convolutions": [{**entry, "taps": [0, 1, 8]}], "correctors": []},
            {"convolutions": [{**entry, "taps": [0, 1]}], "correctors": []},
            {"convolutions": [], "correctors": [{"layer": 0, "rank": True, "gated": True}]},
            {"convolutions": [], "correctors": [{"layer": 0, "rank": 32, "gated": "yes"}]},
            {"convolutions": [], "correctors": [{"layer": 0, "rank": 32, "gated": False}] * 2},
        )
        for plan in plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_candidate({**self.candidate, "kind": "architecture", "architecture": plan})

    def test_graph_identity_ignores_prose_and_declaration_order(self):
        first = validate_candidate({
            **self.candidate, "kind": "architecture",
            "architecture": {
                "convolutions": [],
                "correctors": [{"layer": 3, "rank": 64, "gated": False}, {"layer": 1, "rank": 32, "gated": True}],
            },
        })
        second = validate_candidate({
            **first, "hypothesis": "A different explanation", "predicted_metric": "A different prediction",
            "architecture": {**first["architecture"], "correctors": list(reversed(first["architecture"]["correctors"]))},
        })
        self.assertEqual(architecture_fingerprint(first), architecture_fingerprint(second))
        changed = validate_candidate({**first, "selector_rank": 384})
        self.assertNotEqual(architecture_fingerprint(first), architecture_fingerprint(changed))

    def test_unchanged_architecture_requires_an_explicit_control(self):
        baseline = {**self.candidate, "kind": "architecture"}
        with self.assertRaises(ValueError):
            validate_candidate(baseline)
        self.assertEqual(validate_candidate(baseline, allow_baseline=True), baseline)

    def test_changed_protected_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "policy.json"
            original = b'{"max_output_tokens":2048}'
            file.write_bytes(original)
            manifest = {str(file): hashlib.sha256(original).hexdigest()}
            file.write_bytes(b'{"max_output_tokens":16}')
            with self.assertRaises(RuntimeError):
                verify_protected(manifest)

    def test_teacher_ratios_pool_counts_instead_of_conversation_means(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for numerator, denominator in ((9, 10), (0, 1)):
                ratios = {"acc": [numerator, denominator], "candidate_ids": [12345, 54321]}
                for position in range(1, 8):
                    ratios[f"position_{position}/teacher/unary_distribution_overlap"] = [numerator, denominator]
                records.append({"file": "private-prefix.npz", "ratio_metrics": ratios})
            file = root / "diagnostics.json"
            file.write_text(json.dumps(records))
            result = teacher_matching({"diagnostics": {"before": str(file), "after": str(file)}}, root)
            self.assertAlmostEqual(result["before"]["pooled_ratios"]["token_accuracy"], 9 / 11)
            self.assertEqual(result["after"]["position_bins"]["teacher_overlap"], [9 / 11] * 7)
            self.assertNotIn("candidate_ids", json.dumps(result))
            self.assertNotIn("private-prefix", json.dumps(result))

    def test_zero_denominators_do_not_create_fake_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "diagnostics.json"
            file.write_text(json.dumps([{"ratio_metrics": {"acc": [0, 0]}}]))
            result = teacher_matching({"diagnostics": {"before": str(file), "after": str(file)}}, root)
            self.assertNotIn("token_accuracy", result["before"]["pooled_ratios"])


if __name__ == "__main__":
    unittest.main()
