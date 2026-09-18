"""Unit tests for SWE-bench, MLE-bench, and ARC Prize benchmark harnesses."""

import unittest
from hyper_agent_os.benchmarks import (
    SWEBenchTask,
    SWEBenchRunner,
    MLEBenchCompetition,
    MLEBenchRunner,
    MedalTier,
    ARCTask,
    ARCPair,
    ARCEvaluator,
)


class TestBenchmarksSubsystem(unittest.TestCase):

    def test_swe_bench_verified_evaluator(self):
        task1 = SWEBenchTask(
            instance_id="django__django-11000",
            repo="django/django",
            base_commit="d1100",
            problem_statement="Fix database connection leakage",
            fail_to_pass_tests=["test_connection_leak"],
            pass_to_pass_tests=["test_query"],
        )

        runner = SWEBenchRunner([task1])

        # Valid diff patch fixing the issue
        valid_patch = (
            "diff --git a/django/db/backends.py b/django/db/backends.py\n"
            "--- a/django/db/backends.py\n"
            "+++ b/django/db/backends.py\n"
            "@@ -10,3 +10,4 @@\n"
            " def test_connection_leak():\n"
            "+    close_idle_connections()\n"
        )
        res_valid = runner.evaluate_task(task1, lambda t: valid_patch)
        self.assertTrue(res_valid.resolved)

        # Invalid patch
        invalid_patch = "random text without diff headers"
        res_invalid = runner.evaluate_task(task1, lambda t: invalid_patch)
        self.assertFalse(res_invalid.resolved)

    def test_mle_bench_kaggle_medal_scoring(self):
        comp = MLEBenchCompetition(
            competition_id="house-prices-advanced-regression",
            title="House Prices",
            metric="rmse",
            bronze_threshold=0.15,
            silver_threshold=0.12,
            gold_threshold=0.10,
            higher_is_better=False,  # Lower RMSE is better
        )

        runner = MLEBenchRunner([comp])

        # Score achieving Silver (0.11 is <= 0.12)
        res = runner.evaluate_competition(comp, lambda c: 0.11)
        self.assertEqual(res.medal, MedalTier.SILVER)

        # Score achieving Gold (0.09 is <= 0.10)
        res_gold = runner.evaluate_competition(comp, lambda c: 0.09)
        self.assertEqual(res_gold.medal, MedalTier.GOLD)

        # Score below Bronze (0.20 > 0.15)
        res_none = runner.evaluate_competition(comp, lambda c: 0.20)
        self.assertEqual(res_none.medal, MedalTier.NONE)

    def test_arc_reasoning_evaluator(self):
        train_p = ARCPair([[1, 0], [0, 1]], [[0, 1], [1, 0]])
        test_p = ARCPair([[2, 0], [0, 2]], [[0, 2], [2, 0]])

        task = ARCTask("arc_rot_01", [train_p], [test_p])
        evaluator = ARCEvaluator([task])

        # Correct solver (swapping diagonals)
        res_correct = evaluator.evaluate_task(task, lambda t: [[0, 2], [2, 0]])
        self.assertTrue(res_correct.solved)

        # Incorrect solver
        res_incorrect = evaluator.evaluate_task(task, lambda t: [[2, 2], [0, 0]])
        self.assertFalse(res_incorrect.solved)


if __name__ == "__main__":
    unittest.main()
