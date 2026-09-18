"""
ARC Prize (Abstraction and Reasoning Corpus) Evaluation Harness.

Measures out-of-distribution reasoning and program synthesis on novel visual/symbolic
grid transformation puzzles created by François Chollet.
"""

from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Grid = List[List[int]]


@dataclass
class ARCPair:
    """A demonstration or test input/output grid pair."""
    input_grid: Grid
    output_grid: Grid


@dataclass
class ARCTask:
    """An ARC challenge puzzle with train demonstration pairs and test cases."""
    task_id: str
    train_pairs: List[ARCPair]
    test_pairs: List[ARCPair]


@dataclass
class ARCResult:
    """Evaluation output for an ARC task."""
    task_id: str
    solved: bool
    predicted_grid: Optional[Grid]
    expected_grid: Grid
    duration_seconds: float


ARCSolver = Callable[[ARCTask], Grid]


class ARCEvaluator:
    """
    Evaluator scoring agent reasoning on ARC Prize benchmarks.
    """

    def __init__(self, tasks: Optional[List[ARCTask]] = None):
        self.tasks: List[ARCTask] = tasks or []

    def evaluate_task(self, task: ARCTask, solver_fn: ARCSolver) -> ARCResult:
        """Run agent solver on an ARC task and verify exact grid match."""
        start = time.time()
        test_case = task.test_pairs[0]
        expected = test_case.output_grid

        try:
            predicted = solver_fn(task)
            duration = time.time() - start
            solved = (predicted == expected)
            return ARCResult(
                task_id=task.task_id,
                solved=solved,
                predicted_grid=predicted,
                expected_grid=expected,
                duration_seconds=duration,
            )
        except Exception:
            return ARCResult(
                task_id=task.task_id,
                solved=False,
                predicted_grid=None,
                expected_grid=expected,
                duration_seconds=time.time() - start,
            )

    def run_suite(self, solver_fn: ARCSolver) -> Dict[str, any]:
        """Evaluate entire suite of ARC tasks and compute accuracy."""
        results = [self.evaluate_task(t, solver_fn) for t in self.tasks]
        total = len(results)
        solved_count = sum(1 for r in results if r.solved)
        acc = (solved_count / total * 100.0) if total > 0 else 0.0

        return {
            "total_tasks": total,
            "solved": solved_count,
            "accuracy_percent": acc,
            "details": [
                {
                    "task_id": r.task_id,
                    "solved": r.solved,
                    "duration_s": round(r.duration_seconds, 3),
                }
                for r in results
            ],
        }
