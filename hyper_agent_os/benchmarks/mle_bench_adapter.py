"""
MLE-bench (Kaggle) Benchmark Evaluation Adapter.

Evaluates autonomous AI agents across machine learning engineering competitions
curated from Kaggle, grading submissions against historical human leaderboards
to assign Kaggle Medal tiers (Bronze, Silver, Gold).
"""

from __future__ import annotations
import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


class MedalTier(enum.Enum):
    NONE = "NONE"
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"


@dataclass
class MLEBenchCompetition:
    """A benchmark competition instance modeled after OpenAI MLE-bench / Kaggle."""
    competition_id: str
    title: str
    metric: str  # e.g., "accuracy", "roc_auc", "rmse", "f1"
    bronze_threshold: float
    silver_threshold: float
    gold_threshold: float
    higher_is_better: bool = True
    dataset_description: str = ""


@dataclass
class MLEBenchResult:
    """Outcome of an agent's end-to-end ML solution."""
    competition_id: str
    metric_achieved: float
    medal: MedalTier
    execution_time_seconds: float
    submission_valid: bool
    summary: str


AgentMLExecutor = Callable[[MLEBenchCompetition], float]


class MLEBenchRunner:
    """
    Evaluates ML engineering agents on real Kaggle competition suites.
    """

    def __init__(self, competitions: Optional[List[MLEBenchCompetition]] = None):
        self.competitions: List[MLEBenchCompetition] = competitions or []

    def evaluate_competition(
        self, comp: MLEBenchCompetition, agent_fn: AgentMLExecutor
    ) -> MLEBenchResult:
        """Run agent on an ML competition and compute medal tier."""
        start = time.time()
        try:
            score = agent_fn(comp)
            duration = time.time() - start

            # Determine medal tier
            if comp.higher_is_better:
                if score >= comp.gold_threshold:
                    medal = MedalTier.GOLD
                elif score >= comp.silver_threshold:
                    medal = MedalTier.SILVER
                elif score >= comp.bronze_threshold:
                    medal = MedalTier.BRONZE
                else:
                    medal = MedalTier.NONE
            else:
                if score <= comp.gold_threshold:
                    medal = MedalTier.GOLD
                elif score <= comp.silver_threshold:
                    medal = MedalTier.SILVER
                elif score <= comp.bronze_threshold:
                    medal = MedalTier.BRONZE
                else:
                    medal = MedalTier.NONE

            return MLEBenchResult(
                competition_id=comp.competition_id,
                metric_achieved=score,
                medal=medal,
                execution_time_seconds=duration,
                submission_valid=True,
                summary=f"Achieved {comp.metric}={score:.4f} -> {medal.value} Medal",
            )
        except Exception as err:
            return MLEBenchResult(
                competition_id=comp.competition_id,
                metric_achieved=0.0,
                medal=MedalTier.NONE,
                execution_time_seconds=time.time() - start,
                submission_valid=False,
                summary=f"Error: {str(err)}",
            )

    def run_suite(self, agent_fn: AgentMLExecutor) -> Dict[str, any]:
        """Evaluate agent across all competitions and aggregate medal rates."""
        results = [self.evaluate_competition(c, agent_fn) for c in self.competitions]
        total = len(results)

        gold_count = sum(1 for r in results if r.medal == MedalTier.GOLD)
        silver_count = sum(1 for r in results if r.medal == MedalTier.SILVER)
        bronze_count = sum(1 for r in results if r.medal == MedalTier.BRONZE)
        any_medal_count = gold_count + silver_count + bronze_count

        return {
            "total_competitions": total,
            "any_medal_count": any_medal_count,
            "medal_rate_percent": (any_medal_count / total * 100.0) if total > 0 else 0.0,
            "gold_count": gold_count,
            "silver_count": silver_count,
            "bronze_count": bronze_count,
            "details": [
                {
                    "competition_id": r.competition_id,
                    "score": r.metric_achieved,
                    "medal": r.medal.value,
                    "time_s": round(r.execution_time_seconds, 2),
                }
                for r in results
            ],
        }
