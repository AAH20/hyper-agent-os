"""Unit tests for Bi-Temporal Super-Memory and Spatial Index."""

import time
import unittest
from hyper_agent_os.memory import (
    BiTemporalKnowledgeGraph,
    TemporalQuery,
    SpatialMultimodalIndex,
    BoundingBox3D,
    VectorEmbedding,
    UnifiedRetriever,
)


class TestMemorySubsystem(unittest.TestCase):

    def setUp(self):
        self.kg = BiTemporalKnowledgeGraph()
        self.spatial = SpatialMultimodalIndex()

    def test_bi_temporal_assertion_and_as_of_query(self):
        t0 = 1000.0
        t1 = 2000.0

        # Assert fact valid from t0 to t1
        f1 = self.kg.assert_fact("AgentX", "role", "Worker", valid_from=t0, valid_to=t1)
        self.assertTrue(f1.is_valid_at(1500.0))
        self.assertFalse(f1.is_valid_at(2500.0))

        # Query as of valid time 1500
        results = self.kg.query(TemporalQuery(as_of_valid_time=1500.0, subject="AgentX"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].target, "Worker")

        # Query as of valid time 2500 -> should be empty
        results_after = self.kg.query(TemporalQuery(as_of_valid_time=2500.0, subject="AgentX"))
        self.assertEqual(len(results_after), 0)

    def test_fact_retraction_rollback(self):
        f = self.kg.assert_fact("LLM", "output", "HallucinatedCode")
        self.assertTrue(f.is_active)

        # Retract hallucination
        retracted = self.kg.retract_fact(f.id)
        self.assertTrue(retracted)
        self.assertFalse(f.is_active)

        # Active query should not include retracted fact
        active = self.kg.query(TemporalQuery(subject="LLM"))
        self.assertEqual(len(active), 0)

    def test_spatial_bounding_box_and_collision(self):
        box1 = BoundingBox3D(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        box2 = BoundingBox3D(0.5, 0.5, 0.5, 1.5, 1.5, 1.5)
        box3 = BoundingBox3D(5.0, 5.0, 5.0, 6.0, 6.0, 6.0)

        self.assertTrue(box1.intersects(box2))
        self.assertFalse(box1.intersects(box3))
        self.assertTrue(box1.contains((0.5, 0.5, 0.5)))
        self.assertFalse(box1.contains((2.0, 0.5, 0.5)))

    def test_multimodal_vector_similarity(self):
        e1 = self.spatial.register_spatial_object(
            name="RoboticArm",
            category="actuator",
            bbox=BoundingBox3D(0, 0, 0, 1, 1, 1),
            embedding=[1.0, 0.0, 0.0],
        )
        e2 = self.spatial.register_spatial_object(
            name="CameraSensor",
            category="sensor",
            bbox=BoundingBox3D(2, 2, 2, 3, 3, 3),
            embedding=[0.0, 1.0, 0.0],
        )

        matches = self.spatial.query_vector_similarity([0.95, 0.05, 0.0], top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].name, "RoboticArm")
        self.assertGreater(matches[0][1], 0.9)

    def test_unified_multi_hop_retrieval(self):
        self.kg.assert_fact("User", "owns", "RobotArm")
        self.kg.assert_fact("RobotArm", "controlled_by", "ROS2Bridge")
        self.kg.assert_fact("ROS2Bridge", "governed_by", "ControlBarrierFunction")

        retriever = UnifiedRetriever(self.kg, self.spatial)
        res = retriever.retrieve("Audit chain", subject="User", max_hops=3)

        self.assertEqual(len(res.facts), 1)
        self.assertGreaterEqual(len(res.graph_paths), 2)
        self.assertIn("ROS2Bridge", res.synthesized_context)


if __name__ == "__main__":
    unittest.main()
