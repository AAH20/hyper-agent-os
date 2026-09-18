"""Unit tests for Control Barrier Function Safety Governor and Aegis Attestation."""

import unittest
from hyper_agent_os.safety import (
    ControlBarrierFunction,
    SafetyEnvelope,
    ActuatorCommand,
    AegisAttestationBridge,
)


class TestSafetySubsystem(unittest.TestCase):

    def setUp(self):
        self.envelope = SafetyEnvelope(
            max_linear_velocity=2.0,
            max_angular_velocity=1.0,
            max_force_torque=40.0,
            min_obstacle_distance=0.5,
            allowed_workspace_bounds=(-5.0, -5.0, 5.0, 5.0),
        )
        self.cbf = ControlBarrierFunction(self.envelope)

    def test_safe_action_passthrough(self):
        safe_cmd = ActuatorCommand(
            target_id="robot_01",
            vx=1.0,
            vy=0.5,
            omega=0.2,
            force=20.0,
            current_position=(0.0, 0.0),
            nearest_obstacle_distance=2.0,
        )
        filtered = self.cbf.filter_action(safe_cmd)

        self.assertFalse(filtered.was_modified)
        self.assertEqual(filtered.safe_command.vx, 1.0)
        self.assertGreaterEqual(filtered.barrier_value, 0.0)

    def test_speed_and_torque_clamping(self):
        excessive_cmd = ActuatorCommand(
            target_id="robot_01",
            vx=3.0,
            vy=4.0,  # speed = 5.0 m/s > 2.0 m/s
            omega=2.5,  # omega > 1.0
            force=80.0,  # force > 40.0
            current_position=(0.0, 0.0),
            nearest_obstacle_distance=2.0,
        )
        filtered = self.cbf.filter_action(excessive_cmd)

        self.assertTrue(filtered.was_modified)
        # Scaled speed should be <= 2.0
        speed = (filtered.safe_command.vx ** 2 + filtered.safe_command.vy ** 2) ** 0.5
        self.assertAlmostEqual(speed, 2.0, places=4)
        self.assertEqual(filtered.safe_command.omega, 1.0)
        self.assertEqual(filtered.safe_command.force, 40.0)

    def test_barrier_breach_emergency_halt(self):
        crash_imminent_cmd = ActuatorCommand(
            target_id="robot_01",
            vx=1.5,
            vy=1.0,
            omega=0.0,
            force=10.0,
            current_position=(0.0, 0.0),
            nearest_obstacle_distance=0.2,  # < 0.5m barrier
        )
        filtered = self.cbf.filter_action(crash_imminent_cmd)

        self.assertTrue(filtered.was_modified)
        self.assertLess(filtered.barrier_value, 0.0)
        # Safe command must halt immediately
        self.assertEqual(filtered.safe_command.vx, 0.0)
        self.assertEqual(filtered.safe_command.vy, 0.0)

    def test_aegis_attestation_hash_chain_integrity(self):
        bridge = AegisAttestationBridge(agent_id="test_agent")

        r1 = bridge.create_receipt("tool_exec", {"cmd": "ls"}, {"files": ["a.txt"]})
        r2 = bridge.create_receipt("memory_update", {"key": "x"}, {"stored": True})

        self.assertTrue(bridge.verify_chain_integrity())
        self.assertEqual(r2.previous_receipt_hash, r1.receipt_hash)

        # Tampering with r1 output hash should break verification
        r1.output_hash = "tampered_hash"
        self.assertFalse(bridge.verify_chain_integrity())


if __name__ == "__main__":
    unittest.main()
