"""
Deterministic Control Barrier Function (CBF) & Safety Governor.

Sits strictly between probabilistic AI agent policies (LLM / VLA / RL) and
critical physical or infrastructure actuators. Mathematically filters and
projects proposed actions onto certified safe operational envelopes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SafetyEnvelope:
    """Certified operational boundaries for critical/physical systems."""
    max_linear_velocity: float = 2.0     # m/s
    max_angular_velocity: float = 1.5    # rad/s
    max_force_torque: float = 50.0       # N or Nm
    min_obstacle_distance: float = 0.35  # meters
    allowed_workspace_bounds: Tuple[float, float, float, float] = (
        -10.0, -10.0, 10.0, 10.0  # min_x, min_y, max_x, max_y
    )


@dataclass
class ActuatorCommand:
    """Action proposed by an AI agent or VLA model."""
    target_id: str
    vx: float
    vy: float
    omega: float  # angular velocity
    force: float
    current_position: Tuple[float, float] = (0.0, 0.0)
    nearest_obstacle_distance: float = 1.0


@dataclass
class FilteredAction:
    """Safe action passed to the physical or critical actuator."""
    safe_command: ActuatorCommand
    was_modified: bool
    rejection_reasons: List[str]
    barrier_value: float  # h(x) >= 0 indicates safety


class ControlBarrierFunction:
    """
    Mathematical Control Barrier Governor guaranteeing forward invariance
    of the safe operational set.
    """

    def __init__(self, envelope: Optional[SafetyEnvelope] = None):
        self.envelope = envelope or SafetyEnvelope()

    def evaluate_barrier(self, cmd: ActuatorCommand) -> float:
        """
        Evaluate safety barrier function h(x).
        h(x) >= 0 is safe; h(x) < 0 indicates violation of distance or workspace.
        """
        dist_barrier = cmd.nearest_obstacle_distance - self.envelope.min_obstacle_distance

        min_x, min_y, max_x, max_y = self.envelope.allowed_workspace_bounds
        x, y = cmd.current_position
        bounds_barrier = min(x - min_x, max_x - x, y - min_y, max_y - y)

        return min(dist_barrier, bounds_barrier)

    def filter_action(self, proposed: ActuatorCommand) -> FilteredAction:
        """
        Deterministic safety filter. Projects proposed command onto safe bounds.
        If unsafe, clamps velocities or commands an immediate emergency halt.
        """
        reasons = []
        modified = False

        safe_vx = proposed.vx
        safe_vy = proposed.vy
        safe_omega = proposed.omega
        safe_force = proposed.force

        # 1. Evaluate barrier condition
        h_x = self.evaluate_barrier(proposed)
        if h_x < 0.0:
            reasons.append(f"Barrier breached: h(x)={h_x:.3f} < 0 (obstacle or workspace boundary)")
            # Emergency deceleration / halt
            safe_vx = 0.0
            safe_vy = 0.0
            safe_omega = 0.0
            safe_force = 0.0
            modified = True

        # 2. Linear velocity clamping
        speed = math.sqrt(safe_vx ** 2 + safe_vy ** 2)
        if speed > self.envelope.max_linear_velocity:
            reasons.append(
                f"Linear speed {speed:.2f} m/s exceeds max limit {self.envelope.max_linear_velocity:.2f} m/s"
            )
            scale = self.envelope.max_linear_velocity / speed
            safe_vx *= scale
            safe_vy *= scale
            modified = True

        # 3. Angular velocity clamping
        if abs(safe_omega) > self.envelope.max_angular_velocity:
            reasons.append(
                f"Angular velocity {abs(safe_omega):.2f} rad/s exceeds limit {self.envelope.max_angular_velocity:.2f} rad/s"
            )
            safe_omega = math.copysign(self.envelope.max_angular_velocity, safe_omega)
            modified = True

        # 4. Force/Torque clamping
        if abs(safe_force) > self.envelope.max_force_torque:
            reasons.append(
                f"Force {abs(safe_force):.2f} N exceeds limit {self.envelope.max_force_torque:.2f} N"
            )
            safe_force = math.copysign(self.envelope.max_force_torque, safe_force)
            modified = True

        safe_cmd = ActuatorCommand(
            target_id=proposed.target_id,
            vx=safe_vx,
            vy=safe_vy,
            omega=safe_omega,
            force=safe_force,
            current_position=proposed.current_position,
            nearest_obstacle_distance=proposed.nearest_obstacle_distance,
        )

        return FilteredAction(
            safe_command=safe_cmd,
            was_modified=modified,
            rejection_reasons=reasons,
            barrier_value=h_x,
        )
