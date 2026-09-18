"""
Unified Command-Line Interface for Hyper-Agent OS.

Provides commands to inspect memory, execute swarm workflows, test 24/7
durable daemons, verify streaming pipelines, test safety governors, and
run benchmark suites.
"""

from __future__ import annotations
import argparse
import json
import sys
import time

from hyper_agent_os.memory import (
    BiTemporalKnowledgeGraph,
    TemporalQuery,
    SpatialMultimodalIndex,
    BoundingBox3D,
    UnifiedRetriever,
)
from hyper_agent_os.swarm import (
    StateGraph,
    SwarmAgent,
    SwarmTask,
    SwarmCrew,
    ByzantineQuorum,
    VoteEnvelope,
)
from hyper_agent_os.daemon import (
    DurableEngine,
    DurableWorkflow,
    EventBus,
    HealthMonitor,
)
from hyper_agent_os.streaming import (
    VoiceStreamPipeline,
    AudioChunk,
    VideoStreamPipeline,
    VideoFrame,
)
from hyper_agent_os.safety import (
    ControlBarrierFunction,
    SafetyEnvelope,
    ActuatorCommand,
    AegisAttestationBridge,
)
from hyper_agent_os.benchmarks import (
    SWEBenchTask,
    SWEBenchRunner,
    MLEBenchCompetition,
    MLEBenchRunner,
    ARCTask,
    ARCPair,
    ARCEvaluator,
)


def cmd_memory(args):
    print("=== [Subsystem 1] Bi-Temporal Super-Memory Substrate ===")
    kg = BiTemporalKnowledgeGraph()

    t0 = time.time() - 3600  # 1 hour ago
    t1 = time.time() - 1800  # 30 mins ago
    now = time.time()

    print("[1] Recording historical and current facts...")
    kg.assert_fact("AgentCore", "depends_on", "LangGraph", valid_from=t0)
    kg.assert_fact("AgentCore", "integrates_with", "CrewAI", valid_from=t1)
    kg.assert_fact("AgentCore", "governed_by", "ControlBarrierFunction", valid_from=now)

    print("[2] Querying facts valid 45 minutes ago (t0 to t1)...")
    facts_45m = kg.query(TemporalQuery(as_of_valid_time=time.time() - 2700))
    for f in facts_45m:
        print(f"  - {f.subject} {f.predicate} {f.target} (valid since {int(now - f.valid_from)}s ago)")

    print("[3] Adding 3D Spatial Entity...")
    spatial = SpatialMultimodalIndex()
    spatial.register_spatial_object(
        name="CriticalActuatorArm",
        category="robotics",
        bbox=BoundingBox3D(0.0, 0.0, 0.0, 1.2, 0.8, 1.5),
        embedding=[0.85, 0.12, 0.44],
    )

    retriever = UnifiedRetriever(kg, spatial)
    res = retriever.retrieve("What does AgentCore integrate with?", subject="AgentCore", query_vector=[0.8, 0.1, 0.4])
    print(f"[4] Retrieved {len(res.facts)} facts, {len(res.citations)} citations in {res.execution_time_ms:.2f}ms")
    print(res.synthesized_context)


def cmd_swarm(args):
    print("=== [Subsystem 2] Cognitive Swarm & LangGraph Orchestration ===")
    graph = StateGraph()

    def planner_node(state):
        print("  -> Planner Agent: Synthesizing plan...")
        return {"plan": "Analyze system requirements", "status": "PLANNED"}

    def executor_node(state):
        print("  -> Executor Agent: Executing state mutations...")
        return {"result": "Changes applied successfully", "status": "EXECUTED"}

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", StateGraph.END)

    compiled = graph.compile()
    final_state = compiled.invoke({"input": "Optimize agent loop"})
    print(f"[1] StateGraph finished in {final_state.iteration} iterations. Final state values:")
    print(f"    {final_state.values}")

    print("[2] Byzantine Consensus Evaluation on Execution:")
    quorum = ByzantineQuorum(authorized_nodes={"node_1", "node_2", "node_3", "node_4"})
    v1 = VoteEnvelope("node_1", "prop_1", True)
    v2 = VoteEnvelope("node_2", "prop_1", True)
    v3 = VoteEnvelope("node_3", "prop_1", True)
    v4 = VoteEnvelope("node_4", "prop_1", False)
    for v in [v1, v2, v3, v4]:
        v.sign()

    decision = quorum.evaluate_proposal("prop_1", [v1, v2, v3, v4])
    print(f"    Quorum threshold: {decision.quorum_threshold} of {decision.total_nodes} nodes.")
    print(f"    Approved: {decision.approved} (Votes For: {decision.votes_for}, Against: {decision.votes_against})")
    print(f"    Audit Hash: {decision.audit_hash[:16]}...")


def cmd_daemon(args):
    print("=== [Subsystem 3] 24/7 Durable Cloud Daemon ===")
    engine = DurableEngine()
    workflow = DurableWorkflow("critical-backup-daemon", engine)

    call_counts = {"step_a": 0, "step_b": 0}

    def step_a(state):
        call_counts["step_a"] += 1
        return {"snapshot_id": "snap_9942", "timestamp": time.time()}

    def step_b(state):
        call_counts["step_b"] += 1
        return {"uploaded_to_s3": True}

    workflow.add_step("create_snapshot", step_a)
    workflow.add_step("upload_backup", step_b)

    # First run
    res1 = workflow.execute({"database": "primary_db"})
    wf_id = res1["workflow_id"]
    print(f"[1] Completed workflow '{wf_id}' initial run.")

    # Replay run to demonstrate idempotency and recovery
    print("[2] Simulating daemon restart and replay...")
    res2 = workflow.execute({"database": "primary_db"}, workflow_id=wf_id)
    print(f"    Step A invocations: {call_counts['step_a']} (Cached from replay, did not re-run!)")
    print(f"    Workflow status: {res2['status']}")


def cmd_streaming(args):
    print("=== [Subsystem 4] Real-Time Multimodal Streaming Engine ===")
    pipeline = VoiceStreamPipeline(vad_threshold=100.0)
    pipeline.agent_is_speaking = True

    # Simulate quiet audio chunk
    silent_pcm = b"\x00\x00" * 160
    c1 = AudioChunk(pcm_bytes=silent_pcm)
    state1 = pipeline.ingest_chunk(c1)
    print(f"  Chunk 1 state: {state1} (Agent speaking: {pipeline.agent_is_speaking})")

    # Simulate user speaking loudly (triggering barge-in)
    loud_pcm = b"\x50\x20" * 160
    c2 = AudioChunk(pcm_bytes=loud_pcm)
    state2 = pipeline.ingest_chunk(c2)
    print(f"  Chunk 2 state: {state2} -> Barge-in triggered! (Agent speaking: {pipeline.agent_is_speaking})")

    print("[2] Video Keyframe Sampler:")
    v_pipe = VideoStreamPipeline(min_interval_seconds=0.1)
    f1 = VideoFrame(1, 100.0, 640, 480, b"\x10" * 1024)
    f2 = VideoFrame(2, 100.05, 640, 480, b"\x10" * 1024)  # Identical & too fast
    f3 = VideoFrame(3, 100.2, 640, 480, b"\x80" * 1024)   # Different & after interval

    print(f"  Frame 1 sampled: {v_pipe.ingest_frame(f1)}")
    print(f"  Frame 2 sampled: {v_pipe.ingest_frame(f2)} (discarded redundant frame)")
    print(f"  Frame 3 sampled: {v_pipe.ingest_frame(f3)} (sampled new keyframe)")


def cmd_safety(args):
    print("=== [Subsystem 5] Critical Systems & VLA Safety Governor ===")
    cbf = ControlBarrierFunction(SafetyEnvelope(max_linear_velocity=1.5, min_obstacle_distance=0.5))

    # Dangerous command proposed by unconstrained AI
    dangerous_cmd = ActuatorCommand(
        target_id="actuator_alpha",
        vx=4.5,
        vy=3.0,
        omega=0.5,
        force=15.0,
        current_position=(0.0, 0.0),
        nearest_obstacle_distance=0.2,  # Breaches 0.5m barrier!
    )

    filtered = cbf.filter_action(dangerous_cmd)
    print(f"  Proposed Speed: {dangerous_cmd.vx} m/s, Obstacle Distance: {dangerous_cmd.nearest_obstacle_distance} m")
    print(f"  Governor Modified Action: {filtered.was_modified}")
    print(f"  Filtered Safe Speed: vx={filtered.safe_command.vx}, vy={filtered.safe_command.vy}")
    print(f"  Rejection Reasons: {filtered.rejection_reasons}")

    # Cryptographic attestation receipt
    bridge = AegisAttestationBridge()
    receipt = bridge.create_receipt(
        action_name="actuator_safety_filter",
        input_data={"vx": dangerous_cmd.vx},
        output_data={"safe_vx": filtered.safe_command.vx},
    )
    print(f"  Attestation Receipt: {receipt.receipt_id} [SHA256: {receipt.receipt_hash[:16]}...]")
    print(f"  Chain integrity: {bridge.verify_chain_integrity()}")


def cmd_benchmark(args):
    print("=== [Subsystem 6] Benchmark Evaluation Suite ===")

    # 1. SWE-bench Verified Mock
    task = SWEBenchTask(
        instance_id="sympy__sympy-13480",
        repo="sympy/sympy",
        base_commit="a94ef2",
        problem_statement="Hyperbolic tangent evaluation error",
        fail_to_pass_tests=["test_tanh_eval"],
        pass_to_pass_tests=["test_basic"],
    )
    swe_runner = SWEBenchRunner([task])
    swe_summary = swe_runner.run_suite(
        lambda t: "diff --git a/sympy/functions.py b/sympy/functions.py\n--- a/sympy/functions.py\n+++ b/sympy/functions.py\n@@ -1,3 +1,3 @@\n def test_tanh_eval():\n+    return True\n"
    )
    print(f"[1] SWE-bench Verified: {swe_summary['resolved']}/{swe_summary['total_tasks']} resolved ({swe_summary['resolve_rate_percent']:.1f}%)")

    # 2. MLE-bench (Kaggle) Mock
    comp = MLEBenchCompetition(
        competition_id="titanic-ml",
        title="Titanic: Machine Learning from Disaster",
        metric="accuracy",
        bronze_threshold=0.78,
        silver_threshold=0.82,
        gold_threshold=0.85,
    )
    mle_runner = MLEBenchRunner([comp])
    mle_summary = mle_runner.run_suite(lambda c: 0.835)
    print(f"[2] MLE-bench (Kaggle): {mle_summary['any_medal_count']}/{mle_summary['total_competitions']} Medals won ({mle_summary['medal_rate_percent']:.1f}%) -> {mle_summary['details'][0]['medal']} Medal")

    # 3. ARC Reasoning Mock
    arc_task = ARCTask(
        task_id="arc_sample_01",
        train_pairs=[ARCPair([[1, 0], [0, 1]], [[0, 1], [1, 0]])],
        test_pairs=[ARCPair([[2, 0], [0, 2]], [[0, 2], [2, 0]])],
    )
    arc_eval = ARCEvaluator([arc_task])
    arc_summary = arc_eval.run_suite(lambda t: [[0, 2], [2, 0]])
    print(f"[3] ARC Prize: {arc_summary['solved']}/{arc_summary['total_tasks']} puzzles solved ({arc_summary['accuracy_percent']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Hyper-Agent OS & Swarm Substrate CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subsystem commands")

    subparsers.add_parser("memory", help="Demonstrate Bi-Temporal Super-Memory")
    subparsers.add_parser("swarm", help="Demonstrate LangGraph Swarm & Quorum")
    subparsers.add_parser("daemon", help="Demonstrate 24/7 Durable Cloud Daemon")
    subparsers.add_parser("streaming", help="Demonstrate Multimodal Voice/Video Streaming")
    subparsers.add_parser("safety", help="Demonstrate VLA Critical Safety Governor")
    subparsers.add_parser("benchmark", help="Demonstrate Benchmark Evaluation Suite")
    subparsers.add_parser("all", help="Execute all subsystem demonstrations")

    args = parser.parse_args()

    if args.command == "memory":
        cmd_memory(args)
    elif args.command == "swarm":
        cmd_swarm(args)
    elif args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "streaming":
        cmd_streaming(args)
    elif args.command == "safety":
        cmd_safety(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "all" or args.command is None:
        cmd_memory(args)
        print()
        cmd_swarm(args)
        print()
        cmd_daemon(args)
        print()
        cmd_streaming(args)
        print()
        cmd_safety(args)
        print()
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
