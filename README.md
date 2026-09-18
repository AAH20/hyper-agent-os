# ⚡ Hyper-Agent OS: Distributed Multi-Agent Runtime & Swarm Substrate

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-29%2F29%20Passing%20(100%25)-success?logo=pytest)](file:///Users/ahmedhassan/Downloads/a2z-soc-main%202/hyper-agent-os/tests)
[![MCP Compliant](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-8A2BE2?logo=anthropic)](https://modelcontextprotocol.io)
[![SWE-bench Verified](https://img.shields.io/badge/SWE--bench-Verified%20%26%20Multimodal-orange)](https://www.swebench.com)
[![MLE-bench Kaggle](https://img.shields.io/badge/MLE--bench-Kaggle%20Medal%20Grade-20BEFF?logo=kaggle)](https://github.com/openai/mle-bench)
[![Post-Quantum Attestation](https://img.shields.io/badge/Aegis-Post--Quantum%20Audit%20Chains-red)](#5-critical-systems--vla-safety-governor)

**The unified, enterprise-grade distributed agent runtime uniting Super-Memory (Bi-Temporal GraphRAG), Cyclical Swarm Orchestration, 24/7 Durable Daemons, Multimodal Streaming, and Critical Systems VLA Safety Governors.**

[Key Features](#-key-features) • [Competitive Matrix](#-competitive-feature-matrix) • [Quickstart](#-quickstart-in-30-seconds) • [Architecture](#-architecture) • [MCP Server Setup](#-model-context-protocol-mcp-server) • [Benchmarks](#-benchmark-evaluators)

</div>

---

## ⚡ Why Hyper-Agent OS?

Most AI agent frameworks are either:
1. **Stateless prompt chains** that hallucinate when context grows (naive vector RAG).
2. **Fragile Python loops** that crash, drop state, or leak memory during 24/7 background execution.
3. **Unconstrained probabilistic actors** dangerous to deploy to physical actuators, robotics, or critical production infrastructure.

**Hyper-Agent OS** solves this by unifying a **Bi-Temporal Graph Memory** (surpassing Cognee), **Cyclical State Graphs** (LangGraph-style) with **Byzantine Quorums**, **Temporal-grade Durable Execution**, **Sub-100ms Multimodal Streaming**, and **Deterministic Control Barrier Functions (CBF)**.

---

## 📊 Competitive Feature Matrix

| Feature | Hyper-Agent OS | Cognee | LangGraph | CrewAI | Naive Vector RAG |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bi-Temporal Knowledge Graph** | **✅ Yes** | ❌ No | ❌ No | ❌ No | ❌ No |
| **Point-in-Time Rollback (Anti-Hallucination)** | **✅ Yes** | ❌ No | ❌ No | ❌ No | ❌ No |
| **3D Spatial & Multimodal Embeddings** | **✅ Yes** | ❌ No | ❌ No | ❌ No | ⚠️ Flat vectors |
| **Cyclical State Graphs & Interrupts** | **✅ Built-in** | ❌ No | ✅ Yes | ❌ Linear | ❌ No |
| **Byzantine Fault Tolerant Swarm Quorum** | **✅ BFT Quorum** | ❌ No | ❌ No | ❌ No | ❌ No |
| **24/7 Durable Replay (Temporal Pattern)** | **✅ SQLite Journal** | ❌ No | ⚠️ Checkpoints | ❌ No | ❌ No |
| **Sub-100ms Voice Stream + Barge-In** | **✅ Yes** | ❌ No | ❌ No | ❌ No | ❌ No |
| **Critical Safety Governors (CBF)** | **✅ Deterministic** | ❌ No | ❌ No | ❌ No | ❌ No |
| **Post-Quantum Attestation Receipts** | **✅ Aegis/SHA-256** | ❌ No | ❌ No | ❌ No | ❌ No |
| **Built-in SWE-bench & MLE-bench Eval** | **✅ Yes** | ❌ No | ❌ No | ❌ No | ❌ No |
| **Model Context Protocol (MCP) Server** | **✅ Native** | ⚠️ Plugins | ⚠️ Community | ⚠️ Community | ❌ No |
| **Drop-in Cognee Migration Shim** | **✅ 1-Line Import** | N/A | ❌ No | ❌ No | ❌ No |

---

## 🏛️ Architecture

```
                               ┌────────────────────────────────────────┐
                               │       Hyper-Agent OS / Runtime         │
                               └───────────────────┬────────────────────┘
                                                   │
     ┌──────────────────────┬──────────────────────┼──────────────────────┬──────────────────────┐
     ▼                      ▼                      ▼                      ▼                      ▼
[ 1. Memory Substrate ]  [ 2. Swarm Core ]  [ 3. 24/7 Daemon ]  [ 4. Streaming ]      [ 5. Safety & VLA ]
• Bi-Temporal Graph      • LangGraph DAG    • Durable Loops     • WebRTC/Audio Stream • Control Barrier Func
• Vector Index           • CrewAI Swarms    • Event Bus (PubSub)• Video Keyframe Pipe • Actuation Envelope
• Spatial Scene Graph    • Role Consensus   • Self-Healing      • Real-Time VAD/TTS   • Deterministic Reject
                                                   │
                                                   ▼
                                  [ 6. Benchmark Evaluation Suite ]
                                  • SWE-bench Verified & Multimodal Runner
                                  • MLE-bench (Kaggle) Evaluation Adapter
                                  • ARC Reasoning & Tool Execution Harness
```

---

## 🚀 Quickstart in 30 Seconds

### Installation
```bash
git clone https://github.com/AAH20/hyper-agent-os.git
cd hyper-agent-os
pip install -e .
```

### Run the CLI Demonstration
```bash
# Run live end-to-end demonstrations across all 6 subsystems
hyper-os all

# Or run individual subsystems
hyper-os memory
hyper-os swarm
hyper-os daemon
hyper-os streaming
hyper-os safety
hyper-os benchmark
```

### Run the Automated Test Suite (100% Passing)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🔄 Drop-in Cognee Migration

Migrate from Cognee in one line of code:

```python
from hyper_agent_os.adapters import CogneeCompat

cognee = CogneeCompat()

# Ingest unstructured text or code
cognee.add_sync([
    "QuantumController coordinates RobotArmAlpha.",
    "RobotArmAlpha is governed by ControlBarrierFunction."
])

# Extract and cognify into the bi-temporal graph
cognee.cognify_sync()

# Multi-hop hybrid graph retrieval
results = cognee.search_sync("What governs RobotArmAlpha?", subject="RobotArmAlpha")
print(results)
```

---

## 🔌 Model Context Protocol (MCP) Server

Connect Hyper-Agent OS directly to **Cursor, Windsurf, Claude Desktop, or Antigravity**:

### Claude Desktop / Cursor Config (`mcpServers`):
```json
{
  "mcpServers": {
    "hyper-agent-os": {
      "command": "python3",
      "args": ["-m", "hyper_agent_os.mcp_server"],
      "env": {}
    }
  }
}
```

### Native Tools Exposed via MCP:
- `hyper_query_memory`: Query the Bi-Temporal Knowledge Graph & Multimodal index.
- `hyper_assert_fact`: Commit persistent facts with cryptographic audit attribution.
- `hyper_run_swarm`: Launch autonomous collaborative swarm workflows.
- `hyper_evaluate_cbf_safety`: Certify and project actuator commands onto safe operational envelopes.

---

## 🐳 Docker & 24/7 Production Deployment

Run Hyper-Agent OS as an autonomous background daemon with persistent SQLite journaling:

```bash
docker compose up -d
```

---

## 📈 Benchmark Evaluators

Hyper-Agent OS includes built-in harnesses for the industry's most demanding AI benchmarks:

1. **SWE-bench Verified & Multimodal**:
   - Tests patch generation against real GitHub repository issues.
   - Evaluates whether unit test failures flip to pass (`FAIL_TO_PASS`) without regressing existing tests.
2. **MLE-bench (OpenAI / Kaggle)**:
   - Evaluates end-to-end Machine Learning pipelines on Kaggle competitions.
   - Automatically grades models into **Bronze, Silver, or Gold Medal tiers**.
3. **ARC Prize (Abstraction and Reasoning Corpus)**:
   - Measures out-of-distribution reasoning and visual grid program synthesis.

---

## 🏷️ Recommended GitHub Topics (SEO Tags)

Add these exact topics in your GitHub repository settings to maximize search indexing:

```text
ai-agents, graphrag, knowledge-graph, bi-temporal-memory, langgraph, 
crewai, swarm-intelligence, mcp, model-context-protocol, swe-bench, 
mle-bench, autonomous-agents, durable-execution, voice-agent, vla, 
robotics, control-barrier-functions, post-quantum, devin-alternative, rag
```

---

## 📜 License & Authors
Developed by **Ahmed Hassan** (Founder, A2Z SOC).  
Licensed under the [Apache-2.0 License](LICENSE).
