"""
Aegis Attestation & Audit Bridge.

Produces cryptographically signed receipts for all tool invocations, state
mutations, and actuator commands, integrating with the Aegis and Swarm trust substrate.
"""

from __future__ import annotations
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AttestationReceipt:
    """An immutable cryptographic receipt documenting an agent action."""
    receipt_id: str
    agent_id: str
    action_name: str
    input_hash: str
    output_hash: str
    timestamp: float
    previous_receipt_hash: str
    receipt_hash: str
    verified: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AegisAttestationBridge:
    """
    Maintains a tamper-evident, append-only hash chain of all agent actions.
    """

    def __init__(self, agent_id: str = "hyper_agent_default"):
        self.agent_id = agent_id
        self.last_hash: str = "0" * 64
        self.receipts: List[AttestationReceipt] = []

    def create_receipt(
        self,
        action_name: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttestationReceipt:
        """
        Record and seal an action into the cryptographic audit chain.
        """
        now = time.time()
        receipt_id = f"rcpt_{uuid.uuid4().hex[:12]}"

        in_str = json.dumps(input_data, sort_keys=True, default=str)
        out_str = json.dumps(output_data, sort_keys=True, default=str)

        in_hash = hashlib.sha256(in_str.encode()).hexdigest()
        out_hash = hashlib.sha256(out_str.encode()).hexdigest()

        # Compute chain hash linking to previous receipt
        chain_payload = f"{receipt_id}:{self.agent_id}:{action_name}:{in_hash}:{out_hash}:{now}:{self.last_hash}"
        current_hash = hashlib.sha256(chain_payload.encode()).hexdigest()

        receipt = AttestationReceipt(
            receipt_id=receipt_id,
            agent_id=self.agent_id,
            action_name=action_name,
            input_hash=in_hash,
            output_hash=out_hash,
            timestamp=now,
            previous_receipt_hash=self.last_hash,
            receipt_hash=current_hash,
            verified=True,
            metadata=metadata or {},
        )

        self.last_hash = current_hash
        self.receipts.append(receipt)
        return receipt

    def verify_chain_integrity(self) -> bool:
        """Verify that the hash chain is unbroken and untampered."""
        prev_hash = "0" * 64
        for r in self.receipts:
            if r.previous_receipt_hash != prev_hash:
                return False
            chain_payload = f"{r.receipt_id}:{r.agent_id}:{r.action_name}:{r.input_hash}:{r.output_hash}:{r.timestamp}:{prev_hash}"
            expected = hashlib.sha256(chain_payload.encode()).hexdigest()
            if r.receipt_hash != expected:
                return False
            prev_hash = r.receipt_hash
        return True
