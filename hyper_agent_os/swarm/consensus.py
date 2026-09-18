"""
Byzantine-Tolerant Multi-Agent Quorum Consensus.

Provides cryptographic attribution, vote validation, and quorum consensus
for swarm decision-making before committing critical mutations or tool calls.
"""

from __future__ import annotations
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class VoteEnvelope:
    """An attributable vote cast by an agent in the swarm."""
    voter_id: str
    proposal_id: str
    vote: bool  # True for approve, False for reject
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def sign(self, secret_key: str = "swarm_shared_key") -> str:
        """Compute cryptographic attribution signature over vote contents."""
        payload = f"{self.voter_id}:{self.proposal_id}:{self.vote}:{self.timestamp}"
        self.signature = hashlib.sha256(f"{payload}:{secret_key}".encode()).hexdigest()
        return self.signature

    def verify(self, secret_key: str = "swarm_shared_key") -> bool:
        """Verify the cryptographic signature of the vote envelope."""
        payload = f"{self.voter_id}:{self.proposal_id}:{self.vote}:{self.timestamp}"
        expected = hashlib.sha256(f"{payload}:{secret_key}".encode()).hexdigest()
        return self.signature == expected


@dataclass
class ConsensusDecision:
    """The outcome of a swarm consensus round."""
    proposal_id: str
    decided: bool
    approved: bool
    total_nodes: int
    quorum_threshold: int
    votes_for: int
    votes_against: int
    byzantine_nodes: List[str]
    audit_hash: str


class ByzantineQuorum:
    """
    Evaluates multi-agent voting rounds with Byzantine Fault Tolerance.
    Tolerates up to f < n/3 faulty/malicious nodes with a quorum of floor(2n/3) + 1.
    """

    def __init__(self, authorized_nodes: Set[str], secret_key: str = "swarm_shared_key"):
        self.authorized_nodes = authorized_nodes
        self.secret_key = secret_key

    @property
    def quorum_threshold(self) -> int:
        n = len(self.authorized_nodes)
        return math.floor(2 * n / 3) + 1

    def evaluate_proposal(
        self, proposal_id: str, votes: List[VoteEnvelope]
    ) -> ConsensusDecision:
        """
        Tally attributable votes, verify signatures, detect Byzantine violations,
        and determine if quorum consensus has been achieved.
        """
        seen_voters: Set[str] = set()
        byzantine_nodes: List[str] = []
        votes_for = 0
        votes_against = 0

        for vote in votes:
            # 1. Check if voter is an authorized swarm node
            if vote.voter_id not in self.authorized_nodes:
                byzantine_nodes.append(f"unauthorized:{vote.voter_id}")
                continue

            # 2. Check for double-voting (equivocation)
            if vote.voter_id in seen_voters:
                byzantine_nodes.append(f"equivocation:{vote.voter_id}")
                continue
            seen_voters.add(vote.voter_id)

            # 3. Check cryptographic signature validity
            if not vote.verify(self.secret_key):
                byzantine_nodes.append(f"invalid_sig:{vote.voter_id}")
                continue

            # 4. Tally verified votes
            if vote.vote is True:
                votes_for += 1
            else:
                votes_against += 1

        n = len(self.authorized_nodes)
        thresh = self.quorum_threshold

        decided = False
        approved = False

        if votes_for >= thresh:
            decided = True
            approved = True
        elif votes_against >= thresh or (n - len(byzantine_nodes) < thresh):
            decided = True
            approved = False

        # Produce deterministic SHA-256 audit digest of the round
        tally_summary = f"{proposal_id}:{approved}:{votes_for}:{votes_against}"
        audit_hash = hashlib.sha256(tally_summary.encode()).hexdigest()

        return ConsensusDecision(
            proposal_id=proposal_id,
            decided=decided,
            approved=approved,
            total_nodes=n,
            quorum_threshold=thresh,
            votes_for=votes_for,
            votes_against=votes_against,
            byzantine_nodes=byzantine_nodes,
            audit_hash=audit_hash,
        )
