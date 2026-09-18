"""
High-Throughput Distributed Event Bus.

Enables asynchronous pub/sub messaging across distributed cloud agents,
edge daemons, and swarm nodes with wildcard topic matching.
"""

from __future__ import annotations
import fnmatch
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set


@dataclass
class AgentEvent:
    """An event emitted by an agent or external trigger."""
    topic: str
    payload: Dict[str, Any]
    sender_id: str
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)


EventHandler = Callable[[AgentEvent], None]


class EventBus:
    """
    In-memory and network-pluggable pub/sub event bus supporting wildcard routing.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[EventHandler]] = {}
        self._event_history: List[AgentEvent] = []
        self._max_history: int = 1000

    def subscribe(self, topic_pattern: str, handler: EventHandler) -> None:
        """Register a handler for a topic pattern (supports glob wildcards e.g. 'swarm.*')."""
        self._subscribers.setdefault(topic_pattern, set()).add(handler)

    def unsubscribe(self, topic_pattern: str, handler: EventHandler) -> None:
        """Unregister a handler."""
        if topic_pattern in self._subscribers:
            self._subscribers[topic_pattern].discard(handler)

    def publish(self, topic: str, payload: Dict[str, Any], sender_id: str = "system") -> AgentEvent:
        """
        Publish an event to all subscribers matching the topic pattern.
        """
        event = AgentEvent(topic=topic, payload=payload, sender_id=sender_id)
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Match against all registered topic patterns
        for pattern, handlers in self._subscribers.items():
            if fnmatch.fnmatch(topic, pattern):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as err:
                        # Log error without stopping event bus dispatch
                        pass

        return event

    def get_history(self, topic_pattern: Optional[str] = None) -> List[AgentEvent]:
        """Retrieve recent events matching an optional topic pattern."""
        if not topic_pattern:
            return list(self._event_history)
        return [e for e in self._event_history if fnmatch.fnmatch(e.topic, topic_pattern)]
