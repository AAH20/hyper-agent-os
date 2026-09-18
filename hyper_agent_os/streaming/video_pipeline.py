"""
Real-Time Video Streaming & Keyframe Sampling.

Ingests high-framerate visual streams, performs visual delta thresholding,
filters redundant frames, and emits semantic keyframes to VLM/VLA agents.
"""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class VideoFrame:
    """A single frame from a video stream."""
    frame_id: int
    timestamp: float
    width: int
    height: int
    raw_bytes: bytes
    metadata: dict = field(default_factory=dict)

    @property
    def frame_hash(self) -> str:
        return hashlib.md5(self.raw_bytes).hexdigest()


class KeyframeSampler:
    """
    Downsamples continuous video streams to retain only informative keyframes
    based on temporal pacing and visual delta change.
    """

    def __init__(
        self,
        min_interval_seconds: float = 0.5,
        delta_threshold: float = 0.05,
    ):
        self.min_interval_seconds = min_interval_seconds
        self.delta_threshold = delta_threshold
        self.last_keyframe: Optional[VideoFrame] = None
        self.last_keyframe_time: float = 0.0

    def compute_frame_delta(self, frame_a: VideoFrame, frame_b: VideoFrame) -> float:
        """Compute relative difference ratio between two frame byte arrays."""
        if len(frame_a.raw_bytes) != len(frame_b.raw_bytes):
            return 1.0
        if not frame_a.raw_bytes:
            return 0.0

        # Sample every 16th byte for fast real-time estimation
        sample_a = frame_a.raw_bytes[::16]
        sample_b = frame_b.raw_bytes[::16]
        diff_count = sum(1 for b_a, b_b in zip(sample_a, sample_b) if abs(b_a - b_b) > 20)
        return diff_count / max(1, len(sample_a))

    def should_sample(self, frame: VideoFrame) -> bool:
        """Determine if the current frame constitutes a keyframe."""
        if self.last_keyframe is None:
            self.last_keyframe = frame
            self.last_keyframe_time = frame.timestamp
            return True

        elapsed = frame.timestamp - self.last_keyframe_time
        if elapsed < self.min_interval_seconds:
            return False

        delta = self.compute_frame_delta(self.last_keyframe, frame)
        if delta >= self.delta_threshold:
            self.last_keyframe = frame
            self.last_keyframe_time = frame.timestamp
            return True

        return False


class VideoStreamPipeline:
    """
    Manages live video frame streams and dispatches sampled keyframes to observers.
    """

    def __init__(
        self,
        min_interval_seconds: float = 0.5,
        on_keyframe: Optional[Callable[[VideoFrame], None]] = None,
    ):
        self.sampler = KeyframeSampler(min_interval_seconds=min_interval_seconds)
        self.on_keyframe = on_keyframe
        self.total_frames_received: int = 0
        self.keyframes_emitted: int = 0

    def ingest_frame(self, frame: VideoFrame) -> bool:
        """Ingest a video frame and notify if sampled as a keyframe."""
        self.total_frames_received += 1
        is_key = self.sampler.should_sample(frame)
        if is_key:
            self.keyframes_emitted += 1
            if self.on_keyframe:
                self.on_keyframe(frame)
        return is_key
