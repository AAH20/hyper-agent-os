"""
Real-Time Voice Streaming Pipeline.

Handles low-latency streaming audio buffers, energy-based Voice Activity
Detection (VAD), full-duplex turn-taking, and barge-in / interruption handling.
"""

from __future__ import annotations
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class AudioChunk:
    """A streaming chunk of raw audio PCM data."""
    pcm_bytes: bytes
    sample_rate: int = 16000
    timestamp: float = field(default_factory=time.time)

    @property
    def rms_energy(self) -> float:
        """Compute Root Mean Square (RMS) energy of 16-bit PCM audio samples."""
        if not self.pcm_bytes:
            return 0.0
        # Assume 16-bit signed integer PCM
        count = len(self.pcm_bytes) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", self.pcm_bytes[:count * 2])
        sum_sq = sum(s * s for s in shorts)
        return math.sqrt(sum_sq / count)


class VoiceActivityDetector:
    """
    Real-time energy-based VAD for speech segment detection.
    """

    def __init__(self, energy_threshold: float = 500.0, silence_chunks_threshold: int = 3):
        self.energy_threshold = energy_threshold
        self.silence_chunks_threshold = silence_chunks_threshold
        self.is_speaking = False
        self._consecutive_silence = 0

    def process_chunk(self, chunk: AudioChunk) -> str:
        """
        Process incoming audio chunk and return transition state:
        'SILENCE', 'SPEECH_STARTED', 'SPEECH_ONGOING', 'SPEECH_ENDED'.
        """
        energy = chunk.rms_energy
        if energy >= self.energy_threshold:
            self._consecutive_silence = 0
            if not self.is_speaking:
                self.is_speaking = True
                return "SPEECH_STARTED"
            return "SPEECH_ONGOING"
        else:
            if self.is_speaking:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self.silence_chunks_threshold:
                    self.is_speaking = False
                    self._consecutive_silence = 0
                    return "SPEECH_ENDED"
                return "SPEECH_ONGOING"
            return "SILENCE"


@dataclass
class ConversationTurn:
    """An interaction turn in a full-duplex conversational session."""
    speaker: str  # "user" or "agent"
    text: str
    start_time: float
    end_time: float
    interrupted: bool = False


class VoiceStreamPipeline:
    """
    Full-duplex conversational voice pipeline supporting barge-in interruptions.
    """

    def __init__(
        self,
        vad_threshold: float = 500.0,
        on_user_speech_started: Optional[Callable[[], None]] = None,
        on_user_speech_ended: Optional[Callable[[bytes], None]] = None,
    ):
        self.vad = VoiceActivityDetector(energy_threshold=vad_threshold)
        self.on_user_speech_started = on_user_speech_started
        self.on_user_speech_ended = on_user_speech_ended
        self.agent_is_speaking: bool = False
        self._audio_buffer: bytearray = bytearray()
        self.turns: List[ConversationTurn] = []

    def ingest_chunk(self, chunk: AudioChunk) -> str:
        """
        Ingest client mic stream. If the agent is speaking and user begins talking,
        triggers an immediate barge-in interruption.
        """
        state = self.vad.process_chunk(chunk)

        if state == "SPEECH_STARTED":
            if self.agent_is_speaking:
                # Barge-in: user interrupted the agent!
                self.agent_is_speaking = False
                if self.turns and self.turns[-1].speaker == "agent":
                    self.turns[-1].interrupted = True

            self._audio_buffer.clear()
            self._audio_buffer.extend(chunk.pcm_bytes)
            if self.on_user_speech_started:
                self.on_user_speech_started()

        elif state == "SPEECH_ONGOING":
            self._audio_buffer.extend(chunk.pcm_bytes)

        elif state == "SPEECH_ENDED":
            self._audio_buffer.extend(chunk.pcm_bytes)
            complete_audio = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            if self.on_user_speech_ended:
                self.on_user_speech_ended(complete_audio)

        return state
