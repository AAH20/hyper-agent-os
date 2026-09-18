"""Unit tests for Multimodal Voice and Video Streaming Pipelines."""

import unittest
from hyper_agent_os.streaming import (
    VoiceStreamPipeline,
    AudioChunk,
    VoiceActivityDetector,
    VideoStreamPipeline,
    VideoFrame,
)


class TestStreamingSubsystem(unittest.TestCase):

    def test_vad_silence_vs_speech(self):
        vad = VoiceActivityDetector(energy_threshold=200.0, silence_chunks_threshold=2)

        # Silent chunk
        silent = AudioChunk(pcm_bytes=b"\x00\x00" * 80)
        self.assertEqual(vad.process_chunk(silent), "SILENCE")

        # Loud speech chunk (signed 16-bit values ~ 2000)
        loud = AudioChunk(pcm_bytes=b"\xd0\x07" * 80)
        self.assertEqual(vad.process_chunk(loud), "SPEECH_STARTED")
        self.assertEqual(vad.process_chunk(loud), "SPEECH_ONGOING")

        # Return to silence
        self.assertEqual(vad.process_chunk(silent), "SPEECH_ONGOING")
        self.assertEqual(vad.process_chunk(silent), "SPEECH_ENDED")

    def test_voice_barge_in_interruption(self):
        pipeline = VoiceStreamPipeline(vad_threshold=150.0)
        pipeline.agent_is_speaking = True

        loud_chunk = AudioChunk(pcm_bytes=b"\x00\x10" * 100)
        state = pipeline.ingest_chunk(loud_chunk)

        self.assertEqual(state, "SPEECH_STARTED")
        # Barge-in should immediately stop agent speaking
        self.assertFalse(pipeline.agent_is_speaking)

    def test_video_keyframe_sampler(self):
        emitted_frames = []
        pipeline = VideoStreamPipeline(
            min_interval_seconds=0.1,
            on_keyframe=lambda f: emitted_frames.append(f.frame_id),
        )

        # Frame 1 at t=1.0 -> should emit
        f1 = VideoFrame(1, 1.0, 320, 240, b"AAAA" * 64)
        self.assertTrue(pipeline.ingest_frame(f1))

        # Frame 2 at t=1.02 -> too soon (< 0.1s)
        f2 = VideoFrame(2, 1.02, 320, 240, b"AAAA" * 64)
        self.assertFalse(pipeline.ingest_frame(f2))

        # Frame 3 at t=1.2 -> identical bytes -> no visual change -> should discard
        f3 = VideoFrame(3, 1.2, 320, 240, b"AAAA" * 64)
        self.assertFalse(pipeline.ingest_frame(f3))

        # Frame 4 at t=1.35 -> distinct bytes -> should emit
        f4 = VideoFrame(4, 1.35, 320, 240, b"ZZZZ" * 64)
        self.assertTrue(pipeline.ingest_frame(f4))

        self.assertEqual(emitted_frames, [1, 4])


if __name__ == "__main__":
    unittest.main()
