"""
core/transcriber.py
Transcription using OpenAI Whisper.
Transcribes the FULL audio once with word-level timestamps, then maps
words to each diarized speaker segment by time overlap. This is far
faster than re-running Whisper per segment and avoids the "whole audio
text leaking into every segment" bug.
"""

import os
import whisper
import torch
from dotenv import load_dotenv

load_dotenv()


class VideoTranscriber:
    """Transcribe audio using OpenAI Whisper"""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📢 Using device: {self.device} for transcription")
        self.model = whisper.load_model(model_size, device=self.device)
        self._full_transcript_cache = {}  # audio_path -> word list

        os.makedirs("downloads/transcripts", exist_ok=True)

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """Transcribe entire audio file once, with word-level timestamps"""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            word_timestamps=True,
            verbose=False
        )
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"]
        }

    def _get_words(self, audio_path: str) -> list:
        """Transcribe once and cache word-level timestamps for this audio file"""
        if audio_path in self._full_transcript_cache:
            return self._full_transcript_cache[audio_path]

        print("📝 Running Whisper on full audio (once)...")
        result = self.transcribe(audio_path)

        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"]
                })

        self._full_transcript_cache[audio_path] = words
        return words

    def transcribe_segment(self, audio_path: str, start: float, end: float) -> str:
        """
        Get the text for a specific time range by pulling words from the
        cached full-audio transcript that overlap [start, end].
        """
        try:
            words = self._get_words(audio_path)

            segment_words = [
                w["word"] for w in words
                if w["start"] < end and w["end"] > start
            ]

            text = "".join(segment_words).strip()
            return text

        except Exception as e:
            print(f"⚠️ Segment transcription error: {e}")
            return ""