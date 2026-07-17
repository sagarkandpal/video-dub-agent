"""
core/dubber.py
Voice cloning using edge-tts (free, no API key needed).
"""

import os
import asyncio
import subprocess
import wave
import numpy as np
import edge_tts

# Free voice pool per target language. Two distinct voices per language
# so different speakers get different voices (identity preserved).
VOICE_POOL = {
    "Hindi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "Spanish": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "French": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "German": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "Japanese": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "Chinese": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "Arabic": ["ar-SA-ZariyahNeural", "ar-SA-HamedNeural"],
    "Portuguese": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"],
    "Russian": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
}


class VoiceDubber:
    """Generate dubbed voices using edge-tts"""

    def __init__(self, use_elevenlabs: bool = False):
        # use_elevenlabs kept as a parameter only so pipeline.py doesn't break;
        # edge-tts is always used now.
        self.speaker_voice_map = {}
        print("✅ edge-tts dubber initialized (free, no API key needed)")

    def get_voice_for_speaker(self, speaker_id: str, target_language: str = "Hindi") -> str:
        """Return a stable voice for a given speaker_id (consistent across the video)"""
        voices = VOICE_POOL.get(target_language, VOICE_POOL["Hindi"])
        if speaker_id not in self.speaker_voice_map:
            idx = len(self.speaker_voice_map) % len(voices)
            self.speaker_voice_map[speaker_id] = voices[idx]
            print(f"🎙️ Assigned voice {self.speaker_voice_map[speaker_id]} to {speaker_id}")
        return self.speaker_voice_map[speaker_id]

    def generate_dub(self, text: str, output_path: str, speaker_voice: str = None) -> str:
        """Generate dubbed audio from text using edge-tts, saved as real WAV"""
        if not text or not text.strip():
            text = "..."

        voice = speaker_voice or "hi-IN-SwaraNeural"
        temp_mp3 = output_path.replace(".wav", "_raw.mp3")

        try:
            async def _run():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(temp_mp3)

            asyncio.run(_run())

            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                raise Exception("edge-tts produced empty file")

            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "44100", "-ac", "1", output_path],
                check=True, capture_output=True
            )
            os.remove(temp_mp3)

            print(f"✅ Generated audio: {os.path.basename(output_path)} "
                  f"({os.path.getsize(output_path)/1024:.1f} KB) [voice={voice}]")
            return output_path

        except Exception as e:
            print(f"⚠️ edge-tts generation failed: {e}")
            print("Using silent audio fallback...")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            return self._create_silent_audio(text, output_path, duration=5)

    def _create_silent_audio(self, text: str, output_path: str, duration: float = 5.0) -> str:
        """Create silent audio file as fallback"""
        try:
            sample_rate = 16000
            samples = int(duration * sample_rate)
            silent_audio = np.zeros(samples, dtype=np.int16)

            with wave.open(output_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(silent_audio.tobytes())

            print(f"🔇 Created silent audio: {os.path.basename(output_path)} ({duration}s)")
            return output_path

        except Exception as e:
            print(f"⚠️ Could not create silent audio: {e}")
            return output_path

    def generate_segment_dubs(self, segments: list, output_dir: str = "downloads/dubbed",
                               target_language: str = "Hindi") -> list:
        """Generate dubs for multiple text segments, one stable voice per speaker"""
        os.makedirs(output_dir, exist_ok=True)
        dubbed_segments = []

        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "SPEAKER_01")
            output_path = f"{output_dir}/{speaker}_{i}.wav"

            text = segment.get("text", "")
            if not text or len(text.strip()) == 0:
                text = "..."

            segment_duration = segment.get("end", 5) - segment.get("start", 0)
            if segment_duration < 3:
                segment_duration = 5

            voice_id = segment.get("speaker_voice") or self.get_voice_for_speaker(speaker, target_language)

            self.generate_dub(text=text, output_path=output_path, speaker_voice=voice_id)

            try:
                with wave.open(output_path, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    actual_duration = frames / float(rate)
                if actual_duration == 0:
                    actual_duration = segment_duration
            except:
                actual_duration = segment_duration

            dubbed_segments.append({
                "speaker": speaker,
                "audio_path": output_path,
                "start": segment.get("start", 0),
                "end": segment.get("start", 0) + actual_duration,
                "text": text,
                "duration": actual_duration
            })

        return dubbed_segments