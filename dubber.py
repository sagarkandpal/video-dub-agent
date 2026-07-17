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
# Free voice pool per target language, split by gender
VOICE_POOL = {
    "Hindi":      {"male": ["hi-IN-MadhurNeural"],   "female": ["hi-IN-SwaraNeural"]},
    "Spanish":    {"male": ["es-ES-AlvaroNeural"],   "female": ["es-ES-ElviraNeural"]},
    "French":     {"male": ["fr-FR-HenriNeural"],    "female": ["fr-FR-DeniseNeural"]},
    "German":     {"male": ["de-DE-ConradNeural"],   "female": ["de-DE-KatjaNeural"]},
    "Japanese":   {"male": ["ja-JP-KeitaNeural"],    "female": ["ja-JP-NanamiNeural"]},
    "Chinese":    {"male": ["zh-CN-YunxiNeural"],    "female": ["zh-CN-XiaoxiaoNeural"]},
    "Arabic":     {"male": ["ar-SA-HamedNeural"],    "female": ["ar-SA-ZariyahNeural"]},
    "Portuguese": {"male": ["pt-BR-AntonioNeural"],  "female": ["pt-BR-FranciscaNeural"]},
    "Russian":    {"male": ["ru-RU-DmitryNeural"],   "female": ["ru-RU-SvetlanaNeural"]},
}


class VoiceDubber:
    """Generate dubbed voices using edge-tts"""

    def __init__(self, use_elevenlabs: bool = False):
        # use_elevenlabs kept as a parameter only so pipeline.py doesn't break;
        # edge-tts is always used now.
        self.speaker_voice_map = {}
        print("✅ edge-tts dubber initialized (free, no API key needed)")


    def get_voice_for_speaker(self, speaker_id: str, target_language: str = "Hindi", gender: str = "unknown") -> str:
        """Return a voice for this speaker, matched to detected gender when known"""
        lang_pool = VOICE_POOL.get(target_language, VOICE_POOL["Hindi"])

        if speaker_id not in self.speaker_voice_map:
            if gender == "male":
                voice = lang_pool["male"][0]
            elif gender == "female":
                voice = lang_pool["female"][0]
            else:
                # unknown gender: alternate between pools so different speakers still sound different
                idx = len(self.speaker_voice_map) % 2
                voice = lang_pool["female"][0] if idx == 0 else lang_pool["male"][0]

            self.speaker_voice_map[speaker_id] = voice
            print(f"🎙️ Assigned voice {voice} to {speaker_id} (gender={gender})")

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



    def generate_segment_dubs(self, segments: list, output_dir: str = "downloads/dubbed", target_language: str = "Hindi") -> list:

        """Generate dubs for multiple text segments IN PARALLEL, one stable voice per speaker"""
        os.makedirs(output_dir, exist_ok=True)

        # Pre-assign voices for all speakers first (must be sequential/deterministic)
        jobs = []
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

            jobs.append({
                "index": i,
                "speaker": speaker,
                "output_path": output_path,
                "text": text,
                "voice": voice_id,
                "start": segment.get("start", 0),
                "end": segment.get("end", segment.get("start", 0) + segment_duration),
                "segment_duration": segment_duration
            })

        # Run all TTS generations concurrently
        asyncio.run(self._generate_all_async(jobs))

        # Build final results, reading actual durations after generation
        dubbed_segments = []
        for job in jobs:
            output_path = job["output_path"]
            try:
                with wave.open(output_path, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    actual_duration = frames / float(rate)
                if actual_duration == 0:
                    actual_duration = job["segment_duration"]
            except:
                actual_duration = job["segment_duration"]

            dubbed_segments.append({
                "speaker": job["speaker"],
                "audio_path": output_path,
                "start": job["start"],
                "end": job["end"],              # ← original diarization end, TTS duration se nahi
                "text": job["text"],
                "duration": actual_duration      # actual TTS audio duration alag se rakhi, syncer isko stretch karega
            })      

        return dubbed_segments


    async def _generate_all_async(self, jobs: list):
        """Run edge-tts generation for all jobs concurrently, limited batches to avoid overload"""
        semaphore = asyncio.Semaphore(5)  # max 5 concurrent TTS calls at once

        async def _generate_one(job):
            async with semaphore:
                await self._generate_dub_async(job["text"], job["output_path"], job["voice"])

        await asyncio.gather(*[_generate_one(job) for job in jobs])


    async def _generate_dub_async(self, text: str, output_path: str, voice: str):
        """Async version of generate_dub, used internally for parallel generation"""
        if not text or not text.strip():
            text = "..."

        temp_mp3 = output_path.replace(".wav", "_raw.mp3")

        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_mp3)

            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                raise Exception("edge-tts produced empty file")

            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "44100", "-ac", "1", output_path],
                check=True, capture_output=True
            )
            os.remove(temp_mp3)

            print(f"✅ Generated audio: {os.path.basename(output_path)} [voice={voice}]")

        except Exception as e:
            print(f"⚠️ edge-tts generation failed for {output_path}: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            self._create_silent_audio(text, output_path, duration=5)
