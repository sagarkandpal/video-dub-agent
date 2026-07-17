"""
core/diarizer.py
Speaker diarization using pyannote.audio.
"""

import soundfile as sf
import os
import torch
import warnings
from huggingface_hub import login
from dotenv import load_dotenv
import librosa

warnings.filterwarnings("ignore")

load_dotenv()


class SpeakerDiarizer:
    """Identify speakers and their time segments in audio"""
    
    def __init__(self):
        """Initialize the diarization pipeline"""
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.token:
            print("⚠️ HUGGINGFACE_TOKEN not found. Using simplified diarization.")
            self.pipeline = None
            return
        
        try:
            # Login to Hugging Face
            try:
                login(token=self.token, add_to_git_credential=True)
                print("✅ Hugging Face login successful")
            except Exception as e:
                print(f"⚠️ Hugging Face login warning: {e}")
            
            from pyannote.audio import Pipeline
            
            # Load pipeline with token
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=self.token
            )
            
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                
            print("✅ Pyannote diarization loaded successfully!")
                
        except Exception as e:
            print(f"⚠️ Pyannote loading error: {e}")
            print("Using simplified diarization...")
            self.pipeline = None
    
    def diarize(self, audio_path: str) -> list:
        """Identify speakers in the audio file"""
        if self.pipeline is None:
            # Get audio duration
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(audio_path)
                duration = len(audio) / 1000.0
            except:
                duration = 60
            
            # Split into 2 speakers if longer than 30 seconds
            if duration > 30:
                return [
                    {"speaker": "SPEAKER_01", "start": 0, "end": duration/2, "duration": duration/2},
                    {"speaker": "SPEAKER_02", "start": duration/2, "end": duration, "duration": duration/2}
                ]
            else:
                return [
                    {"speaker": "SPEAKER_01", "start": 0, "end": duration, "duration": duration}
                ]
        
        try:
            # Load audio via soundfile and convert to waveform tensor
            # (bypasses torchcodec, which has DLL issues on Windows)
            data, sample_rate = sf.read(audio_path, dtype='float32')
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)   # mono -> (1, time)
            else:
                waveform = torch.from_numpy(data.T)               # (time, ch) -> (ch, time)

            diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

            speakers = []
            for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.end - segment.start
                })
            
            if not speakers:
                # Fallback
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_wav(audio_path)
                    duration = len(audio) / 1000.0
                except:
                    duration = 60
                
                return [
                    {"speaker": "SPEAKER_01", "start": 0, "end": duration, "duration": duration}
                ]
            
            return speakers
            
        except Exception as e:
            print(f"⚠️ Diarization failed: {e}")
            # Fallback
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(audio_path)
                duration = len(audio) / 1000.0
            except:
                duration = 60
            
            return [
                {"speaker": "SPEAKER_01", "start": 0, "end": duration, "duration": duration}
            ]

    
    def detect_speaker_gender(self, audio_path: str, speakers: list) -> dict:
        """
        Estimate gender per speaker by concatenating all of that speaker's
        audio segments into one array, then running pitch analysis once.
        """
        import numpy as np

        try:
            import soundfile as sf
            y, sr = sf.read(audio_path, dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)
        except Exception as e:
            print(f"⚠️ Could not load audio for gender detection: {e}")
            return {}

        speaker_segments = {}
        for s in speakers:
            speaker_segments.setdefault(s["speaker"], []).append(s)

        gender_map = {}

        for speaker_id, segs in speaker_segments.items():
            chunks = []
            for seg in segs:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                chunk = y[start_sample:end_sample]
                if len(chunk) > 0:
                    chunks.append(chunk)

            if not chunks:
                gender_map[speaker_id] = "unknown"
                continue

            # Concatenate all this speaker's audio into one long array
            combined = np.concatenate(chunks)

            if len(combined) < sr * 0.3:  # still too short overall
                gender_map[speaker_id] = "unknown"
                print(f"⚠️ Not enough audio for {speaker_id}, defaulting to unknown")
                continue

            try:
                f0, voiced_flag, _ = librosa.pyin(
                    combined,
                    fmin=65.0,
                    fmax=400.0,
                    sr=sr,
                    frame_length=1024,
                    hop_length=256
                )
                valid_f0 = f0[~np.isnan(f0)]

                if len(valid_f0) == 0:
                    gender_map[speaker_id] = "unknown"
                    print(f"⚠️ No voiced pitch found for {speaker_id}, defaulting to unknown")
                    continue

                avg_pitch = float(np.median(valid_f0))
                gender = "male" if avg_pitch < 165 else "female"
                gender_map[speaker_id] = gender
                print(f"🎯 {speaker_id}: avg pitch = {avg_pitch:.1f} Hz → {gender}")

            except Exception as e:
                print(f"⚠️ Pitch detection failed for {speaker_id}: {e}")
                gender_map[speaker_id] = "unknown"

        return gender_map
    
    def get_unique_speakers(self, speakers: list) -> list:
        unique = list(set([s["speaker"] for s in speakers]))
        return sorted(unique)
    
    def get_speaker_segments(self, speakers: list, speaker_id: str) -> list:
        return [s for s in speakers if s["speaker"] == speaker_id]
