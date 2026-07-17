"""
core/syncer.py
Audio-video synchronization.
Builds the combined dubbed audio track with pydub, then muxes it onto
the video directly via ffmpeg (bypasses moviepy's write_videofile, which
can silently produce videos with no audio track on some Windows setups).
"""

import os
import subprocess
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import wave


class VideoSyncer:
    """Synchronize dubbed audio with video"""

    def __init__(self):
        os.makedirs("downloads/output", exist_ok=True)
        os.makedirs("downloads/dubbed_stretched", exist_ok=True)

    def get_audio_duration(self, audio_path: str) -> float:
        try:
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / float(rate)
        except Exception:
            return 0

    def _atempo_chain(self, ratio: float) -> str:
        filters = []
        remaining = ratio
        while remaining < 0.5 or remaining > 2.0:
            step = 2.0 if remaining > 2.0 else 0.5
            filters.append(f"atempo={step}")
            remaining /= step
        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)

    def stretch_audio_to_duration(self, input_path: str, target_duration: float, output_path: str) -> str:
        current_duration = self.get_audio_duration(input_path)
        if current_duration <= 0 or target_duration <= 0:
            return input_path

        ratio = current_duration / target_duration
        ratio = max(0.5, min(ratio, 3.0))
        atempo_filter = self._atempo_chain(ratio)

        try:
            cmd = ["ffmpeg", "-y", "-i", input_path, "-filter:a", atempo_filter, output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            print(f"⚠️ Time-stretch failed for {input_path}: {e}")
            return input_path

    def sync_segments_to_video(self, video_path: str, segments: list, output_path: str) -> str:
        """Build combined audio with pydub, then mux onto video with ffmpeg directly"""
        try:
            video = VideoFileClip(video_path)
            video_duration = video.duration
            video.close()

            print(f"📹 Video duration: {video_duration:.2f} seconds")

            # Base silent track for the whole video length
            combined = AudioSegment.silent(duration=int(video_duration * 1000), frame_rate=44100)

            stretched_dir = "downloads/dubbed_stretched"
            os.makedirs(stretched_dir, exist_ok=True)
            placed_any = False

            for i, segment in enumerate(segments):
                try:
                    audio_path = segment["audio_path"]
                    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                        print(f"⚠️ Audio file missing/empty: {audio_path}")
                        continue

                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", start_time + 1)
                    target_duration = max(end_time - start_time, 0.3)

                    if start_time >= video_duration:
                        continue

                    stretched_path = f"{stretched_dir}/seg_{i}.wav"
                    final_audio_path = self.stretch_audio_to_duration(audio_path, target_duration, stretched_path)

                    seg_audio = AudioSegment.from_wav(final_audio_path)
                    seg_audio = seg_audio.set_frame_rate(44100).set_channels(1)

                    start_ms = int(start_time * 1000)
                    combined = combined.overlay(seg_audio, position=start_ms)
                    placed_any = True

                    print(f"  Segment {i} [{segment.get('speaker', '?')}]: "
                          f"start={start_time:.2f}s, duration={target_duration:.2f}s")

                except Exception as e:
                    print(f"⚠️ Skipping segment {i}: {e}")
                    continue

            if not placed_any:
                print("❌ No valid audio clips to sync! Keeping original audio.")
                cmd = ["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path]
                subprocess.run(cmd, check=True, capture_output=True)
                return output_path

            # Export combined audio to a temp file
            combined_audio_path = "downloads/dubbed_stretched/_combined.wav"
            combined.export(combined_audio_path, format="wav")

            # Mux video + combined audio directly with ffmpeg (reliable, no moviepy audio bugs)
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", combined_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ ffmpeg mux error: {result.stderr[-500:]}")
                raise Exception("ffmpeg muxing failed")

            print(f"✅ Video synced: {os.path.basename(output_path)}")
            return output_path

        except Exception as e:
            print(f"❌ Sync error: {e}")
            try:
                cmd = ["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path]
                subprocess.run(cmd, check=True, capture_output=True)
                return output_path
            except:
                raise e