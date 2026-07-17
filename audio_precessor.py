from git import List
import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloaders"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_yt_audio(url: str) -> str:
    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': r'C:\ffmpeg\bin',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.wav').replace('.m4a', '.wav')
        return audio_file


#manchester = https://youtu.be/iR5U92Eq-_8?si=uNVthlmViqmjwgtQ

def convert_to_wav(input_file: str) -> str:
    output_path = os.path.splitext(input_file)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes : int = 10) -> list:
    """Chunk the WAV file into smaller segments of specified duration"""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks


def process_audio(source: str) -> list:
    if source.startswith("http://url") or source.startswith("https://"):
        print("Detected youtube url, Downloading audio...")
        wav_path = download_yt_audio(source)
    else:
        print("Detected local audio file. Converting to WAV...")
        wav_path = convert_to_wav(source)
    
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio processed into - {len(chunks)} chunks created.")
    return chunks