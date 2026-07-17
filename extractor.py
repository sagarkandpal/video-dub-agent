"""
core/extractor.py
Audio extraction from video files and YouTube URLs.
All files saved in downloads/ folder.
"""

import os
import sys
from pydub import AudioSegment
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from downloaders.yt_downloader import YouTubeDownloader

load_dotenv()


class AudioExtractor:
    """Extract audio from video files and YouTube URLs"""
    
    def __init__(self):
        """Initialize the audio extractor"""
        # Initialize YouTube downloader with downloads folder
        self.youtube_downloader = YouTubeDownloader(base_dir="downloads")
        
        # Get base directory
        self.base_dir = "downloads"
        self.audio_dir = f"{self.base_dir}/audio"
        
        # Ensure directories exist
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(f"{self.base_dir}/uploads", exist_ok=True)
        
        print(f"📁 Audio directory: {self.audio_dir}")
    
    def extract_from_video(self, video_path: str, output_path: str = None) -> str:
        """
        Extract audio from video file or YouTube URL
        
        Args:
            video_path: Path to video file OR YouTube URL
            output_path: Where to save audio (in downloads/audio/)
            
        Returns:
            Path to extracted audio
        """
        # Check if input is YouTube URL
        if self.youtube_downloader.is_youtube_url(video_path):
            print("🎬 YouTube URL detected. Downloading audio...")
            
            # Generate output path in downloads/audio/
            if output_path is None:
                # Get video title from URL
                info = self.youtube_downloader.get_video_info(video_path)
                title = info.get("title", "youtube_audio")
                # Clean title for filename
                title = re.sub(r'[^\w\s-]', '', title).strip()
                title = re.sub(r'[-\s]+', '_', title)
                output_path = f"{self.audio_dir}/{title}.wav"
            
            # Download audio directly
            return self.youtube_downloader.download_audio(video_path, output_path)
        
        # Local file - extract audio to downloads/audio/
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{self.audio_dir}/{base_name}.wav"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            audio = AudioSegment.from_file(video_path)
            audio.export(output_path, format='wav')
            print(f"✅ Audio extracted: {os.path.basename(output_path)}")
            print(f"📁 Location: {output_path}")
            return output_path
        except Exception as e:
            raise Exception(f"Failed to extract audio: {str(e)}")
    
    def download_youtube_video(self, url: str) -> str:
        """
        Download YouTube video to downloads/videos/
        
        Args:
            url: YouTube URL
            
        Returns:
            Path to downloaded video file
        """
        return self.youtube_downloader.download_video(url)
    
    def get_youtube_info(self, url: str) -> dict:
        """
        Get YouTube video information
        
        Args:
            url: YouTube URL
            
        Returns:
            Dict with video metadata
        """
        return self.youtube_downloader.get_video_info(url)


# Add missing import
import re