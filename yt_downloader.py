"""
downloaders/youtube_downloader.py
YouTube video and audio downloader using yt-dlp.
Handles all downloads and processing in the downloads folder.
"""

import os
import re
import yt_dlp
from pathlib import Path


class YouTubeDownloader:
    """Download and process videos from YouTube"""
    
    def __init__(self, base_dir: str = "downloads"):
        """
        Initialize the YouTube downloader
        
        Args:
            base_dir: Base directory for all downloads
        """
        self.base_dir = base_dir
        
        # Create all subdirectories
        self.dirs = {
            'videos': f"{base_dir}/videos",
            'audio': f"{base_dir}/audio",
            'transcripts': f"{base_dir}/transcripts",
            'dubbed': f"{base_dir}/dubbed",
            'output': f"{base_dir}/output",
            'uploads': f"{base_dir}/uploads",
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 Downloads directory: {base_dir}/")
    
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube link"""
        patterns = [
            r'(?:www\.)?youtube\.com/watch\?v=',
            r'(?:www\.)?youtu\.be/',
            r'(?:www\.)?youtube\.com/shorts/',
            r'(?:www\.)?youtube\.com/embed/'
        ]
        return any(re.search(p, url) for p in patterns)
    
    def download_video(self, url: str, quality: str = "bestvideo[height<=720]+bestaudio/best[height<=720]") -> str:
        """
        Download YouTube video to downloads/videos/
        
        Args:
            url: YouTube URL
            quality: Video quality setting
            
        Returns:
            Path to downloaded video file
        """
        output_template = f"{self.dirs['videos']}/%(title)s.%(ext)s"
        
        ydl_opts = {
            'format': quality,
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                
                # Ensure correct file exists
                if not os.path.exists(video_path):
                    base_path = os.path.splitext(video_path)[0]
                    for ext in ['.mp4', '.webm', '.mkv']:
                        test_path = base_path + ext
                        if os.path.exists(test_path):
                            video_path = test_path
                            break
                
                print(f"✅ Video downloaded: {os.path.basename(video_path)}")
                print(f"📁 Location: {video_path}")
                return video_path
                
        except Exception as e:
            raise Exception(f"YouTube download failed: {str(e)}")
    
    def download_audio(self, url: str, output_path: str = None) -> str:
        """
        Download audio from YouTube to downloads/audio/
        
        Args:
            url: YouTube URL
            output_path: Custom output path (optional)
            
        Returns:
            Path to downloaded audio file
        """
        if output_path is None:
            output_path = f"{self.dirs['audio']}/%(title)s.wav"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_path.replace('.wav', ''),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = output_path
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    if os.path.exists(output_path.replace('.wav', '.wav')):
                        audio_path = output_path.replace('.wav', '.wav')
                
                print(f"✅ Audio downloaded: {os.path.basename(audio_path)}")
                print(f"📁 Location: {audio_path}")
                return audio_path
                
        except Exception as e:
            raise Exception(f"YouTube audio download failed: {str(e)}")
    
    def get_video_info(self, url: str) -> dict:
        """
        Get video information without downloading
        
        Args:
            url: YouTube URL
            
        Returns:
            Dict with video metadata
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "views": info.get("view_count", 0),
                    "description": info.get("description", "")[:200],
                    "thumbnail": info.get("thumbnail", ""),
                }
        except Exception as e:
            return {"error": str(e)}
    
    def get_path(self, folder: str, filename: str) -> str:
        """Get full path for a file in a specific folder"""
        return f"{self.dirs[folder]}/{filename}"