"""
main.py
Local testing script for Video Dubbing Agent.
All files saved in downloads/ folder.
"""

import os
import sys
import time
import re
from dotenv import load_dotenv
from core.pipeline import DubbingPipeline
from downloaders.yt_downloader import YouTubeDownloader

load_dotenv()

# Initialize YouTube downloader
youtube_downloader = YouTubeDownloader(base_dir="downloads")


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def main():
    """Main function to test the dubbing pipeline"""
    
    print_header("🎬 VIDEO DUBBING AGENT - LOCAL TEST")
    print("\nThis script supports:")
    print("  ✅ Local video files (MP4, MOV, AVI, etc.)")
    print("  ✅ YouTube URLs (direct download)")
    print("  ✅ All files saved in 'downloads/' folder")
    print("  ✅ Hugging Face token for speaker detection")
    print("  ✅ Groq API key for translation")
    print("  ✅ ElevenLabs API key for voice cloning")
    
    # Get video input
    print("\n" + "-"*60)
    print("📁 VIDEO INPUT")
    print("-"*60)
    print("\nEnter a local video path OR YouTube URL")
    print("Examples:")
    print("  📹 Local: C:/Users/username/Videos/video.mp4")
    print("  📹 YouTube: https://youtube.com/watch?v=xxxxx")
    
    video_input = input("\nEnter video path or YouTube URL: ").strip()
    
    if not video_input:
        print("\n❌ No input provided!")
        sys.exit(1)
    
    # Check if it's a YouTube URL
    if youtube_downloader.is_youtube_url(video_input):
        print(f"\n🎬 YouTube URL detected: {video_input}")
        
        # Show video info
        print("\n📊 Fetching video info...")
        info = youtube_downloader.get_video_info(video_input)
        if "error" not in info:
            print(f"  📹 Title: {info.get('title', 'Unknown')}")
            print(f"  ⏱️ Duration: {info.get('duration', 0)} seconds")
            print(f"  👤 Uploader: {info.get('uploader', 'Unknown')}")
            print(f"  👁️ Views: {info.get('views', 0):,}")
        
        try:
            video_path = youtube_downloader.download_video(video_input)
        except Exception as e:
            print(f"❌ Failed to download YouTube video: {e}")
            sys.exit(1)
    else:
        video_path = video_input.strip()
        if not os.path.exists(video_path):
            print(f"\n❌ File not found: {video_path}")
            sys.exit(1)
    
    print(f"\n✅ Using video: {video_path}")
    if os.path.exists(video_path):
        print(f"📊 File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Get target language
    print("\n" + "-"*60)
    print("🌍 TARGET LANGUAGE")
    print("-"*60)
    
    languages = ["Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Arabic", "Portuguese", "Russian"]
    
    print("\nAvailable languages:")
    for i, lang in enumerate(languages, 1):
        print(f"  {i}. {lang}")
    
    lang_choice = input("\nSelect language (number) [1]: ").strip()
    
    if not lang_choice or lang_choice == "1":
        target_language = "Hindi"
    else:
        try:
            target_language = languages[int(lang_choice) - 1]
        except:
            target_language = "Hindi"
    
    print(f"\n✅ Target language: {target_language}")
    
    # Confirm and start
    print("\n" + "-"*60)
    print("🚀 START PROCESSING")
    print("-"*60)
    
    print("\nSummary:")
    print(f"  📹 Video: {os.path.basename(video_path)}")
    print(f"  🌍 Language: {target_language}")
    print(f"  📁 All files saved in: downloads/")
    
    confirm = input("\nStart processing? (y/n): ").lower()
    if confirm != 'y':
        print("\n❌ Cancelled by user.")
        sys.exit(0)
    
    # Run pipeline
    print_header("🔧 PROCESSING PIPELINE")
    print("\n⏳ Initializing pipeline...")
    pipeline = DubbingPipeline()
    
    # Process with progress tracking
    def log_progress(step, message):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {step}: {message}")
    
    # Override pipeline methods to log progress
    original_extract = pipeline._extract_audio
    original_diarize = pipeline._diarize_speakers
    original_transcribe = pipeline._transcribe_segments
    original_translate = pipeline._translate_text
    original_dub = pipeline._generate_dub
    original_sync = pipeline._sync_video
    
    def log_extract(state):
        log_progress("Step 1/6", "Extracting audio from video...")
        result = original_extract(state)
        log_progress("Step 1/6", f"✅ Audio extracted: {os.path.basename(result['audio_path'])}")
        return result
    
    def log_diarize(state):
        log_progress("Step 2/6", "Detecting speakers...")
        result = original_diarize(state)
        speakers = len(set([s["speaker"] for s in result["speakers"]]))
        log_progress("Step 2/6", f"✅ Found {speakers} speakers")
        return result
    
    def log_transcribe(state):
        log_progress("Step 3/6", "Transcribing audio...")
        result = original_transcribe(state)
        total_segments = sum(len(seg) for seg in result["transcripts"].values())
        log_progress("Step 3/6", f"✅ Transcribed {total_segments} segments")
        return result
    
    def log_translate(state):
        log_progress("Step 4/6", f"Translating to {state.get('target_language', 'Hindi')}...")
        result = original_translate(state)
        log_progress("Step 4/6", "✅ Translation complete")
        return result
    
    def log_dub(state):
        log_progress("Step 5/6", "Generating dubbed voices...")
        result = original_dub(state)
        log_progress("Step 5/6", f"✅ Generated {len(result['dubbed_segments'])} voice segments")
        return result
    
    def log_sync(state):
        log_progress("Step 6/6", "Syncing audio with video...")
        result = original_sync(state)
        log_progress("Step 6/6", f"✅ Video synced: {os.path.basename(result['output_path'])}")
        return result
    
    pipeline._extract_audio = log_extract
    pipeline._diarize_speakers = log_diarize
    pipeline._transcribe_segments = log_transcribe
    pipeline._translate_text = log_translate
    pipeline._generate_dub = log_dub
    pipeline._sync_video = log_sync
    
    try:
        start_time = time.time()
        
        print("\n⏳ Processing started...\n")
        result = pipeline.process_video(video_path, target_language)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print_header("✅ PROCESSING COMPLETE")
        
        print(f"\n⏱️ Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        print("\n📊 Results:")
        print(f"  📹 Output video: {result.get('output_path', 'Not found')}")
        print(f"  👤 Speakers detected: {len(set([s['speaker'] for s in result.get('speakers', [])]))}")
        print(f"  📝 Segments processed: {len(result.get('dubbed_segments', []))}")
        print(f"  📌 Status: {result.get('status', 'Complete')}")
        
        # Show all file locations
        print("\n" + "-"*60)
        print("📁 ALL FILES LOCATION")
        print("-"*60)
        print("\n📂 downloads/ folder contains:")
        print(f"  📹 Videos: downloads/videos/")
        print(f"  🎵 Audio: downloads/audio/")
        print(f"  📝 Transcripts: downloads/transcripts/")
        print(f"  🎙️ Dubbed: downloads/dubbed/")
        print(f"  📤 Output: downloads/output/")
        
        # Check output file
        if result.get("output_path") and os.path.exists(result["output_path"]):
            output_size = os.path.getsize(result["output_path"]) / (1024*1024)
            print(f"\n📦 Output file: {result['output_path']}")
            print(f"📦 Output size: {output_size:.2f} MB")
            
            # Ask to play
            play = input("\nDo you want to play the video? (y/n): ").lower()
            if play == 'y':
                try:
                    import subprocess
                    import platform
                    
                    if platform.system() == "Darwin":
                        subprocess.run(["open", result["output_path"]])
                    elif platform.system() == "Windows":
                        subprocess.run(["start", result["output_path"]], shell=True)
                    else:
                        subprocess.run(["xdg-open", result["output_path"]])
                    print("\n🎬 Video player opened!")
                except Exception as e:
                    print(f"\n⚠️ Could not open video: {e}")
        else:
            print("\n❌ No output video found!")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Process interrupted by user.")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        print("\n" + "-"*60)
        print("📋 FULL ERROR TRACEBACK:")
        print("-"*60)
        traceback.print_exc()


if __name__ == "__main__":
    main()