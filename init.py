"""
core/__init__.py
Package initializer for the core module.
Exposes all main components for easy import.
"""

# Import all core components
from .extractor import AudioExtractor
from .transcriber import VideoTranscriber
from .translator import TextTranslator
from .diarizer import SpeakerDiarizer
from .dubber import VoiceDubber
from .syncer import VideoSyncer
from .pipeline import DubbingPipeline

# Define what gets imported with "from core import *"
__all__ = [
    'AudioExtractor',
    'VideoTranscriber',
    'TextTranslator',
    'SpeakerDiarizer',
    'VoiceDubber',
    'VideoSyncer',
    'DubbingPipeline'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "AI-powered video dubbing with speaker detection and voice cloning"


# ============================================================
# 📝 SHORT NOTES - WHAT THIS FILE DOES
# ============================================================

"""
YEH FILE KYA KARTI HAI:
- Core folder ko Python package banati hai
- Sare components (extractor, transcriber, etc.) ko ek jagah import karti hai
- "from core import ClassName" kaam karne ke liye

KYUN ZAROORI HAI:
- Bina is file ke "from core import ..." kaam nahi karega
- Har file mein alag se import nahi karna padta

WORKFLOW:
1. Jab koi "from core import DubbingPipeline" likhta hai
2. Python __init__.py check karta hai
3. __init__.py mein DubbingPipeline imported hai
4. Toh directly access mil jata hai

BASIC EXAMPLE:
from core import DubbingPipeline  ✅ Clean
from core.pipeline import DubbingPipeline  ❌ Lamba
"""

# ============================================================
# END OF FILE
# ============================================================