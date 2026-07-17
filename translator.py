"""
core/translator.py
Translation using Groq LLM.
Converts text from one language to another using Llama model.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TextTranslator:
    """Translate text using Groq LLM (Llama 3.3)"""
    
    def __init__(self):
        """Initialize the Groq LLM client"""
        # Get API key from .env file
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Fast and capable model
            temperature=0.3,                   # Low temp = more consistent translations
            api_key=self.api_key
        )
    
    def translate(self, text: str, target_language: str = "Hindi") -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Language to translate into
            
        Returns:
            Translated text
        """
        # Create prompt for translation
        prompt = f"""
        You are an expert translator. Translate the following text to {target_language}.
        
        IMPORTANT RULES:
        1. Keep the tone natural and conversational
        2. Don't add any extra information
        3. Don't explain the translation
        4. Return ONLY the translated text
        5. Preserve the original meaning and context
        
        Text to translate:
        {text}
        
        Translation:
        """
        
        # Get response from Groq
        response = self.llm.invoke(prompt)
        
        # Extract and clean the translated text
        translated_text = response.content.strip()
        
        return translated_text
    
    def translate_batch(self, texts: list, target_language: str = "Hindi") -> list:
        """
        Translate multiple texts at once
        
        Args:
            texts: List of text strings to translate
            target_language: Language to translate into
            
        Returns:
            List of translated texts
        """
        translated = []
        
        for text in texts:
            result = self.translate(text, target_language)
            translated.append(result)
        
        return translated
    
    def translate_with_context(self, text: str, context: str, target_language: str = "Hindi") -> str:
        """
        Translate text with additional context for better accuracy
        
        Args:
            text: Text to translate
            context: Additional context about the text (e.g., "This is a formal business meeting")
            target_language: Language to translate into
            
        Returns:
            Translated text with context-aware translation
        """
        prompt = f"""
        You are an expert translator. Translate the following text to {target_language}.
        
        CONTEXT:
        {context}
        
        TEXT TO TRANSLATE:
        {text}
        
        IMPORTANT:
        - Keep the tone appropriate for the context
        - Return ONLY the translated text
        - Don't add any explanation
        
        Translation:
        """
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def translate_segments(self, segments: list, target_language: str = "Hindi") -> list:
        """
        Translate multiple text segments (for multi-speaker content)
        
        Args:
            segments: List of dicts with 'text', 'speaker', 'start', 'end'
            target_language: Language to translate into
            
        Returns:
            List of dicts with original data + translated text
        """
        translated_segments = []
        
        for segment in segments:
            translated_text = self.translate(
                segment["text"],
                target_language
            )
            
            translated_segments.append({
                "speaker": segment.get("speaker", "Unknown"),
                "original_text": segment["text"],
                "translated_text": translated_text,
                "start": segment.get("start", 0),
                "end": segment.get("end", 0)
            })
        
        return translated_segments
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language name
        """
        prompt = f"""
        Detect the language of the following text.
        Return ONLY the language name (e.g., "English", "Hindi", "Spanish").
        
        Text: {text}
        
        Language:
        """
        
        response = self.llm.invoke(prompt)
        return response.content.strip()


# ============================================================
# 📝 SHORT NOTES - WHAT THIS FILE DOES
# ============================================================

"""
YEH FILE KYA KARTI HAI:
- Text ko ek language se doosri language mein translate karti hai
- Groq LLM (Llama 3.3 model) use karti hai
- Natural aur conversational translations deti hai

KYUN ZAROORI HAI:
- Original script ko target language mein badalna zaroori hai
- Dubbing ke liye translated text chahiye
- Multi-language support ke liye

WORKFLOW:
1. Input: English text + Target language (e.g., "Hindi")
2. Process: Groq LLM translate karega
3. Output: Translated text in target language

EXAMPLE:
Input:  "Hello, how are you?" + "Hindi"
Output: "नमस्ते, आप कैसे हैं?"

FUNCTIONS:

A) translate():
   - Simple translation
   - Best for: Single text translation
   - Fast and reliable

B) translate_batch():
   - Multiple texts ek saath translate karega
   - Best for: Bulk translation

C) translate_with_context():
   - Context ke saath translation (better accuracy)
   - Best for: Formal/informal tone adjustment

D) translate_segments():
   - Speaker-wise segments translate karega
   - Best for: Multi-speaker dubbing

E) detect_language():
   - Text ki language pehchane
   - Auto-detection ke liye

GROQ LLM ADVANTAGES:
- Free to use (limited credits)
- Very fast (10x faster than other LLMs)
- Good quality translations
- 70B parameter model (more accurate)

CONFIGURATION:
Model: llama-3.3-70b-versatile
Temperature: 0.3 (balanced - not too random, not too rigid)

TEMPERATURE EXPLANATION:
- 0.0 = Always same output (consistent)
- 0.3 = Good balance (recommended)
- 1.0 = Creative/varied (not for translation)

API KEY SETUP:
1. Get key from console.groq.com
2. Save in .env: GROQ_API_KEY=your_key_here

OUTPUT EXAMPLE:
Input:  {
    "speaker": "SPEAKER_01",
    "text": "Welcome to our channel",
    "start": 0.0,
    "end": 2.5
}
Output: {
    "speaker": "SPEAKER_01",
    "original_text": "Welcome to our channel",
    "translated_text": "हमारे चैनल में आपका स्वागत है",
    "start": 0.0,
    "end": 2.5
}

DEPENDENCIES:
- langchain-groq (Groq integration)
- python-dotenv (API key management)

NOTE: Groq provides 30 million tokens per day free!
"""

# ============================================================
# END OF FILE
# ============================================================