import asyncio
import edge_tts
import io
import os
import tempfile
import gc  # For garbage collection

class AudioGenerator:
    def __init__(self):
        # English voices
        self.english_voices = [
            {"name": "English (Male)", "voice": "en-US-GuyNeural"},
            {"name": "English (Female)", "voice": "en-US-JennyNeural"},
            {"name": "English (British Male)", "voice": "en-GB-RyanNeural"},
            {"name": "English (British Female)", "voice": "en-GB-SoniaNeural"},
            {"name": "English (Australian)", "voice": "en-AU-NatashaNeural"}
        ]
        
        # Indian language voices
        self.indian_voices = [
            {"name": "Hindi (Female)", "voice": "hi-IN-SwaraNeural"},
            {"name": "Hindi (Male)", "voice": "hi-IN-MadhurNeural"},
            {"name": "Tamil", "voice": "ta-IN-PallaviNeural"},
            {"name": "Telugu", "voice": "te-IN-ShrutiNeural"},
            {"name": "Punjabi", "voice": "pa-IN-GurleenNeural"}
        ]
    
    def get_all_voices(self):
        """Return all available voice options"""
        return self.english_voices + self.indian_voices
    
    async def generate_audio(self, text, voice_id):
        """Generate audio using edge-tts with the specified voice"""
        temp_path = None
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_path = temp_file.name
            
            print(f"Creating audio with voice {voice_id} at path {temp_path}")
            
            # Configure the TTS communication
            tts = edge_tts.Communicate(text=text, voice=voice_id)
            
            # Generate audio and save to temporary file
            await tts.save(temp_path)
            
            # Read the file into bytes
            with open(temp_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"Temp file {temp_path} deleted successfully")
                
            return audio_data
        
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            # Clean up if file exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"Cleaned up temp file {temp_path} after error")
                except Exception as cleanup_error:
                    print(f"Failed to clean temp file: {cleanup_error}")
            raise