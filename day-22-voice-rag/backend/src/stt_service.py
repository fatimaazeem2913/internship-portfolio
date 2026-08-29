import os
import tempfile
from faster_whisper import WhisperModel

class SpeechToTextService:
    """Enterprise STT Service powered by CTranslate2 faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        print(f"[STT Service] Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        # Downloads model once and caches it locally
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT Service] ✅ Whisper model loaded successfully.")

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
        """Transcribes incoming audio bytes to plain text."""
        suffix = os.path.splitext(filename)[1] or ".webm"
        
        # Write binary stream to a temporary file for ffmpeg decoding
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            # beam_size=5 ensures accurate beam search decoding
            segments, info = self.model.transcribe(
                temp_path, 
                beam_size=5, 
                language="en",
                vad_filter=True, # Voice Activity Detection strips silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            transcript_parts = [segment.text.strip() for segment in segments]
            full_transcript = " ".join(transcript_parts).strip()
            
            print(f"[STT Service] Transcribed ({info.language} | {info.duration:.2f}s): \"{full_transcript}\"")
            return full_transcript
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Singleton instance
stt_service = SpeechToTextService(model_size="base", device="cpu", compute_type="int8")