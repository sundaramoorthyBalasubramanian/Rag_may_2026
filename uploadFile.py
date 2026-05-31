import os
from pypdf import PdfReader
import whisper
from moviepy.editor import VideoFileClip

# Initialize Whisper model as a private global variable
_whisper_model = None

def _get_whisper_model():
    """Loads the Whisper model into memory only when needed."""
    global _whisper_model
    if _whisper_model is None:
        # Using the 'base' model for a good balance of speed and precision
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits a body of text into overlapping word chunks for RAG."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdf(file_path: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Reads a PDF file, extracts its text, and returns chunks."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    if not full_text.strip():
        raise ValueError("The provided PDF file contains no extractable text.")
        
    return chunk_text(full_text, chunk_size, chunk_overlap)

def process_mp4(file_path: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Extracts audio from an MP4 file, converts it to text via Whisper, and returns chunks."""
    audio_path = os.path.splitext(file_path)[0] + "_temp.mp3"
    
    try:
        # 1. Extract audio stream out of the video file
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        
        # 2. Transcribe audio to text
        model = _get_whisper_model()
        result = model.transcribe(audio_path)
        transcript = result.get("text", "")
        
    finally:
        # 3. Always clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    if not transcript.strip():
        raise ValueError("Whisper could not extract any speech or text from the video.")
        
    return chunk_text(transcript, chunk_size, chunk_overlap)
