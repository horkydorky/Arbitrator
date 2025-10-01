# in video_processor.py

import os
import subprocess
import whisper
from typing import Optional
# torch is not directly used, but good to know it's a dependency for whisper/pyannote
# import torch 
from pyannote.audio import Pipeline
from dotenv import load_dotenv


load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


def process_video_to_transcript(video_path: str, num_speakers: Optional[int] = None) -> str:
    """
    Takes the path to a video file and performs the full transcription and diarization pipeline,
    normalizing speaker labels to SPEAKER_00, SPEAKER_01, etc.
    Optionally constrains the diarization to a specific number of speakers.
    """
    if not HUGGING_FACE_TOKEN:
        return "ERROR: HUGGING_FACE_TOKEN not found in .env file."


    temp_audio_path = f"temp_audio_{os.path.basename(video_path)}.wav"
    
    try:
        print(f"Step 1/4: Extracting audio from '{video_path}'...")
       
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", "-y", temp_audio_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True) # Use text=True for better error logs

        print("Step 2/4: Transcribing audio with Whisper...")
        model = whisper.load_model("base")
        transcribe_result = model.transcribe(temp_audio_path, word_timestamps=True)

        print("Step 3/4: Diarizing speakers with Pyannote...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGING_FACE_TOKEN
        )
        
        # --- FLEXIBLE DIARIZATION LOGIC ---
        if num_speakers and num_speakers > 0:
            print(f"  - Applying constraint for {num_speakers} speakers.")
            diarization_result = diarization_pipeline(temp_audio_path, num_speakers=num_speakers)
        else:
            print("  - Using automatic speaker detection.")
            diarization_result = diarization_pipeline(temp_audio_path)
        # --- END OF FLEXIBLE LOGIC ---

        print("Step 4/4: Aligning transcript and normalizing speakers...")
        word_segments = [word for seg in transcribe_result.get('segments', []) for word in seg.get('words', [])]
        speaker_turns = list(diarization_result.itertracks(yield_label=True))
        
        if not speaker_turns:
            return "ERROR: Pyannote could not detect any speakers."
        if not word_segments:
             return "ERROR: Whisper could not transcribe any words."

        # --- This logic is already dynamic and works perfectly for N speakers ---
        original_speaker_labels = sorted(list(set([label for _, _, label in speaker_turns])))
        speaker_map = {original_label: f"SPEAKER_{i:02d}" for i, original_label in enumerate(original_speaker_labels)}

        full_transcript = ""
        for turn, _, speaker in speaker_turns:
            normalized_speaker = speaker_map.get(speaker, "UNKNOWN_SPEAKER")
            
            words_in_turn = [word['word'] for word in word_segments if turn.start <= word.get('start', -1) < turn.end]
            if words_in_turn:
                full_transcript += f"{normalized_speaker}: {''.join(words_in_turn).strip()}\n"
                
        if not full_transcript:
            return "ERROR: Failed to align words with speaker turns."

        return full_transcript.strip()

    except subprocess.CalledProcessError as e:
        return f"ERROR: FFmpeg failed with error: {e.stderr}"
    except Exception as e:
        return f"ERROR: An unexpected error occurred in video processing: {str(e)}"
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)