# in main.py

import os
import shutil
import asyncio
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv

# --- Import ALL logic modules ---
from video_processor import process_video_to_transcript
from fact_filter import analyze_debate_transcript
from fact_analyzer import analyze_facts 

load_dotenv()

# --- Define allowed video content types for robust input validation ---
ALLOWED_VIDEO_TYPES: List[str] = [
    "video/mp4", 
    "video/quicktime", # .mov
    "video/x-msvideo", # .avi
    "video/x-matroska", # .mkv
    "video/webm"
]

app = FastAPI(
    title="Debate Video Analysis Engine",
    description="A comprehensive API to analyze debate videos.",
    version="5.2.0" # Version bump for input validation
)

@app.post("/analyze-video/", tags=["Analysis"])
async def analyze_video_endpoint(
    video_file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None, description="Optional: Specify the exact number of speakers in the video.")
):
    # --- ADDED: Input validation to prevent FFmpeg errors on non-video files ---
    if video_file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Please upload a video file. "
                   f"Received '{video_file.content_type}', but allowed types are: {', '.join(ALLOWED_VIDEO_TYPES)}"
        )

    temp_video_path = f"temp_{video_file.filename}"
    
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
        
    try:
        # --- Stage 1: Transcription (with optional speaker count) ---
        transcript = process_video_to_transcript(temp_video_path, num_speakers=num_speakers)
        if transcript.startswith("ERROR:"):
             raise HTTPException(status_code=500, detail=transcript)
        
        initial_analysis = analyze_debate_transcript(transcript)
        if not initial_analysis or "error" in initial_analysis:
            error_detail = initial_analysis.get('error', 'Unknown error during claim analysis.')
            raise HTTPException(status_code=500, detail=f"Failed to perform initial claim analysis: {error_detail}")

        # --- Stage 2: Prepare Data for Deep-Dive Analysis (DYNAMICALLY) ---
        facts_to_verify = []
        # This loop now handles any number of speakers returned by the analysis
        for key, claims in initial_analysis.items():
            if key.startswith("speaker") and isinstance(claims, list):
                # Extract the number from the key (e.g., '0' from 'speaker0')
                speaker_num_str = key.replace("speaker", "")
                try:
                    speaker_id = f"SPEAKER_{int(speaker_num_str):02d}"
                    for claim in claims:
                        clean_claim = claim.replace("Fact-based claim:", "").replace("Subjective opinion:", "").strip()
                        if clean_claim and "no fact-based claims" not in clean_claim.lower() and "no subjective opinions" not in clean_claim.lower():
                            facts_to_verify.append({"fact_text": clean_claim, "speaker": speaker_id})
                except (ValueError, IndexError):
                    continue # Skip keys that don't match the pattern, like 'winner' or 'transcript'


        if not facts_to_verify:
            return {
                "message": "No specific claims were extracted for detailed analysis.",
                "initial_analysis": initial_analysis
            }

        # --- Stage 3: Run the Detailed, Claim-by-Claim Analysis ---
        final_report = await analyze_facts(facts_to_verify)
        
        # Combine results for a complete response
        # Create a dynamic categorization dictionary
        speaker_categorization = {
            f"speaker{i}_claims": initial_analysis.get(f"speaker{i}")
            for i in range(20) if f"speaker{i}" in initial_analysis # Increased range for more speakers
        }

        full_response = {
            "summary_verdict": final_report.get("final_verdict"),
            "detailed_claim_analysis": final_report.get("detailed_analysis"),
            "original_transcript": initial_analysis.get("transcript"),
            "initial_categorization": {
                **speaker_categorization,
                "logical_fallacies_summary": initial_analysis.get("logical_fallacies")
            }
        }
        return full_response

    except Exception as e:
        # Log the exception for debugging
        print(f"An unexpected server error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Debate Analysis API. Go to /docs to use."}
