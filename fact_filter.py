# fact_filter.py

import os
import google.generativeai as genai
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def create_dynamic_analysis_prompt(speaker_labels: list) -> str:
    """Creates a Gemini prompt dynamically based on the detected speaker labels."""
    
    # Dynamically create the JSON key examples for the prompt
    speaker_key_examples = "\n".join([f'  "speaker{i}": [...],' for i, _ in enumerate(speaker_labels)])


    prompt = f"""
You are an expert debate analyst. Analyze the following transcript and return results in STRICT JSON.
The speakers in this transcript are: {", ".join(speaker_labels)}

Rules:
1. For each speaker ({", ".join(speaker_labels)}):
   - Create a corresponding JSON key (e.g., "speaker0", "speaker1").
   - Extract all fact-based, verifiable claims into that speaker's key. Prefix them with "Fact-based claim:".
   - Extract all subjective/unverifiable claims into that same key. Prefix them with "Subjective opinion:".
   - If a speaker has none of one type, explicitly include: "No fact-based claims" or "No subjective opinions".
2. For "logical_fallacies": list any fallacies and explain them. If none, return ["No logical fallacies detected"].
3. For "winner":
   - Count fact-based claims for each speaker. The speaker with the most wins.
   - If speakers rely only on subjective claims or have an equal number of fact-based claims, the result is "controversial".
4. Always include the full, original "transcript".

Output format ONLY:
{{
{speaker_key_examples}
  "logical_fallacies": [...],
  "winner": "SPEAKER_00 / SPEAKER_01 / ... / controversial",
  "transcript": "<the full transcript here>"
}}

Now, analyze the following transcript:
"""
    return prompt

def analyze_debate_transcript(transcript: str) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")

    # --- DYNAMIC PROMPT LOGIC ---
    # 1. Find all unique speakers in the transcript (e.g., SPEAKER_00, SPEAKER_01)
    detected_speakers = sorted(list(set(re.findall(r"^(SPEAKER_\d+):", transcript, re.MULTILINE))))
    if not detected_speakers:
        return { "error": "No speakers found in the transcript matching the 'SPEAKER_XX:' format." }
    
    # 2. Generate the prompt tailored to these speakers
    dynamic_prompt = create_dynamic_analysis_prompt(detected_speakers)
    # --- END DYNAMIC PROMPT LOGIC ---

    print(f"Step 2/4: Analyzing standardized debate transcript for speakers: {', '.join(detected_speakers)}...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash') 

    try:
        # We send the DYNAMIC prompt and the transcript
        response = model.generate_content([dynamic_prompt, transcript])
        time.sleep(2)  

        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_text)

        print("  - Debate analysis complete.")
        return data

    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        error_response_text = getattr(response, 'text', 'No response text available.')
        print(f"Error parsing AI response in fact_filter: {e}. Response was: {error_response_text}")

        error_data = {
            f"speaker{i}": [f"Error during analysis: {e}" if i == 0 else f"Raw AI response: {error_response_text}"]
            for i, _ in enumerate(detected_speakers)
        }
        error_data["logical_fallacies"] = ["Failed to parse AI response."]
        error_data["winner"] = "controversial"
        error_data["transcript"] = transcript
        return error_data