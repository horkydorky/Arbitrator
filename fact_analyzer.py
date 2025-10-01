# in fact_analyzer.py

import os
import google.generativeai as genai
import json
import asyncio
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configure the AI model once ---
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-2.0-flash') 


WEB_VERIFICATION_PROMPT_TEMPLATE = """
You are a meticulous fact-checker. Based *only* on the provided search results, analyze the following statement.
Statement: "{claim}"
Search Results:
---
{search_results}
---
1.  **Classification:** Classify the statement as 'Likely True', 'Likely False', or 'Uncertain/Controversial'.
2.  **Confidence Score:** Provide a confidence score for your classification on a scale of 1 (low confidence) to 10 (high confidence).
3.  **Explanation:** Provide a brief, one-sentence explanation for your decision.
Format your response as a JSON object with the keys "classification", "confidence_score", and "explanation". Your response must be ONLY the JSON object.
"""
FALLACY_PROMPT_TEMPLATE = "You are a specialist in rhetoric and logical reasoning. Analyze the following statement to determine if it contains a common logical fallacy.\nStatement to analyze: \"{claim}\"\n- If a fallacy is present, name the fallacy and provide a brief, one-sentence explanation.\n- If no clear logical fallacy is detected, respond with 'No significant fallacy detected'."
BIAS_PROMPT_TEMPLATE = "You are an analyst specializing in cognitive and rhetorical bias. Analyze the following statement for potential bias (e.g., Confirmation Bias, Appeal to Emotion, Loaded Language).\nStatement: \"{claim}\"\n- If bias is present, name the type of bias and explain it in one sentence.\n- If no significant bias is detected, respond with 'No significant bias detected'."
DEBATE_WINNER_PROMPT_TEMPLATE = """
You are an impartial and expert debate judge. Based on the provided JSON analysis report, determine which speaker presented a more logical, evidence-based, and coherent argument.
Criteria for your judgment:
1.  **Use of Verified Facts:** Who supported their arguments with claims that were likely true?
2.  **Logical Soundness:** Who avoided using logical fallacies and cognitive biases in their reasoning?
3.  **Argumentative Strength:** Whose claims, taken as a whole, form a more compelling case?
Announce a winner (e.g., "Winner: SPEAKER_00") and provide a 3-4 sentence justification for your decision, referencing specific strengths or weaknesses from the report.
Analysis Report:
{report}
"""

async def async_analyze_claim(fact_obj: dict, semaphore: asyncio.Semaphore) -> dict:
    """Analyzes a single claim concurrently, respecting the semaphore."""
    fact_text = fact_obj['fact_text']
    speaker = fact_obj['speaker']
    
    # --- NEW: Guard against empty or whitespace-only claims ---
    if not fact_text or not fact_text.strip():
        return {
            "speaker": speaker, "claim": fact_text, "verification": {"classification": "Skipped", "explanation": "Claim was empty."},
            "logical_fallacy": "N/A", "bias_detection": "N/A"
        }

    # --- MODIFIED: Use the semaphore to limit concurrency ---
    async with semaphore:
        # A) Verify with web search (this is a synchronous call, so we run it in an executor)
        loop = asyncio.get_running_loop()
        try:
            search_results_list = await loop.run_in_executor(None, lambda: DDGS().text(fact_text, max_results=3))
            search_results_text = "\n".join([r['body'] for r in search_results_list])
        except Exception as e:
            print(f"  - DuckDuckGo search failed for claim '{fact_text[:30]}...': {e}")
            search_results_text = "Search failed or returned no results."


        # B) Set up all three API calls to run in parallel
        verification_task = MODEL.generate_content_async(WEB_VERIFICATION_PROMPT_TEMPLATE.format(claim=fact_text, search_results=search_results_text))
        fallacy_task = MODEL.generate_content_async(FALLACY_PROMPT_TEMPLATE.format(claim=fact_text))
        bias_task = MODEL.generate_content_async(BIAS_PROMPT_TEMPLATE.format(claim=fact_text))

        # C) Await all API calls at the same time
        responses = await asyncio.gather(verification_task, fallacy_task, bias_task, return_exceptions=True)

        # D) Process the results, now checking for exceptions
        try:
            if isinstance(responses[0], Exception):
                raise responses[0]
            verification_text = responses[0].text.strip().replace("```json", "").replace("```", "")
            verification_result = json.loads(verification_text)
        except (json.JSONDecodeError, IndexError, Exception) as e:
            error_text = getattr(responses[0], 'text', str(e))
            verification_result = {"classification": "Error parsing response", "confidence_score": 0, "explanation": error_text}
            
        fallacy_result = responses[1].text.strip() if len(responses) > 1 and not isinstance(responses[1], Exception) else "Error"
        bias_result = responses[2].text.strip() if len(responses) > 2 and not isinstance(responses[2], Exception) else "Error"

    return {
        "speaker": speaker,
        "claim": fact_text,
        "verification": verification_result,
        "logical_fallacy": fallacy_result,
        "bias_detection": bias_result
    }

async def analyze_facts(facts_list: list) -> dict:
    """
    Performs a deep-dive analysis of each claim concurrently with rate limiting.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
        
    print(f"Step 3/4: Concurrently analyzing and verifying {len(facts_list)} claims...")

    # --- NEW: Create a semaphore to limit concurrent API calls to 5 at a time ---
    semaphore = asyncio.Semaphore(5)

    # Create a list of analysis tasks to run in parallel
    analysis_tasks = [async_analyze_claim(fact_obj, semaphore) for fact_obj in facts_list]
    
    # Run all tasks concurrently and wait for them all to finish
    analysis_report = await asyncio.gather(*analysis_tasks)

    print("  - Performing final meta-analysis...")
    report_json_string = json.dumps(analysis_report, indent=2)
    winner_response = await MODEL.generate_content_async(DEBATE_WINNER_PROMPT_TEMPLATE.format(report=report_json_string))
    winner_prediction = winner_response.text.strip()
    
    return {
        "detailed_analysis": analysis_report,
        "final_verdict": winner_prediction
    }