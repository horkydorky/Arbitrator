# AI Debate Video Analysis Engine



An AI-powered FastAPI application that ingests a debate video and produces a detailed, multi-layered analysis of the conversation. It transcribes the audio, separates speakers, identifies key claims, fact-checks them against web search results, and analyzes them for logical fallacies and cognitive bias.

## Core Features

-   **Full Video-to-Text Pipeline**: Extracts audio, transcribes it with high accuracy using OpenAI Whisper, and saves it for analysis.
-   **Speaker Diarization**: Automatically detects and separates different speakers in the video using Pyannote, attributing text to the correct person.
-   **Dynamic Claim Extraction**: Uses Google Gemini to intelligently parse the transcript and distinguish between verifiable, fact-based claims and subjective opinions for any number of detected speakers.
-   **Concurrent Fact-Checking**: For each extracted claim, it performs a web search via DuckDuckGo and uses Gemini to classify its truthfulness based on the search results.
-   **Rhetorical Analysis**: Each claim is concurrently analyzed for common logical fallacies and cognitive biases, providing deeper insight into the speakers' arguments.
-   **Final Verdict Generation**: A final AI-powered meta-analysis determines a "winner" based on the evidence, logical soundness, and overall argumentative strength.
-   **Asynchronous API**: Built with FastAPI for high-performance, non-blocking request handling.

## Technology Stack

-   **Backend**: FastAPI, Uvicorn
-   **AI & Machine Learning**:
    -   **LLM**: Google Gemini (`gemini-2.0-flash`)
    -   **Transcription**: OpenAI Whisper
    -   **Speaker Diarization**: Pyannote Audio 3.1
-   **Web Verification**: `duckduckgo-search`
-   **Audio/Video Processing**: FFmpeg
-   **Configuration**: `python-dotenv`
-   **Concurrency**: Python `asyncio`

## System Architecture Flow

The application processes a video through a multi-stage pipeline:

1.  **Video Upload**: User uploads a video file via the API endpoint.
2.  **Audio Extraction**: `FFmpeg` extracts a standardized `.wav` audio file from the video.
3.  **Transcription & Diarization**:
    -   `Whisper` transcribes the entire audio file with word-level timestamps.
    -   `Pyannote` identifies unique speakers and their speaking segments.
4.  **Transcript Assembly**: The word timestamps are aligned with the speaker segments to create a single, speaker-labeled transcript (e.g., `SPEAKER_00: ...`).
5.  **Initial Analysis (Claim Filtering)**: The full transcript is sent to `Gemini` to extract fact-based claims and subjective opinions for each speaker.
6.  **Deep-Dive Analysis (Concurrent Fan-out)**:
    -   The list of extracted claims is processed asynchronously.
    -   For **each** claim, three parallel tasks are launched:
        1.  **Web Verification**: Search DuckDuckGo for evidence.
        2.  **LLM Analysis 1**: `Gemini` classifies the claim's truthfulness based on search results.
        3.  **LLM Analysis 2**: `Gemini` checks for logical fallacies.
        4.  **LLM Analysis 3**: `Gemini` checks for cognitive bias.
7.  **Final Report Aggregation**:
    -   The results of all deep-dive analyses are collected into a single JSON report.
    -   This report is sent to `Gemini` one last time for a final verdict on the debate winner.
8.  **API Response**: The complete JSON object, including the final verdict, detailed analysis, and original transcript, is returned to the user.

## Setup and Installation

### Prerequisites

-   Python 3.9+
-   [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system's PATH.
-   Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Create a `requirements.txt` file** with the following content:
    ```
    fastapi
    uvicorn[standard]
    python-dotenv
    google-generativeai
    openai-whisper
    pyannote.audio
    torch
    torchaudio
    duckduckgo-search
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your API keys:
    ```.env
    GEMINI_API_KEY="your_google_gemini_api_key"
    HUGGING_FACE_TOKEN="your_hugging_face_user_access_token"
    ```
    *You can get a Hugging Face token from your [Hugging Face profile settings](https://huggingface.co/settings/tokens) to use the `pyannote/speaker-diarization-3.1` model.*

## How to Run

Start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive documentation at `http://127.0.0.1:8000/docs`.

## API Usage

### Endpoint: `/analyze-video/`

-   **Method**: `POST`
-   **Description**: Upload a video file to start the full analysis pipeline.
-   **Form Data**:
    -   `video_file`: The video file to be analyzed (e.g., `.mp4`, `.mov`).
    -   `num_speakers` (optional): An integer specifying the exact number of speakers to look for. If omitted, the model will detect the number automatically.

#### Example `cURL` Request

```bash
curl -X POST "http://127.0.0.1:8000/analyze-video/" \
-H "Content-Type: multipart/form-data" \
-F "video_file=@/path/to/your/debate_video.mp4" \
-F "num_speakers=2"
```

### Sample JSON Output

The API returns a detailed JSON object containing the full analysis, similar to the structure below.

```json
{
  "summary_verdict": "Winner: SPEAKER_00...",
  "detailed_claim_analysis": [
    {
      "speaker": "SPEAKER_00",
      "claim": "Iran is taking over Iran.",
      "verification": {
        "classification": "Uncertain/Controversial",
        "confidence_score": 7,
        "explanation": "..."
      },
      "logical_fallacy": "No significant fallacy detected.",
      "bias_detection": "No significant bias detected."
    }
  ],
  "original_transcript": "SPEAKER_02: wants open borders now...",
  "initial_categorization": {
    "speaker0_claims": [
      "Fact-based claim: Iran is taking over Iran.",
      "Subjective opinion: we don't gain anything."
    ],
    "logical_fallacies_summary": [
      "Ad hominem: \"It's such a nasty woman.\"..."
    ]
  }
}
```
