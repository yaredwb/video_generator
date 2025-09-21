Veo Video Generator (Streamlit)

A simple Streamlit web app to generate videos using Google's Veo model via the Gemini API. Enter a text prompt, optionally upload an image, and download the resulting video.

Features
- Text-to-video with Veo
- Optional image-to-video conditioning
- Image generation and editing with Gemini 2.5 Flash
  - Single-image edits or multi-image composition/style transfer
- Minimal parameters for Gemini API compatibility (prompt + optional image(s))
- Download generated media; no persistent storage by default

Prerequisites
- Python 3.10+
- A valid Gemini API key with access to Veo. You can get one from Google AI Studio. Make sure your key has access to the Veo model you intend to use (e.g., Veo 3 Fast).

Setup

1. Create and activate a virtual environment (optional but recommended):
   - Windows PowerShell:
     py -3.13 -m venv .venv
     .venv\Scripts\Activate.ps1
   - macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Provide your API key in one of the following ways:
   - Set environment variable `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
   - Create a `.env` file in the project root:
     GEMINI_API_KEY=your_api_key_here
   - Or paste it into the app sidebar at runtime.

Run
streamlit run app.py

Open the URL Streamlit prints (usually http://localhost:8501).

Model ID
The app defaults to `veo-2.0-generate-001` (known working example). If you have access to Veo 3 Fast, replace the model ID in the sidebar with the exact model name from AI Studio (e.g., `veo-3.*-fast-...`).

Notes
- Video generation is a long-running operation; the app polls until completion.
- FPS, audio, compression, resolution, and aspect ratio are not configurable via the Gemini API in this app; model defaults are used.
- If the API returns a URI instead of bytes, the app will show it; downloading from the URI is not automated here.
- Storage efficiency: downloads are served from memory and the app avoids writing to disk on Streamlit Cloud by default.

Troubleshooting
- If you see `API key not valid` errors, verify the key and that it has Veo access.
- If generation fails, try shorter prompts or a different model ID.
- Ensure you are on the latest `google-genai` package.
