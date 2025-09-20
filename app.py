import io
import os
import time
import base64
import pathlib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

try:
    import google.genai as genai
    from google.genai import types as genai_types
except Exception as e:
    genai = None
    genai_types = None


# ---------- Helpers ----------
def load_api_key_from_env():
    load_dotenv()
    # Support common env var names
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )


def make_client(api_key: str):
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. Install dependencies from requirements.txt"
        )
    return genai.Client(api_key=api_key)


def bytes_from_uploaded_file(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    return uploaded_file.getvalue()


def sanitize_filename(text: str, max_len: int = 60) -> str:
    keep = [c for c in text.strip().replace("\n", " ") if c.isalnum() or c in ("-", "_", " ")]
    s = "".join(keep)
    s = "_".join(s.split())
    return s[:max_len] if s else datetime.now().strftime("%Y%m%d_%H%M%S")


def save_video_bytes(data: bytes, mime_type: str, prompt: str) -> str:
    ext = ".mp4"
    if mime_type and "/" in mime_type:
        subtype = mime_type.split("/", 1)[1]
        # Basic mapping to extension
        if subtype == "mp4":
            ext = ".mp4"
        elif subtype == "webm":
            ext = ".webm"
        elif subtype == "x-matroska" or subtype == "mkv":
            ext = ".mkv"
        else:
            ext = ".bin"

    outputs_dir = pathlib.Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitize_filename(prompt)}{ext}"
    fpath = outputs_dir / fname
    with open(fpath, "wb") as f:
        f.write(data)
    return str(fpath)


def poll_operation_until_done(client, operation, poll_interval: float = 5.0):
    # operation is a google.genai.types.GenerateVideosOperation
    # Poll via client.operations.get(operation)
    placeholder = st.empty()
    last_status = None
    while True:
        # Render a small heartbeat
        md = getattr(operation, "metadata", None) or {}
        status = None
        if isinstance(md, dict):
            status = md.get("state") or md.get("progressMessage")
        if status != last_status and status:
            placeholder.info(f"Status: {status}")
            last_status = status

        if getattr(operation, "done", False):
            break
        time.sleep(poll_interval)
        operation = client.operations.get(operation)
    placeholder.empty()
    return operation


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Veo Video Generator", page_icon="🎬", layout="centered")
st.title("🎬 Veo Video Generator (Gemini API)")

st.markdown(
    "Enter your prompt, optionally add an image, and generate a video using Google AI's Veo model."
)

with st.sidebar:
    st.header("Settings")
    default_key = load_api_key_from_env() or ""
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("api_key", default_key),
        help="Stored only in session. You can also set GEMINI_API_KEY or GOOGLE_API_KEY in .env",
    )
    st.session_state["api_key"] = api_key

    st.divider()
    st.caption("Model")
    # The exact model id for Veo 3 fast can vary; allow override.
    model_id = st.text_input(
        "Model ID",
        value=st.session_state.get("model_id", "veo-2.0-generate-001"),
        help=(
            "Enter the exact model name from AI Studio (e.g., 'veo-3.0-generate-001' or a fast variant if available). "
            "Default uses a known Veo 2 ID."
        ),
    )
    st.session_state["model_id"] = model_id
    
    st.divider()
    st.info("Using minimal parameters for Gemini API compatibility (prompt + optional image). Advanced options are disabled.")

prompt = st.text_area(
    "Prompt",
    value=st.session_state.get("prompt", "Ultra wide drone shot of waves crashing on hawaii.")
)
st.session_state["prompt"] = prompt

uploaded_image = st.file_uploader(
    "Optional: Upload an image to guide the video (JPG/PNG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

if uploaded_image is not None:
    try:
        img = Image.open(uploaded_image)
        st.image(img, caption="Reference image", use_column_width=True)
    except Exception:
        st.warning("Could not preview the uploaded image.")

generate = st.button("Generate Video", type="primary")

if generate:
    if not api_key:
        st.error("Please provide your Gemini API key in the sidebar.")
        st.stop()
    if not model_id:
        st.error("Please provide a model id.")
        st.stop()
    if not prompt and uploaded_image is None:
        st.error("Provide a prompt or upload an image.")
        st.stop()

    try:
        client = make_client(api_key)
    except Exception as e:
        st.error(f"Failed to initialize client: {e}")
        st.stop()

    st.info("Submitting generation request...")
    try:
        # Use the simple signature compatible with Gemini API
        if uploaded_image is not None:
            img_bytes = bytes_from_uploaded_file(uploaded_image)
            image = genai_types.Image(
                image_bytes=img_bytes, mime_type=uploaded_image.type or "image/png"
            )
        else:
            image = None

        operation = client.models.generate_videos(
            model=model_id,
            prompt=prompt,
            image=image,
        )
    except Exception as e:
        st.error(f"Generation request failed: {e}")
        st.stop()

    with st.status("Generating video... this can take a bit", expanded=True) as status_box:
        st.write("Polling operation status...")
        try:
            operation = poll_operation_until_done(client, operation, poll_interval=5.0)
        except Exception as e:
            st.error(f"Polling failed: {e}")
            st.stop()

        if getattr(operation, "error", None):
            st.error(f"Operation error: {operation.error}")
            st.stop()

        result = getattr(operation, "response", None) or getattr(operation, "result", None)
        if not result or not getattr(result, "generated_videos", None):
            st.error("No video returned.")
            st.stop()

        gen_video_entry = result.generated_videos[0]
        # Attempt to download bytes (fills video.video_bytes if URI-only)
        try:
            client.files.download(file=gen_video_entry.video)
        except Exception:
            pass

        gen_video = gen_video_entry.video
        video_bytes = getattr(gen_video, "video_bytes", None)
        video_uri = getattr(gen_video, "uri", None)
        # Ensure a valid mime type string; Streamlit expects a non-empty mimetype
        mime_attr = getattr(gen_video, "mime_type", None)
        mime_type = mime_attr or "video/mp4"

        if video_bytes:
            st.success("Video generated!")
            file_path = save_video_bytes(video_bytes, mime_type, prompt)
            st.video(video_bytes, format=mime_type, start_time=0)
            st.download_button(
                label="Download video",
                data=video_bytes,
                file_name=os.path.basename(file_path),
                mime=mime_type,
            )
            st.caption(f"Saved to {file_path}")
        elif video_uri:
            st.success("Video generated (URI provided)!")
            st.write(f"URI: {video_uri}")
            st.info("Downloading from URI is not automated. Please fetch from the URI.")
        else:
            st.error("Video response did not include bytes or URI.")

        status_box.update(label="Done", state="complete")

st.markdown("---")
st.caption(
    "Tip: If you have access to Veo 3 Fast, enter its exact model ID in the sidebar."
)
