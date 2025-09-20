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

def ensure_img_history():
    if "img_history" not in st.session_state:
        st.session_state["img_history"] = []
    return st.session_state["img_history"]

def push_img_history(png_bytes: bytes, max_items: int = 10):
    hist = ensure_img_history()
    hist.append(png_bytes)
    if len(hist) > max_items:
        del hist[: len(hist) - max_items]

def get_last_img_bytes():
    hist = st.session_state.get("img_history") or []
    return hist[-1] if hist else None


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
st.title("🎬 Image and Video Generator")

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
    # Task selection now in sidebar
    mode = st.selectbox(
        "Task",
        options=["Video", "Image"],
        index=0,
        help="Choose whether to generate/edit images or generate videos",
    )

    st.divider()
    st.caption("Models")
    # Video model select box (maps friendly names to model IDs)
    _video_model_options = {
        "Veo 2": "veo-2.0-generate-001",
        "Veo 3 Fast": "veo-3.0-fast-generate-001",
        "Veo 3": "veo-3.0-generate-001",
    }
    _inverse = {v: k for k, v in _video_model_options.items()}
    _current_id = st.session_state.get("model_id", "veo-2.0-generate-001")
    _default_choice = st.session_state.get("video_model_choice") or _inverse.get(_current_id, "Veo 2")
    _choices = list(_video_model_options.keys())
    _default_index = _choices.index(_default_choice) if _default_choice in _choices else 0
    video_model_choice = st.selectbox(
        "Video Model",
        options=_choices,
        index=_default_index,
        help="Select the Veo model to use",
    )
    st.session_state["video_model_choice"] = video_model_choice
    model_id = _video_model_options[video_model_choice]
    st.session_state["model_id"] = model_id
    
    # Image model for generation/editing
    image_model_id = st.text_input(
        "Image Model",
        value=st.session_state.get("image_model_id", "gemini-2.5-flash-image-preview"),
        help="Model for image generation/editing.",
    )
    st.session_state["image_model_id"] = image_model_id
    
    
    st.session_state["mode"] = mode
    
    
    st.divider()
    # st.info("Using minimal parameters for Gemini API compatibility (prompt + optional image). Advanced options are disabled.")


if st.session_state.get("mode","Video") == "Image":
    st.subheader("Image Generation / Editing")
    img_task = st.selectbox("Image Task", ["Generate image", "Edit image"], index=0)
    img_prompt = st.text_area("Image Prompt", value=st.session_state.get("img_prompt", "A photorealistic nano banana dessert on a glossy plate, Gemini theme."))
    st.session_state["img_prompt"] = img_prompt

    img_uploads = []
    if img_task == "Edit image":
        img_uploads = st.file_uploader(
            "Upload one or more images to edit/compose (PNG/JPG)",
            type=["png","jpg","jpeg"],
            accept_multiple_files=True,
            key="image_edit_upload",
        ) or []
        if img_uploads:
            cols = st.columns(min(3, len(img_uploads)))
            for i, up in enumerate(img_uploads):
                try:
                    preview = Image.open(up)
                    with cols[i % len(cols)]:
                        st.image(preview, caption=f"Image {i+1}", use_container_width=True)
                except Exception:
                    st.warning(f"Could not preview image {i+1}.")

    # Multi-turn option: use last generated image as input
    last_img_bytes = get_last_img_bytes()
    use_last = False
    if last_img_bytes is not None:
        use_last = st.checkbox("Use last generated image as input", value=False)
        if use_last:
            try:
                st.image(Image.open(io.BytesIO(last_img_bytes)), caption="Last generated", use_container_width=True)
            except Exception:
                st.caption("(Could not preview last generated image)")
        if st.button("Clear image history"):
            st.session_state["img_history"] = []
            st.experimental_rerun()

    run_img = st.button("Generate Image", type="primary")
    if run_img:
        if not st.session_state.get("api_key"):
            st.error("Please provide your Gemini API key in the sidebar.")
            st.stop()
        try:
            client = make_client(st.session_state.get("api_key"))
        except Exception as e:
            st.error(f"Failed to initialize client: {e}")
            st.stop()
        contents = [img_prompt]
        # Include last generated image if requested
        if use_last and last_img_bytes is not None:
            try:
                contents.append(Image.open(io.BytesIO(last_img_bytes)))
            except Exception as e:
                st.warning(f"Could not add last generated image: {e}")
        if img_task == "Edit image":
            if not img_uploads and not use_last:
                st.error("Please upload at least one image or enable 'Use last generated image'.")
                st.stop()
            for up in img_uploads:
                try:
                    pil_img = Image.open(up)
                    contents.append(pil_img)
                except Exception as e:
                    st.warning(f"Skipping an image that could not be read: {e}")
        with st.status("Generating image...", expanded=True) as sbox:
            try:
                resp = client.models.generate_content(model=st.session_state.get("image_model_id","gemini-2.5-flash-image-preview"), contents=contents)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
                st.stop()
            images = []
            texts = []
            try:
                cand = resp.candidates[0]
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
                    elif getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None) is not None:
                        data = part.inline_data.data
                        try:
                            img = Image.open(io.BytesIO(data))
                            images.append((img, data))
                        except Exception:
                            images.append((None, data))
            except Exception as e:
                st.error(f"Unexpected image response format: {e}")
                st.stop()
            if texts:
                st.info("\n".join(texts))
            if not images:
                st.error("No image returned.")
                st.stop()
            new_history = []
            for idx, (img_obj, data) in enumerate(images, start=1):
                fname = f"generated_image_{idx}.png"
                if img_obj is not None:
                    st.image(img_obj, caption=f"Generated image {idx}", use_container_width=True)
                    buf = io.BytesIO()
                    img_obj.save(buf, format="PNG")
                    b = buf.getvalue()
                else:
                    b = data
                st.download_button(label=f"Download image {idx}", data=b, file_name=fname, mime="image/png")
                new_history.append(b)
            for b in new_history:
                push_img_history(b)
            sbox.update(label="Done", state="complete")
else:
    # Existing video UI moved under Video branch
    prompt = st.text_area("Prompt", value=st.session_state.get("prompt", "Ultra wide drone shot of waves crashing on hawaii."))
    st.session_state["prompt"] = prompt
    uploaded_image = st.file_uploader("Optional: Upload an image to guide the video (JPG/PNG)", type=["png","jpg","jpeg"], accept_multiple_files=False)
    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            st.image(img, caption="Reference image", use_container_width=True)
        except Exception:
            st.warning("Could not preview the uploaded image.")
    generate = st.button("Generate Video", type="primary")
    if generate:
        if not st.session_state.get("api_key"):
            st.error("Please provide your Gemini API key in the sidebar.")
            st.stop()
        if not st.session_state.get("model_id"):
            st.error("Please provide a model id.")
            st.stop()
        if not prompt and uploaded_image is None:
            st.error("Provide a prompt or upload an image.")
            st.stop()
        try:
            client = make_client(st.session_state.get("api_key"))
        except Exception as e:
            st.error(f"Failed to initialize client: {e}")
            st.stop()
        st.info("Submitting generation request...")
        try:
            if uploaded_image is not None:
                img_bytes = bytes_from_uploaded_file(uploaded_image)
                image = genai_types.Image(image_bytes=img_bytes, mime_type=uploaded_image.type or "image/png")
            else:
                image = None
            operation = client.models.generate_videos(model=st.session_state.get("model_id"), prompt=prompt, image=image)
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
            try:
                client.files.download(file=gen_video_entry.video)
            except Exception:
                pass
            gen_video = gen_video_entry.video
            video_bytes = getattr(gen_video, "video_bytes", None)
            video_uri = getattr(gen_video, "uri", None)
            mime_attr = getattr(gen_video, "mime_type", None)
            mime_type = mime_attr or "video/mp4"
            if video_bytes:
                st.success("Video generated!")
                file_path = save_video_bytes(video_bytes, mime_type, prompt)
                st.video(video_bytes, format=mime_type, start_time=0)
                st.download_button(label="Download video", data=video_bytes, file_name=os.path.basename(file_path), mime=mime_type)
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
    "Tip: Use the sidebar to choose Image or Video and set model IDs."
)
