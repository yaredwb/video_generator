import io
import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

try:
    import google.genai as genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None


# ---------- Helpers ----------
def load_api_key_from_env():
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )


def make_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai is not installed. Install requirements.txt")
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
    safe = [c for c in text.strip().replace("\n", " ") if c.isalnum() or c in ("-", "_", " ")]
    s = "".join(safe)
    s = "_".join(s.split())
    return s[:max_len] if s else datetime.now().strftime("%Y%m%d_%H%M%S")


def _infer_video_ext(mime_type: str) -> str:
    if not mime_type or "/" not in mime_type:
        return ".mp4"
    subtype = mime_type.split("/", 1)[1]
    return {"mp4": ".mp4", "webm": ".webm", "x-matroska": ".mkv", "mkv": ".mkv"}.get(subtype, ".bin")


def suggest_video_filename(prompt: str, mime_type: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitize_filename(prompt)}{_infer_video_ext(mime_type)}"


def poll_operation_until_done(client, operation, poll_interval: float = 5.0):
    placeholder = st.empty()
    last_status = None
    while True:
        md = getattr(operation, "metadata", None) or {}
        status = None
        if isinstance(md, dict):
            status = md.get("state") or md.get("progressMessage")
        if status and status != last_status:
            placeholder.info(f"Status: {status}")
            last_status = status
        if getattr(operation, "done", False):
            break
        time.sleep(poll_interval)
        operation = client.operations.get(operation)
    placeholder.empty()
    return operation


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Image and Video Generator", page_icon="🎬", layout="centered")
st.title("🎬 Image and Video Generator")

# Dynamic description placeholder, updated after task selection
_desc_ph = st.empty()

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

    # Clear persisted media when switching tasks
    prev_task = st.session_state.get("last_task")
    if prev_task is None:
        st.session_state["last_task"] = mode
    elif prev_task != mode:
        st.session_state["last_task"] = mode
        st.session_state.pop("last_images", None)
        st.session_state.pop("last_video", None)
        st.session_state["img_history"] = []

    st.divider()
    st.caption("Model")
    if mode == "Video":
        # Video model select
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
        video_model_choice = st.selectbox("Video Model", options=_choices, index=_default_index)
        st.session_state["video_model_choice"] = video_model_choice
        st.session_state["model_id"] = _video_model_options[video_model_choice]
    else:
        # Image model select
        _image_model_options = {
            "Gemini 2.5 Flash Image": "gemini-2.5-flash-image-preview",
            "Imagen 4": "imagen-4.0-generate-001",
        }
        _img_inverse = {v: k for k, v in _image_model_options.items()}
        _img_current_id = st.session_state.get("image_model_id", "gemini-2.5-flash-image-preview")
        _img_default_choice = st.session_state.get("image_model_choice") or _img_inverse.get(_img_current_id, "Gemini 2.5 Flash Image")
        _img_choices = list(_image_model_options.keys())
        _img_default_index = _img_choices.index(_img_default_choice) if _img_default_choice in _img_choices else 0
        image_model_choice = st.selectbox("Image Model", options=_img_choices, index=_img_default_index)
        st.session_state["image_model_choice"] = image_model_choice
        st.session_state["image_model_id"] = _image_model_options[image_model_choice]

    # Update page description
    if mode == "Image":
        _desc_ph.markdown(
            "Enter your prompt, optionally add one or more images, and generate or edit images with Gemini 2.5 Flash Image, or generate with Imagen 4."
        )
    else:
        _desc_ph.markdown(
            "Enter your prompt, optionally add an image, and generate a video using Google AI's Veo model."
        )


if mode == "Image":
    st.subheader("Image Generation / Editing")
    current_image_model = st.session_state.get("image_model_id", "gemini-2.5-flash-image-preview")

    # Task options depend on model
    allowed_tasks = ["Generate image"] if current_image_model.startswith("imagen-") else ["Generate image", "Edit image"]
    img_task = st.selectbox("Image Task", allowed_tasks, index=0)
    img_prompt = st.text_area("Image Prompt", value=st.session_state.get("img_prompt", "A photorealistic nano banana dessert on a glossy plate, Gemini theme."))
    st.session_state["img_prompt"] = img_prompt

    # Imagen-specific configuration
    imagen_num_images = None
    imagen_aspect = None
    imagen_size = None
    if current_image_model.startswith("imagen-"):
        cols_opts = st.columns(3)
        with cols_opts[0]:
            imagen_num_images = st.number_input("Number of images", min_value=1, max_value=4, value=4)
        with cols_opts[1]:
            imagen_aspect = st.selectbox("Aspect ratio", ["1:1", "3:4", "4:3", "9:16", "16:9"], index=4)
        with cols_opts[2]:
            imagen_size = st.selectbox("Image size", ["1K", "2K"], index=0)

    # Optional uploads (Gemini editing)
    img_uploads = []
    if img_task == "Edit image":
        img_uploads = st.file_uploader(
            "Upload one or more images to edit/compose (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
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

    # Multi-turn option using last generated (Gemini only)
    last_img_bytes = get_last_img_bytes()
    use_last = False
    if (not current_image_model.startswith("imagen-")) and last_img_bytes is not None:
        use_last = st.checkbox("Use last generated image as input", value=False)
        if use_last:
            try:
                st.image(Image.open(io.BytesIO(last_img_bytes)), caption="Last generated", use_container_width=True)
            except Exception:
                st.caption("(Could not preview last generated image)")
        if st.button("Clear image history"):
            st.session_state["img_history"] = []
            st.rerun()

    # Show persisted last images if available
    if st.session_state.get("last_images"):
        imgs = st.session_state.get("last_images")
        ncols = 3
        cols = st.columns(ncols)
        for i, b in enumerate(imgs):
            with cols[i % ncols]:
                try:
                    st.image(Image.open(io.BytesIO(b)), use_container_width=True)
                except Exception:
                    st.caption(f"Image {i+1}")
                st.download_button(label=f"Download image {i+1}", data=b, file_name=f"generated_image_{i+1}.png", mime="image/png", key=f"dl_last_img_{i}")

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

        images = []
        texts = []
        try:
            if current_image_model.startswith("imagen-"):
                cfg_kwargs = {}
                if imagen_num_images:
                    cfg_kwargs["number_of_images"] = int(imagen_num_images)
                if imagen_aspect:
                    cfg_kwargs["aspect_ratio"] = imagen_aspect
                if imagen_size:
                    cfg_kwargs["image_size"] = imagen_size
                resp = client.models.generate_images(
                    model=current_image_model,
                    prompt=img_prompt,
                    config=genai_types.GenerateImagesConfig(**cfg_kwargs) if cfg_kwargs else None,
                )
                for gi in getattr(resp, "generated_images", []) or []:
                    try:
                        client.files.download(file=gi.image)
                    except Exception:
                        pass
                    img_bytes = getattr(gi.image, "image_bytes", None)
                    if img_bytes:
                        try:
                            pil = Image.open(io.BytesIO(img_bytes))
                            images.append((pil, img_bytes))
                        except Exception:
                            images.append((None, img_bytes))
            else:
                contents = [img_prompt]
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
                            contents.append(Image.open(up))
                        except Exception as e:
                            st.warning(f"Skipping an image that could not be read: {e}")
                resp = client.models.generate_content(model=current_image_model, contents=contents)
                cand = resp.candidates[0]
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
                    elif getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None) is not None:
                        data = part.inline_data.data
                        try:
                            pil = Image.open(io.BytesIO(data))
                            images.append((pil, data))
                        except Exception:
                            images.append((None, data))
        except Exception as e:
            st.error(f"Image generation failed: {e}")
            st.stop()

        if texts:
            st.info("\n".join(texts))
        if not images:
            st.error("No image returned.")
            st.stop()

        # Display images in a grid and persist
        last_images = []
        ncols = 3
        cols = st.columns(ncols)
        for idx, (img_obj, data) in enumerate(images, start=1):
            if img_obj is not None:
                buf = io.BytesIO()
                img_obj.save(buf, format="PNG")
                b = buf.getvalue()
            else:
                b = data
            last_images.append(b)
            with cols[(idx - 1) % ncols]:
                try:
                    st.image(Image.open(io.BytesIO(b)), caption=f"Generated image {idx}", use_container_width=True)
                except Exception:
                    st.caption(f"Generated image {idx}")
                st.download_button(label=f"Download image {idx}", data=b, file_name=f"generated_image_{idx}.png", mime="image/png", key=f"dl_img_{idx}")

        st.session_state["last_images"] = last_images
        for b in last_images:
            push_img_history(b)

else:
    # Video mode
    # Show persisted last video if available
    if st.session_state.get("last_video"):
        lv = st.session_state["last_video"]
        st.video(lv.get("bytes"), format=lv.get("mime", "video/mp4"), start_time=0)
        st.download_button(label="Download last video", data=lv.get("bytes"), file_name=suggest_video_filename(lv.get("prompt", "video"), lv.get("mime", "video/mp4")), mime=lv.get("mime", "video/mp4"), key="dl_last_video")

    prompt = st.text_area("Prompt", value=st.session_state.get("prompt", "Ultra wide drone shot of waves crashing on hawaii."))
    st.session_state["prompt"] = prompt
    uploaded_image = st.file_uploader("Optional: Upload an image to guide the video (JPG/PNG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if uploaded_image is not None:
        try:
            st.image(Image.open(uploaded_image), caption="Reference image", use_container_width=True)
        except Exception:
            st.warning("Could not preview the uploaded image.")

    generate = st.button("Generate Video", type="primary")
    if generate:
        if not st.session_state.get("api_key"):
            st.error("Please provide your Gemini API key in the sidebar.")
            st.stop()
        if not st.session_state.get("model_id"):
            st.error("Please select a video model in the sidebar.")
            st.stop()
        if not prompt and uploaded_image is None:
            st.error("Provide a prompt or upload an image.")
            st.stop()
        try:
            client = make_client(st.session_state.get("api_key"))
        except Exception as e:
            st.error(f"Failed to initialize client: {e}")
            st.stop()
        try:
            image = None
            if uploaded_image is not None:
                img_bytes = bytes_from_uploaded_file(uploaded_image)
                image = genai_types.Image(image_bytes=img_bytes, mime_type=uploaded_image.type or "image/png")

            operation = client.models.generate_videos(
                model=st.session_state.get("model_id"),
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
            try:
                client.files.download(file=gen_video_entry.video)
            except Exception:
                pass
            gen_video = gen_video_entry.video
            video_bytes = getattr(gen_video, "video_bytes", None)
            mime_type = getattr(gen_video, "mime_type", None) or "video/mp4"
            video_uri = getattr(gen_video, "uri", None)
            if video_bytes:
                st.success("Video generated!")
                st.video(video_bytes, format=mime_type, start_time=0)
                st.download_button(label="Download video", data=video_bytes, file_name=suggest_video_filename(prompt, mime_type), mime=mime_type)
                st.session_state["last_video"] = {"bytes": video_bytes, "mime": mime_type, "prompt": prompt}
            elif video_uri:
                st.success("Video generated (URI provided)!")
                st.write(f"URI: {video_uri}")
            else:
                st.error("Video response did not include bytes or URI.")
            status_box.update(label="Done", state="complete")

st.markdown("---")
st.caption("Tip: Use the sidebar to choose Image or Video and set model IDs.")
