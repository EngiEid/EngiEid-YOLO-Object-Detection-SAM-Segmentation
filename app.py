from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import ImageDraw

from main import YOLOSAMPipeline, load_image


st.set_page_config(
    page_title="YOLO + SAM Vision Studio",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 YOLO Object Detection + SAM Segmentation")
st.caption(
    "Detect objects with YOLO, then precisely segment a selected object with SAM "
    "using a point or box prompt."
)


@st.cache_resource(show_spinner="Loading YOLO and SAM models...")
def get_pipeline(yolo_weights: str, sam_weights: str, device: Optional[str]):
    return YOLOSAMPipeline(yolo_weights, sam_weights, device)


def resize_for_selection(image: np.ndarray, max_width: int = 1000):
    """Resize only the interactive copy; keep the original for SAM inference."""
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w)
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def make_preview(image: np.ndarray, max_width: int = 420) -> np.ndarray:
    """Create a compact preview for the three-column result gallery."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    preview_h = max(1, int(h * max_width / w))
    return cv2.resize(image, (max_width, preview_h), interpolation=cv2.INTER_AREA)


def to_original_point(point, scale: float):
    x, y = point
    return float(x / scale), float(y / scale)


def to_original_box(box, scale: float):
    x1, y1, x2, y2 = box
    return (
        float(x1 / scale),
        float(y1 / scale),
        float(x2 / scale),
        float(y2 / scale),
    )


def draw_selection_image(image: np.ndarray, mode: str, points: list[tuple[int, int]]):
    """Draw the current point/box selection on the image shown to the user."""
    pil = YOLOSAMPipeline.image_to_pil(image).copy()
    draw = ImageDraw.Draw(pil)

    if mode == "Point" and points:
        x, y = points[-1]
        r = 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 50, 50), outline=(255, 255, 255), width=2)

    elif mode == "Box" and points:
        if len(points) == 1:
            x, y = points[0]
            r = 8
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 50, 50), outline=(255, 255, 255), width=2)
        elif len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            # ImageDraw requires the second corner to be at or below/right
            # of the first one. The user can click the corners in ANY order,
            # so normalize the coordinates before drawing.
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            draw.rectangle(
                (left, top, right, bottom),
                outline=(0, 255, 0),
                width=4,
            )

    return pil


def event_id(value):
    """Return a stable identifier for a component click event."""
    if value is None:
        return None
    return value.get("time", (value.get("x"), value.get("y")))


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    yolo_weights = st.text_input(
        "YOLO weights",
        value="yolo26n.pt",
        help="Use a custom trained .pt file if you have one.",
    )
    sam_weights = st.text_input(
        "SAM weights",
        value="sam_b.pt",
        help="Ultralytics SAM checkpoint.",
    )
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    device_arg = None if device == "auto" else device

    conf = st.slider(
        "YOLO confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    st.divider()
    st.markdown(
        "**Workflow**\n"
        "1. Upload image\n"
        "2. Run YOLO detection\n"
        "3. Choose Point or Box\n"
        "4. Click the object / two box corners\n"
        "5. Run SAM segmentation"
    )


# ---------------- Upload ----------------
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

# Reset prompt state whenever a new image is uploaded.
image_id = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("image_id") != image_id:
    st.session_state.image_id = image_id
    st.session_state.prompt_points = []
    st.session_state.last_event_id = None
    st.session_state.sam_result = None
    st.session_state.sam_mask = None
    st.session_state.yolo_results = None
    st.session_state.canvas_key = 0

image = load_image(uploaded)
display_image, scale = resize_for_selection(image)

# ---------------- YOLO ----------------
st.subheader("1️⃣ YOLO Detection")

if st.session_state.get("yolo_results") is None:
    if st.button("🔎 Run YOLO Detection", type="primary"):
        with st.spinner("Running YOLO..."):
            pipeline = get_pipeline(yolo_weights, sam_weights, device_arg)
            st.session_state.yolo_results = pipeline.detect(image, conf=conf)
        st.rerun()
else:
    if st.button("🔄 Run YOLO Again"):
        with st.spinner("Running YOLO..."):
            pipeline = get_pipeline(yolo_weights, sam_weights, device_arg)
            st.session_state.yolo_results = pipeline.detect(image, conf=conf)
        st.rerun()

if st.session_state.get("yolo_results") is not None:
    yolo_results = st.session_state.yolo_results
    annotated = YOLOSAMPipeline.yolo_annotated_image(image, yolo_results)
    st.session_state.annotated_image = annotated

    # Keep the previews compact and side-by-side instead of stacking large images.
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**Original**")
        st.image(
            cv2.cvtColor(make_preview(image), cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
    with col2:
        st.markdown("**YOLO Detection**")
        st.image(
            cv2.cvtColor(make_preview(annotated), cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
else:
    st.image(
        cv2.cvtColor(make_preview(image), cv2.COLOR_BGR2RGB),
        caption="Original",
        use_container_width=False,
    )
    st.info("Click **Run YOLO Detection** first.")

# ---------------- Prompt selection ----------------
st.subheader("2️⃣ Select an object for SAM")

mode = st.radio(
    "Prompt type",
    ["Point", "Box"],
    horizontal=True,
    key="prompt_mode",
)

if mode == "Point":
    st.info("🖱️ Click once inside the object you want SAM to segment.")
else:
    st.info("🖱️ Click two corners: first the top-left, then the bottom-right of the object.")

# Clear prompt button
clear_col, status_col = st.columns([1, 3])
with clear_col:
    if st.button("🗑️ Clear Selection"):
        st.session_state.prompt_points = []
        st.session_state.last_event_id = None
        st.session_state.canvas_key += 1
        st.rerun()

points = st.session_state.get("prompt_points", [])
selection_image = draw_selection_image(display_image, mode, points)

# streamlit-image-coordinates is a small custom component that only returns click
# coordinates; unlike streamlit-drawable-canvas, it does not use Streamlit's removed
# internal image_to_url API.
click_value = streamlit_image_coordinates(
    selection_image,
    key=f"image_coordinates_{st.session_state.canvas_key}_{mode}",
)

current_event_id = event_id(click_value)
if click_value is not None and current_event_id != st.session_state.last_event_id:
    x = int(click_value["x"])
    y = int(click_value["y"])
    st.session_state.last_event_id = current_event_id

    if mode == "Point":
        st.session_state.prompt_points = [(x, y)]
    else:
        if len(st.session_state.prompt_points) >= 2:
            st.session_state.prompt_points = [(x, y)]
        else:
            st.session_state.prompt_points = [*st.session_state.prompt_points, (x, y)]

    st.rerun()

points = st.session_state.get("prompt_points", [])

if mode == "Point" and len(points) == 1:
    original_point = to_original_point(points[0], scale)
    st.success(
        f"Selected point: ({original_point[0]:.1f}, {original_point[1]:.1f})"
    )
    prompt = ("point", original_point)

elif mode == "Box" and len(points) == 2:
    x1, y1 = points[0]
    x2, y2 = points[1]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    original_box = to_original_box((left, top, right, bottom), scale)
    st.success(
        "Selected box: "
        f"({original_box[0]:.1f}, {original_box[1]:.1f}, "
        f"{original_box[2]:.1f}, {original_box[3]:.1f})"
    )
    prompt = ("box", original_box)
else:
    prompt = None

# ---------------- SAM ----------------
st.subheader("3️⃣ SAM Segmentation")

if st.button("✂️ Run SAM Segmentation", type="primary", disabled=prompt is None):
    pipeline = get_pipeline(yolo_weights, sam_weights, device_arg)

    with st.spinner("Running SAM segmentation..."):
        if prompt[0] == "point":
            sam_results = pipeline.segment_from_point(image, prompt[1])
        else:
            sam_results = pipeline.segment_from_box(image, prompt[1])

        mask = pipeline.first_mask(sam_results)

    if mask is None:
        st.error("SAM did not return a mask for this prompt.")
    else:
        st.session_state.sam_result = pipeline.overlay_mask(image, mask)
        st.session_state.sam_mask = mask

if st.session_state.get("sam_result") is not None:
    st.subheader("4️⃣ Results")

    # Final result gallery: Original | YOLO Detection | SAM Segmentation
    result_col1, result_col2, result_col3 = st.columns(3, gap="medium")

    original_preview = make_preview(image)
    detected_preview = make_preview(
        st.session_state.get("annotated_image", image)
    )
    segmented_preview = make_preview(st.session_state.sam_result)

    with result_col1:
        st.markdown("**Original**")
        st.image(
            cv2.cvtColor(original_preview, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with result_col2:
        st.markdown("**YOLO Detection**")
        st.image(
            cv2.cvtColor(detected_preview, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with result_col3:
        st.markdown("**SAM Segmentation**")
        st.image(
            cv2.cvtColor(segmented_preview, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    mask = st.session_state.sam_mask
    mask_png = mask.astype(np.uint8) * 255
    ok, encoded = cv2.imencode(".png", mask_png)

    if ok:
        st.download_button(
            "⬇️ Download Mask",
            data=encoded.tobytes(),
            file_name="sam_mask.png",
            mime="image/png",
        )
