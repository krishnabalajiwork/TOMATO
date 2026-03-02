import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from tomato_pipeline import run_pipeline, load_classifier
from ultralytics import YOLO


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Tomato Quality Checker", layout="wide")
st.title("🍅 Tomato Detection + Good/Bad Classification")


# -------------------------------------------------
# Sidebar Settings
# -------------------------------------------------
with st.sidebar:
    st.header("Model Settings")

    detector_upload = st.file_uploader("Upload detector (.pt)", type=["pt"])
    classifier_upload = st.file_uploader("Upload classifier (.pth)", type=["pth"])

    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    img_size = st.number_input("Classifier image size", 64, 1024, 224, 32)
    det_conf = st.slider("Detection threshold", 0.0, 1.0, 0.25, 0.01)
    labels_csv = st.text_input("Labels (index order)", "good,bad")


if device == "cuda" and not torch.cuda.is_available():
    st.warning("CUDA not available, switching to CPU")
    device = "cpu"


# -------------------------------------------------
# Load Models (cached)
# -------------------------------------------------
@st.cache_resource
def load_models(detector_bytes, classifier_bytes, device):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        detector_path = tmpdir / "detector.pt"
        classifier_path = tmpdir / "classifier.pth"

        detector_path.write_bytes(detector_bytes)
        classifier_path.write_bytes(classifier_bytes)

        detector = YOLO(str(detector_path))
        classifier = load_classifier(classifier_path, device)

    return detector, classifier


# -------------------------------------------------
# Input Mode Selection
# -------------------------------------------------
mode = st.radio("Choose Input Mode", ["Upload Image", "Live Camera"])

uploaded_image = None
camera_image = None

if mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload tomato image", type=["jpg", "jpeg", "png"])

else:
    camera_image = st.camera_input("Take a photo")


# -------------------------------------------------
# Run Inference
# -------------------------------------------------
can_run = (
    (uploaded_image is not None or camera_image is not None)
    and detector_upload is not None
    and classifier_upload is not None
)

if st.button("Run Inference", disabled=not can_run):

    labels = [l.strip() for l in labels_csv.split(",")]
    if len(labels) != 2:
        st.error("Provide exactly 2 labels, e.g. good,bad")
        st.stop()

    # Load models once
    detector, classifier = load_models(
        detector_upload.getvalue(),
        classifier_upload.getvalue(),
        device,
    )

    # Prepare image
    if uploaded_image:
        image_bytes = uploaded_image.getvalue()
        input_display = uploaded_image
    else:
        image_bytes = camera_image.getvalue()
        input_display = camera_image

    image_np = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )

    h, w = image_np.shape[:2]

    detections = detector.predict(
        source=image_np,
        conf=float(det_conf),
        device=device,
        verbose=False,
    )

    output = image_np.copy()
    results = []

    for box in detections[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Classification
        from tomato_pipeline import make_transform, classify_crop

        transform = make_transform(int(img_size))
        quality_label, quality_conf = classify_crop(
            crop,
            classifier,
            transform,
            device,
            labels,
        )

        det_score = float(box.conf[0])

        results.append(
            {
                "bbox_xyxy": (x1, y1, x2, y2),
                "detector_confidence": round(det_score, 4),
                "quality_label": quality_label,
                "quality_confidence": round(quality_conf, 4),
            }
        )

        color = (0, 200, 0) if quality_label == "good" else (0, 0, 255)
        text = f"{quality_label} ({quality_conf:.2f})"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(input_display, width="stretch")

    with col2:
        st.subheader("Output")
        st.image(output_rgb, width="stretch")

    st.subheader("Results JSON")
    st.code(json.dumps({
        "num_tomatoes": len(results),
        "results": results
    }, indent=2), language="json")
