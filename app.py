import json
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import torch

from tomato_pipeline import run_pipeline


st.set_page_config(page_title="Tomato Quality Checker", layout="wide")
st.title("🍅 Tomato Detection + Good/Bad Classification")
st.write("Upload your models (`.pt` + `.pth`) and a tomato image, then run inference.")

with st.sidebar:
    st.header("Model Settings")
    source_mode = st.radio("Model source", options=["Upload model files", "Use server file paths"], index=0)

    if source_mode == "Upload model files":
        detector_upload = st.file_uploader("Upload detector model (.pt)", type=["pt"], key="detector_upload")
        classifier_upload = st.file_uploader("Upload classifier model (.pth)", type=["pth"], key="classifier_upload")
        detector_path_input = None
        classifier_path_input = None
    else:
        detector_upload = None
        classifier_upload = None
        detector_path_input = st.text_input("Detector (.pt) path", value="tomato_detection.pt")
        classifier_path_input = st.text_input("Classifier (.pth) path", value="tomato_quality.pth")

    device = st.selectbox(
        "Device",
        options=["cpu", "cuda"],
        index=0,
        help="Use cuda only if GPU is available",
    )
    img_size = st.number_input("Classifier image size", min_value=64, max_value=1024, value=224, step=32)
    det_conf = st.slider("Detection confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    labels_csv = st.text_input("Labels (index order)", value="good,bad")

if device == "cuda" and not torch.cuda.is_available():
    st.warning("CUDA selected but no GPU is available. Falling back to CPU.")
    device = "cpu"

uploaded_image = st.file_uploader("Upload tomato image", type=["jpg", "jpeg", "png"], key="input_image")

if source_mode == "Upload model files":
    can_run = uploaded_image is not None and detector_upload is not None and classifier_upload is not None
else:
    can_run = uploaded_image is not None and bool(detector_path_input) and bool(classifier_path_input)

run_button = st.button("Run Inference", type="primary", disabled=not can_run)

if run_button and uploaded_image is not None:
    labels = [label.strip() for label in labels_csv.split(",") if label.strip()]
    if len(labels) != 2:
        st.error("Please provide exactly 2 labels in order, e.g. `good,bad`.")
        st.stop()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / uploaded_image.name
            output_path = tmpdir_path / "annotated.jpg"

            image_path.write_bytes(uploaded_image.getvalue())

            if source_mode == "Upload model files":
                detector_path = tmpdir_path / detector_upload.name
                classifier_path = tmpdir_path / classifier_upload.name
                detector_path.write_bytes(detector_upload.getvalue())
                classifier_path.write_bytes(classifier_upload.getvalue())
            else:
                detector_path = Path(detector_path_input)
                classifier_path = Path(classifier_path_input)

            results = run_pipeline(
                image_path=image_path,
                detector_path=detector_path,
                classifier_path=classifier_path,
                output_path=output_path,
                device=device,
                img_size=int(img_size),
                labels=labels,
                det_conf=float(det_conf),
            )

            annotated = cv2.imread(str(output_path))
            if annotated is None:
                st.error("Inference ran, but failed to load annotated output image.")
                st.stop()

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input")
                st.image(uploaded_image, use_container_width=True)
            with col2:
                st.subheader("Output")
                st.image(annotated_rgb, use_container_width=True)

            payload = {
                "num_tomatoes": len(results),
                "results": [
                    {
                        "bbox_xyxy": r.bbox_xyxy,
                        "detector_confidence": round(r.detector_confidence, 4),
                        "quality_label": r.quality_label,
                        "quality_confidence": round(r.quality_confidence, 4),
                    }
                    for r in results
                ],
            }

            st.subheader("Results JSON")
            st.code(json.dumps(payload, indent=2), language="json")

    except Exception as exc:
        st.error(f"Error: {exc}")
