import json
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import torch

from tomato_pipeline import run_pipeline


st.set_page_config(page_title="Tomato Quality Checker", layout="wide")
st.title("🍅 Tomato Detection + Good/Bad Classification")

with st.sidebar:
    st.header("Model Settings")

    detector_upload = st.file_uploader("Upload detector (.pt)", type=["pt"])
    classifier_upload = st.file_uploader("Upload classifier (.pth)", type=["pth"])

    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    img_size = st.number_input("Classifier image size", 64, 1024, 224, 32)
    det_conf = st.slider("Detection threshold", 0.0, 1.0, 0.25, 0.01)
    labels_csv = st.text_input("Labels (index order)", "good,bad")

if device == "cuda" and not torch.cuda.is_available():
    st.warning("CUDA not available, using CPU")
    device = "cpu"

uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

can_run = (
    uploaded_image is not None
    and detector_upload is not None
    and classifier_upload is not None
)

if st.button("Run Inference", disabled=not can_run):

    labels = [l.strip() for l in labels_csv.split(",")]
    if len(labels) != 2:
        st.error("Provide exactly 2 labels.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        image_path = tmpdir / uploaded_image.name
        detector_path = tmpdir / detector_upload.name
        classifier_path = tmpdir / classifier_upload.name
        output_path = tmpdir / "output.jpg"

        image_path.write_bytes(uploaded_image.getvalue())
        detector_path.write_bytes(detector_upload.getvalue())
        classifier_path.write_bytes(classifier_upload.getvalue())

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
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(uploaded_image, use_container_width=True)
        with col2:
            st.subheader("Output")
            st.image(annotated, use_container_width=True)

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
