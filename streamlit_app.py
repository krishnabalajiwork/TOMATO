import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from ultralytics import YOLO
from tomato_pipeline import load_classifier, make_transform, classify_crop

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Tomato Sorting System",
    page_icon="🍅",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (LEVEL-10 UI)
# -------------------------------------------------
st.markdown("""
<style>

.main {
background: linear-gradient(135deg,#0f172a,#1e293b);
}

h1 {
text-align:center;
color:#ff4b4b;
}

.card {
background-color:#1e293b;
padding:25px;
border-radius:15px;
box-shadow:0px 6px 20px rgba(0,0,0,0.4);
}

.metric-card{
background:#111827;
padding:20px;
border-radius:12px;
text-align:center;
font-size:20px;
}

.stButton>button {
background: linear-gradient(90deg,#ff4b4b,#ff7b7b);
color:white;
border-radius:10px;
height:55px;
font-size:18px;
border:none;
}

.stButton>button:hover {
background: linear-gradient(90deg,#ff7b7b,#ff4b4b);
}

footer {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
"""
<h1>🍅 AI Tomato Sorting System</h1>
<p style='text-align:center;font-size:18px;color:lightgray'>
Real-time Tomato Detection and Quality Classification using YOLO + EfficientNet
</p>
""",
unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# MODEL PATHS
# -------------------------------------------------
DETECTOR_PATH = Path("best.pt")
CLASSIFIER_PATH = Path("efficientnet_b0_best.pth")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    st.header("⚙️ Settings")

    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

    img_size = st.number_input(
        "Classifier Image Size",
        64,
        1024,
        224,
        32
    )

    det_conf = st.slider(
        "Detection Threshold",
        0.0,
        1.0,
        0.25,
        0.01
    )

    labels_csv = st.text_input(
        "Labels",
        "bad,good"
    )

    st.divider()

    st.markdown("### System Info")
    st.write("Model: YOLO + EfficientNet")
    st.write("Interface: Streamlit")

if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models(device):

    detector = YOLO(str(DETECTOR_PATH))
    classifier = load_classifier(CLASSIFIER_PATH, device)

    return detector, classifier


detector, classifier = load_models(device)

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
st.subheader("📥 Input Image")

mode = st.radio(
    "Select Input Source",
    ["Upload Image", "Camera"],
    horizontal=True
)

uploaded_image = None
camera_image = None

if mode == "Upload Image":

    uploaded_image = st.file_uploader(
        "Upload tomato image",
        type=["jpg","jpeg","png"]
    )

else:

    camera_image = st.camera_input(
        "Take a photo"
    )

run = st.button("🚀 Run AI Detection", use_container_width=True)

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
if run:

    if uploaded_image is None and camera_image is None:
        st.error("Please upload or capture an image")
        st.stop()

    labels = [l.strip() for l in labels_csv.split(",")]

    if uploaded_image:
        image_bytes = uploaded_image.getvalue()
        input_display = uploaded_image
    else:
        image_bytes = camera_image.getvalue()
        input_display = camera_image

    image_np = cv2.imdecode(
        np.frombuffer(image_bytes,np.uint8),
        cv2.IMREAD_COLOR
    )

    h,w = image_np.shape[:2]

    output = image_np.copy()

    transform = make_transform(int(img_size))

    detections = detector.predict(
        source=image_np,
        conf=float(det_conf),
        device=device,
        verbose=False
    )

    results = []

    good_count = 0
    bad_count = 0

    if detections and detections[0].boxes is not None:

        for box in detections[0].boxes:

            x1,y1,x2,y2 = box.xyxy[0].tolist()

            x1=max(0,min(int(x1),w-1))
            y1=max(0,min(int(y1),h-1))
            x2=max(0,min(int(x2),w-1))
            y2=max(0,min(int(y2),h-1))

            crop=image_np[y1:y2,x1:x2]

            if crop.size==0:
                continue

            quality_label,quality_conf=classify_crop(
                crop,
                classifier,
                transform,
                device,
                labels
            )

            if quality_label=="good":
                good_count+=1
                color=(0,200,0)
            else:
                bad_count+=1
                color=(0,0,255)

            text=f"{quality_label} {quality_conf:.2f}"

            cv2.rectangle(output,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                output,
                text,
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            results.append({
                "bbox":(x1,y1,x2,y2),
                "label":quality_label,
                "confidence":float(quality_conf)
            })

    output_rgb=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)

    st.divider()

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    col1,col2,col3=st.columns(3)

    col1.metric("Total Tomatoes",len(results))
    col2.metric("Good Tomatoes",good_count)
    col3.metric("Bad Tomatoes",bad_count)

    st.divider()

    # -------------------------------------------------
    # IMAGE DISPLAY
    # -------------------------------------------------
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(input_display,use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(output_rgb,use_container_width=True)

    st.divider()

    st.subheader("Detection JSON")

    st.code(
        json.dumps(
            {
                "total":len(results),
                "good":good_count,
                "bad":bad_count,
                "detections":results
            },
            indent=2
        ),
        language="json"
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
"""
<hr>
<center style='color:gray'>
AI Tomato Sorting System | YOLO Detection + EfficientNet Classification
</center>
""",
unsafe_allow_html=True
)
