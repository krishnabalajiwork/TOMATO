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
    page_title="Tomato AI Inspector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# MODERN CSS
# -------------------------------------------------
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.block-container{
padding-top:2rem;
}

.big-title{
font-size:48px;
font-weight:700;
}

.subtitle{
font-size:20px;
color:#6b7280;
margin-bottom:30px;
}

.card{
background:#111827;
padding:20px;
border-radius:12px;
box-shadow:0 10px 25px rgba(0,0,0,0.3);
}

.metric-card{
background:#1f2937;
padding:20px;
border-radius:10px;
text-align:center;
}

button[kind="primary"]{
height:50px;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="big-title">🍅 Tomato AI Quality Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLO Detection + EfficientNet Classification</div>', unsafe_allow_html=True)

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

    st.title("⚙ Settings")

    device = st.selectbox(
        "Compute Device",
        ["cpu", "cuda"]
    )

    img_size = st.slider(
        "Classifier Resolution",
        128,
        512,
        224
    )

    det_conf = st.slider(
        "Detection Confidence",
        0.1,
        1.0,
        0.25
    )

    labels_csv = st.text_input(
        "Class Labels",
        "bad,good"
    )

    st.divider()

    st.caption("Model Stack")
    st.write("YOLO Detector")
    st.write("EfficientNet Classifier")


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
# INPUT PANEL
# -------------------------------------------------
left, right = st.columns([1,1])


with left:

    st.subheader("📥 Input")

    mode = st.radio(
        "Image Source",
        ["Upload Image","Camera"],
        horizontal=True
    )

    uploaded_image=None
    camera_image=None

    if mode=="Upload Image":

        uploaded_image=st.file_uploader(
            "Upload Tomato Image",
            type=["jpg","jpeg","png"]
        )

    else:

        camera_image=st.camera_input("Capture Image")


    run = st.button(
        "Run AI Detection",
        type="primary",
        use_container_width=True
    )


with right:

    st.subheader("📊 Detection Stats")

    stat1,stat2,stat3=st.columns(3)

    stat1.metric("Total","-")
    stat2.metric("Good","-")
    stat3.metric("Bad","-")


st.divider()


# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
if run:

    if uploaded_image is None and camera_image is None:
        st.error("Provide an image first.")
        st.stop()

    labels=[l.strip() for l in labels_csv.split(",")]

    if uploaded_image:
        image_bytes=uploaded_image.getvalue()
        input_display=uploaded_image
    else:
        image_bytes=camera_image.getvalue()
        input_display=camera_image


    image_np=cv2.imdecode(
        np.frombuffer(image_bytes,np.uint8),
        cv2.IMREAD_COLOR
    )

    h,w=image_np.shape[:2]

    output=image_np.copy()

    transform=make_transform(img_size)

    detections=detector.predict(
        source=image_np,
        conf=float(det_conf),
        device=device,
        verbose=False
    )

    results=[]
    good_count=0
    bad_count=0


    if detections and detections[0].boxes is not None:

        for box in detections[0].boxes:

            x1,y1,x2,y2=box.xyxy[0].tolist()

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
                (x1,y1-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )


            results.append({
                "bbox":[x1,y1,x2,y2],
                "label":quality_label,
                "confidence":float(quality_conf)
            })


    output_rgb=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)


    # -------------------------------------------------
    # METRICS UPDATE
    # -------------------------------------------------
    stat1.metric("Total",len(results))
    stat2.metric("Good",good_count)
    stat3.metric("Bad",bad_count)


    # -------------------------------------------------
    # RESULT PANEL
    # -------------------------------------------------
    img1,img2=st.columns(2)

    with img1:
        st.subheader("Input Image")
        st.image(input_display,use_container_width=True)

    with img2:
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
