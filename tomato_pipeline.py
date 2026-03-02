import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models
from ultralytics import YOLO


# -----------------------------
# Data Structure
# -----------------------------
@dataclass
class DetectionResult:
    bbox_xyxy: Tuple[int, int, int, int]
    detector_confidence: float
    quality_label: str
    quality_confidence: float


# -----------------------------
# Build EfficientNet Classifier
# -----------------------------
def build_classifier_model(num_classes: int = 2) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


# -----------------------------
# Load Classifier
# -----------------------------
def load_classifier(classifier_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(classifier_path, map_location=device)

    # If full model saved
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint

    # If state_dict saved
    else:
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format.")

        model = build_classifier_model(num_classes=2)

        # Remove 'module.' prefix if exists
        cleaned_state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }

        model.load_state_dict(cleaned_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


# -----------------------------
# Image Transform
# -----------------------------
def make_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# -----------------------------
# Classify One Crop
# -----------------------------
@torch.no_grad()
def classify_crop(
    crop_bgr: np.ndarray,
    model: torch.nn.Module,
    transform: T.Compose,
    device: str,
    labels: List[str],
) -> Tuple[str, float]:

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    tensor = transform(pil_img).unsqueeze(0).to(device)
    logits = model(tensor)

    probs = torch.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)

    return labels[int(idx)], float(conf.item())


# -----------------------------
# Clip Bounding Box
# -----------------------------
def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline(
    image_path: Path,
    detector_path: Path,
    classifier_path: Path,
    output_path: Path,
    device: str,
    img_size: int,
    labels: List[str],
    det_conf: float,
) -> List[DetectionResult]:

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError("Could not read image")

    h, w = image.shape[:2]

    detector = YOLO(str(detector_path))
    classifier = load_classifier(classifier_path, device)
    transform = make_transform(img_size)

    detections = detector.predict(
        source=str(image_path),
        conf=det_conf,
        device=device,
        verbose=False,
    )

    if not detections or detections[0].boxes is None:
        return []

    output = image.copy()
    results: List[DetectionResult] = []

    for box in detections[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        det_score = float(box.conf[0])
        quality_label, quality_conf = classify_crop(
            crop, classifier, transform, device, labels
        )

        results.append(
            DetectionResult(
                bbox_xyxy=(x1, y1, x2, y2),
                detector_confidence=det_score,
                quality_label=quality_label,
                quality_confidence=quality_conf,
            )
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

    cv2.imwrite(str(output_path), output)
    return results
