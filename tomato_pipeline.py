import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models
from ultralytics import YOLO


@dataclass
class DetectionResult:
    bbox_xyxy: Tuple[int, int, int, int]
    detector_confidence: float
    quality_label: str
    quality_confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tomato detection + quality classification pipeline")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--detector", required=True, type=Path, help="Path to tomato detection .pt model")
    parser.add_argument("--classifier", required=True, type=Path, help="Path to quality classifier .pth model")
    parser.add_argument("--output", default=Path("annotated_output.jpg"), type=Path, help="Annotated output image")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--img-size", default=224, type=int, help="Classifier input size")
    parser.add_argument("--labels", nargs=2, default=["good", "bad"], help="Classifier labels in index order")
    parser.add_argument("--det-conf", default=0.25, type=float, help="Detection confidence threshold")
    return parser.parse_args()


def build_classifier_model(num_classes: int = 2, architecture: str = "resnet18") -> torch.nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported classifier architecture: {architecture}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def _infer_classifier_architecture(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    if "conv1.weight" in keys or any(k.startswith("layer1.") for k in keys):
        return "resnet18"
    if any(k.startswith("features.") for k in keys) and any(k.startswith("classifier.") for k in keys):
        return "efficientnet_b0"
    return "resnet18"


def load_classifier(classifier_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(classifier_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError("Unsupported .pth format. Checkpoint state_dict is not a dictionary.")

        state_dict = _normalize_state_dict_keys(state_dict)
        architecture = _infer_classifier_architecture(state_dict)
        model = build_classifier_model(num_classes=2, architecture=architecture)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Could not load classifier weights. "
                "Detected architecture may not match checkpoint. "
                "Please export full model (`torch.save(model, ...)`) or adjust build_classifier_model()."
            ) from exc
    else:
        raise ValueError("Unsupported .pth format. Provide a full model or a state_dict checkpoint.")

    model.to(device)
    model.eval()
    return model


def make_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def classify_crop(crop_bgr: np.ndarray, model: torch.nn.Module, transform: T.Compose, device: str, labels: List[str]) -> Tuple[str, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)

    idx_val = int(idx.item())
    return labels[idx_val], float(conf.item())


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


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
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    detector = YOLO(str(detector_path))
    classifier = load_classifier(classifier_path, device=device)
    transform = make_transform(img_size)

    det_results = detector.predict(source=str(image_path), conf=det_conf, device=device, verbose=False)
    if not det_results:
        return []

    boxes = det_results[0].boxes
    if boxes is None:
        return []

    output = image.copy()
    final_results: List[DetectionResult] = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        det_score = float(box.conf[0].item()) if box.conf is not None else 0.0
        quality_label, quality_conf = classify_crop(crop, classifier, transform, device, labels)

        result = DetectionResult(
            bbox_xyxy=(x1, y1, x2, y2),
            detector_confidence=det_score,
            quality_label=quality_label,
            quality_confidence=quality_conf,
        )
        final_results.append(result)

        text = f"{quality_label} ({quality_conf:.2f})"
        color = (0, 180, 0) if quality_label.lower() == "good" else (0, 0, 220)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), output)
    return final_results


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    results = run_pipeline(
        image_path=args.image,
        detector_path=args.detector,
        classifier_path=args.classifier,
        output_path=args.output,
        device=args.device,
        img_size=args.img_size,
        labels=args.labels,
        det_conf=args.det_conf,
    )

    serializable = [
        {
            "bbox_xyxy": r.bbox_xyxy,
            "detector_confidence": round(r.detector_confidence, 4),
            "quality_label": r.quality_label,
            "quality_confidence": round(r.quality_confidence, 4),
        }
        for r in results
    ]

    print(json.dumps({"num_tomatoes": len(results), "results": serializable}, indent=2))


if __name__ == "__main__":
    main()
