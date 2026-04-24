import os
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps, ImageEnhance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_model(num_classes: int, architecture: str = "efficientnet_b0"):
    architecture = architecture.lower()

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, num_classes)
        )

    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, num_classes)
        )

    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def load_model_bundle(model_path: str, class_path: str, architecture: str = "efficientnet_b0"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Class file not found: {class_path}")

    with open(class_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid class file: {class_path}")

    model = build_model(len(class_names), architecture=architecture)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # support both plain state_dict and checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model, class_names


def _prepare_variants(img: Image.Image) -> List[Image.Image]:
    img = img.convert("RGB")

    bright = ImageEnhance.Brightness(img).enhance(1.08)
    contrast = ImageEnhance.Contrast(img).enhance(1.08)

    return [
        img,
        ImageOps.mirror(img),
        ImageOps.autocontrast(img),
        bright,
        contrast,
    ]


def _predict_probs_single(img: Image.Image, model, transform) -> torch.Tensor:
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    return probs.detach().cpu()


def predict_single_image(
    img: Image.Image,
    model,
    class_names: List[str],
    confidence_threshold: float = 0.70,
    margin_threshold: float = 0.15
) -> Dict:
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL Image object")

    transform = get_inference_transform()
    variants = _prepare_variants(img)

    probs_list = []
    for variant in variants:
        probs = _predict_probs_single(variant, model, transform)
        probs_list.append(probs)

    mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)

    top_values, top_indices = torch.topk(mean_probs, k=min(3, len(class_names)))
    top_values = top_values.tolist()
    top_indices = top_indices.tolist()

    predicted_index = int(top_indices[0])
    confidence = float(top_values[0])
    predicted_class = class_names[predicted_index]

    second_confidence = float(top_values[1]) if len(top_values) > 1 else 0.0
    margin = confidence - second_confidence

    all_confidences = {
        class_names[i]: float(mean_probs[i].item())
        for i in range(len(class_names))
    }

    top_predictions = [
        (class_names[idx], float(score))
        for idx, score in zip(top_indices, top_values)
    ]

    is_uncertain = confidence < confidence_threshold or margin < margin_threshold

    return {
        "predicted_class": "Uncertain" if is_uncertain else predicted_class,
        "raw_predicted_class": predicted_class,
        "confidence": confidence,
        "margin": margin,
        "second_confidence": second_confidence,
        "all_confidences": all_confidences,
        "top_predictions": top_predictions,
    }