import json
import os
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    ANIMAL_TYPE_MODEL,
    ANIMAL_TYPE_CLASSES,
    DOG_BREED_MODEL,
    DOG_BREED_CLASSES,
    COW_BREED_MODEL,
    COW_BREED_CLASSES,
    CAT_BREED_MODEL,
    CAT_BREED_CLASSES,
    DOG_AGE_MODEL,
    DOG_AGE_CLASSES,
    HORSE_BREED_CLASSES,
    HORSE_BREED_MODEL,
    BUFFALO_BREED_MODEL,
    BUFFALO_BREED_CLASSES,
    IMAGE_SIZE,
    ANIMAL_CONFIDENCE_THRESHOLD,
    ANIMAL_MARGIN_THRESHOLD,
    BREED_CONFIDENCE_THRESHOLD,
    BREED_MARGIN_THRESHOLD,
    AGE_CONFIDENCE_THRESHOLD,
    AGE_MARGIN_THRESHOLD,
)


def compute_entropy(probs: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy of a probability distribution.
    Returns a value between 0.0 (perfectly certain) and 1.0 (completely uniform).
    
    - Known animal  → one class dominates  → low entropy  (e.g. 0.05)
    - Unknown image → scores spread evenly → high entropy (e.g. 0.85)
    """
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))          # entropy of uniform distribution
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_top2_ratio(probs: np.ndarray) -> float:
    """
    Ratio of top-2 scores.  Close to 1.0 means model is confused between two classes.
    e.g. horse:35% camel:30% → ratio = 0.857  (suspicious)
         cat:92%   dog:5%   → ratio = 0.054  (confident)
    """
    sorted_probs = np.sort(probs)[::-1]
    if sorted_probs[0] < 1e-10:
        return 1.0
    return float(sorted_probs[1] / sorted_probs[0])


class ProductionPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models: Dict[str, tf.keras.Model] = {}
        self.class_names: Dict[str, list] = {}

        self.model_paths = {
            ("animal", "type"): ANIMAL_TYPE_MODEL,
            ("dog",    "breed"): DOG_BREED_MODEL,
            ("cow",    "breed"): COW_BREED_MODEL,
            ("cat",    "breed"): CAT_BREED_MODEL,
            ("dog",    "age"):   DOG_AGE_MODEL,
            ("horse",  "breed"): HORSE_BREED_MODEL,
            ("buffalo","breed"): BUFFALO_BREED_MODEL,
        }

        self.class_paths = {
            ("animal", "type"): ANIMAL_TYPE_CLASSES,
            ("dog",    "breed"): DOG_BREED_CLASSES,
            ("cow",    "breed"): COW_BREED_CLASSES,
            ("cat",    "breed"): CAT_BREED_CLASSES,
            ("dog",    "age"):   DOG_AGE_CLASSES,
            ("horse",  "breed"): HORSE_BREED_CLASSES,
            ("buffalo","breed"): BUFFALO_BREED_CLASSES,
        }

        self.thresholds = {
            ("animal", "type"): {
                "confidence": ANIMAL_CONFIDENCE_THRESHOLD,
                "margin":     ANIMAL_MARGIN_THRESHOLD,
            },
            ("dog",    "breed"): {"confidence": BREED_CONFIDENCE_THRESHOLD, "margin": BREED_MARGIN_THRESHOLD},
            ("cow",    "breed"): {"confidence": BREED_CONFIDENCE_THRESHOLD, "margin": BREED_MARGIN_THRESHOLD},
            ("cat",    "breed"): {"confidence": BREED_CONFIDENCE_THRESHOLD, "margin": BREED_MARGIN_THRESHOLD},
            ("dog",    "age"):   {"confidence": AGE_CONFIDENCE_THRESHOLD,   "margin": AGE_MARGIN_THRESHOLD},
            ("horse",  "breed"): {"confidence": BREED_CONFIDENCE_THRESHOLD, "margin": BREED_MARGIN_THRESHOLD},
            ("buffalo","breed"): {"confidence": BREED_CONFIDENCE_THRESHOLD, "margin": BREED_MARGIN_THRESHOLD},
        }

        # ── OOD (out-of-distribution) thresholds ──────────────────────────
        # Tune these if you get false unknowns or missed unknowns:
        #   entropy_threshold : normalized entropy above this → unknown
        #   top2_ratio_threshold: second/first score ratio above this → unknown
        self.ood_thresholds = {
            "animal_type": {
                "confidence":       0.80,   # must be very confident for animal type
                "margin":           0.25,   # big gap between top-2
                "entropy":          0.55,   # if entropy > 0.55 → unknown
                "top2_ratio":       0.55,   # if 2nd/1st > 0.55 → too close → unknown
            },
            "breed": {
                "confidence":       0.75,
                "margin":           0.15,
                "entropy":          0.65,
                "top2_ratio":       0.65,
            },
        }

    # ── internal helpers ───────────────────────────────────────────────────

    def _make_key(self, animal_type: str, task_type: str) -> str:
        return f"{animal_type.lower()}_{task_type.lower()}"

    def _get_ood_thresholds(self, task_type: str) -> dict:
        if task_type == "type":
            return self.ood_thresholds["animal_type"]
        return self.ood_thresholds["breed"]

    # ── public API ─────────────────────────────────────────────────────────

    def load_model(self, animal_type: str, task_type: str) -> bool:
        animal_type = animal_type.lower()
        task_type   = task_type.lower()
        key         = self._make_key(animal_type, task_type)

        model_path = self.model_paths.get((animal_type, task_type))
        class_path = self.class_paths.get((animal_type, task_type))

        if not model_path or not class_path:
            print(f"No model/class mapping found for {key}")
            return False
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        if not os.path.exists(class_path):
            print(f"Class file not found: {class_path}")
            return False

        try:
            model = tf.keras.models.load_model(model_path)
            with open(class_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)

            if not isinstance(class_names, list) or len(class_names) == 0:
                print(f"Invalid class names file: {class_path}")
                return False

            self.models[key]       = model
            self.class_names[key]  = class_names
            print(f"Loaded model: {key} | Classes: {class_names}")
            return True

        except Exception as e:
            print(f"Error loading model {key}: {e}")
            return False

    def set_thresholds(self, animal_type: str, task_type: str, confidence: float, margin: float):
        animal_type = animal_type.lower()
        task_type   = task_type.lower()
        self.thresholds[(animal_type, task_type)] = {
            "confidence": confidence,
            "margin":     margin,
        }

    def _prepare_image(self, image):
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(np.array(image)).convert("RGB")

        pil_img   = pil_img.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(pil_img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(
        self,
        image,
        animal_type: str,
        task_type:   str,
        return_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        animal_type = animal_type.lower()
        task_type   = task_type.lower()
        key         = self._make_key(animal_type, task_type)

        if key not in self.models:
            return {"error": f"Model not loaded for {key}"}

        try:
            img_array  = self._prepare_image(image)
            preds      = self.models[key].predict(img_array, verbose=0)[0]
            class_names = self.class_names[key]

            top_indices  = np.argsort(preds)[::-1]
            best_idx     = int(top_indices[0])
            second_idx   = int(top_indices[1]) if len(top_indices) > 1 else best_idx

            predicted_class = class_names[best_idx]
            confidence      = float(preds[best_idx])
            second_conf     = float(preds[second_idx]) if len(top_indices) > 1 else 0.0
            margin          = confidence - second_conf

            # ── OOD detection ──────────────────────────────────────────────
            entropy    = compute_entropy(preds)
            top2_ratio = compute_top2_ratio(preds)

            ood        = self._get_ood_thresholds(task_type)

            is_uncertain = (
                confidence  < ood["confidence"]  or
                margin      < ood["margin"]       or
                entropy     > ood["entropy"]      or
                top2_ratio  > ood["top2_ratio"]
            )

            # ── build result ───────────────────────────────────────────────
            top_3 = [
                {"class": class_names[int(idx)], "score": float(preds[int(idx)])}
                for idx in top_indices[:3]
            ]
            all_scores = [
                {"class": class_names[int(idx)], "score": float(preds[int(idx)])}
                for idx in top_indices
            ]

            result = {
                "predicted_class": predicted_class,
                "raw_class":       predicted_class,
                "confidence":      confidence,
                "margin":          margin,
                "entropy":         entropy,
                "top2_ratio":      top2_ratio,
                "is_uncertain":    is_uncertain,
                "top_3":           top_3,
                "all_scores":      all_scores,
            }

            if return_uncertainty and is_uncertain:
                reasons = []
                if confidence  < ood["confidence"]:
                    reasons.append(f"low confidence {confidence:.1%} < {ood['confidence']:.1%}")
                if margin      < ood["margin"]:
                    reasons.append(f"low margin {margin:.1%} < {ood['margin']:.1%}")
                if entropy     > ood["entropy"]:
                    reasons.append(f"high entropy {entropy:.2f} > {ood['entropy']:.2f}")
                if top2_ratio  > ood["top2_ratio"]:
                    reasons.append(f"top2 ratio {top2_ratio:.2f} > {ood['top2_ratio']:.2f}")
                result["reason"] = " | ".join(reasons) if reasons else "Prediction uncertain"

            # debug log — helpful for tuning thresholds
            print(
                f"[{key}] class={predicted_class} conf={confidence:.3f} "
                f"margin={margin:.3f} entropy={entropy:.3f} top2_ratio={top2_ratio:.3f} "
                f"uncertain={is_uncertain}"
            )

            return result

        except Exception as e:
            return {"error": f"Prediction failed for {key}: {str(e)}"}