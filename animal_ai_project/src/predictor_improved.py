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


class ProductionPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models: Dict[str, tf.keras.Model] = {}
        self.class_names: Dict[str, list] = {}

        self.model_paths = {
            ("animal", "type"): ANIMAL_TYPE_MODEL,
            ("dog", "breed"): DOG_BREED_MODEL,
            ("cow", "breed"): COW_BREED_MODEL,
            ("cat", "breed"): CAT_BREED_MODEL,
            ("dog", "age"): DOG_AGE_MODEL,
            ("horse", "breed"): HORSE_BREED_MODEL,
            ("buffalo", "breed"): BUFFALO_BREED_MODEL,
        }

        self.class_paths = {
            ("animal", "type"): ANIMAL_TYPE_CLASSES,
            ("dog", "breed"): DOG_BREED_CLASSES,
            ("cow", "breed"): COW_BREED_CLASSES,
            ("cat", "breed"): CAT_BREED_CLASSES,
            ("dog", "age"): DOG_AGE_CLASSES,
            ("horse", "breed"): HORSE_BREED_CLASSES,
            ("buffalo", "breed"): BUFFALO_BREED_CLASSES,
        }

        self.thresholds = {
            ("animal", "type"): {
                "confidence": ANIMAL_CONFIDENCE_THRESHOLD,
                "margin": ANIMAL_MARGIN_THRESHOLD,
            },
            ("dog", "breed"): {
                "confidence": BREED_CONFIDENCE_THRESHOLD,
                "margin": BREED_MARGIN_THRESHOLD,
            },
            ("cow", "breed"): {
                "confidence": BREED_CONFIDENCE_THRESHOLD,
                "margin": BREED_MARGIN_THRESHOLD,
            },
            ("cat", "breed"): {
                "confidence": BREED_CONFIDENCE_THRESHOLD,
                "margin": BREED_MARGIN_THRESHOLD,
            },
            ("dog", "age"): {
                "confidence": AGE_CONFIDENCE_THRESHOLD,
                "margin": AGE_MARGIN_THRESHOLD,
            },
            ("horse", "breed"): {
                "confidence": BREED_CONFIDENCE_THRESHOLD,
                "margin": BREED_MARGIN_THRESHOLD,
            },
            ("buffalo", "breed"): {
                "confidence": BREED_CONFIDENCE_THRESHOLD,
                "margin": BREED_MARGIN_THRESHOLD,
            },
        }

    def _make_key(self, animal_type: str, task_type: str) -> str:
        return f"{animal_type.lower()}_{task_type.lower()}"

    def load_model(self, animal_type: str, task_type: str) -> bool:
        animal_type = animal_type.lower()
        task_type = task_type.lower()

        key = self._make_key(animal_type, task_type)

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

            self.models[key] = model
            self.class_names[key] = class_names

            print(f"Loaded model: {key}")
            print(f"Classes: {class_names}")
            return True

        except Exception as e:
            print(f"Error loading model {key}: {e}")
            return False

    def set_thresholds(self, animal_type: str, task_type: str, confidence: float, margin: float):
        animal_type = animal_type.lower()
        task_type = task_type.lower()
        self.thresholds[(animal_type, task_type)] = {
            "confidence": confidence,
            "margin": margin,
        }

    def _prepare_image(self, image):
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(np.array(image)).convert("RGB")

        pil_img = pil_img.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(pil_img)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(
        self,
        image,
        animal_type: str,
        task_type: str,
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        animal_type = animal_type.lower()
        task_type = task_type.lower()
        key = self._make_key(animal_type, task_type)

        if key not in self.models:
            return {"error": f"Model not loaded for {key}"}

        try:
            img_array = self._prepare_image(image)
            preds = self.models[key].predict(img_array, verbose=0)[0]

            class_names = self.class_names[key]

            top_indices = np.argsort(preds)[::-1]
            best_idx = int(top_indices[0])
            second_idx = int(top_indices[1]) if len(top_indices) > 1 else best_idx

            predicted_class = class_names[best_idx]
            confidence = float(preds[best_idx])
            second_conf = float(preds[second_idx]) if len(top_indices) > 1 else 0.0
            margin = confidence - second_conf

            thresholds = self.thresholds.get(
                (animal_type, task_type),
                {"confidence": 0.5, "margin": 0.1}
            )

            is_uncertain = (
                confidence < thresholds["confidence"] or
                margin < thresholds["margin"]
            )

            top_3 = []
            for idx in top_indices[:3]:
                top_3.append({
                    "class": class_names[int(idx)],
                    "score": float(preds[int(idx)])
                })

            all_scores = []
            for idx in top_indices:
                all_scores.append({
                    "class": class_names[int(idx)],
                    "score": float(preds[int(idx)])
                })

            result = {
                "predicted_class": predicted_class,
                "raw_class": predicted_class,
                "confidence": confidence,
                "margin": margin,
                "is_uncertain": is_uncertain,
                "top_3": top_3,
                "all_scores": all_scores,
            }

            if return_uncertainty and is_uncertain:
                reason_parts = []
                if confidence < thresholds["confidence"]:
                    reason_parts.append(
                        f"confidence {confidence:.1%} is below threshold {thresholds['confidence']:.1%}"
                    )
                if margin < thresholds["margin"]:
                    reason_parts.append(
                        f"margin {margin:.1%} is below threshold {thresholds['margin']:.1%}"
                    )
                result["reason"] = " and ".join(reason_parts) if reason_parts else "Prediction uncertain"

            return result

        except Exception as e:
            return {"error": f"Prediction failed for {key}: {str(e)}"}