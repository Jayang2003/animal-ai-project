import json
import os

from config import (
    DOG_BREED_INFO,
    COW_BREED_INFO,
    CAT_BREED_INFO,
    DOG_AGE_INFO,
    HORSE_BREED_INFO,
    BUFFALO_BREED_INFO,
)


class AnimalKnowledgeBase:
    def __init__(self):
        self.data = {
            "dog_breed": self._load_json(DOG_BREED_INFO),
            "cow_breed": self._load_json(COW_BREED_INFO),
            "cat_breed": self._load_json(CAT_BREED_INFO),
            "dog_age": self._load_json(DOG_AGE_INFO),
            "horse_breed": self._load_json(HORSE_BREED_INFO),
            "buffalo_breed": self._load_json(BUFFALO_BREED_INFO),
        }

    def _load_json(self, path):
        abs_path = os.path.abspath(path)
        print("Loading JSON from:", abs_path)

        if not os.path.exists(path):
            print("File not found:", abs_path)
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict):
                normalized_data = {}
                for k, v in loaded.items():
                    normalized_key = str(k).strip().lower().replace(" ", "_").replace("-", "_")
                    normalized_data[normalized_key] = v
                return normalized_data

            return {}

        except Exception as e:
            print(f"Error loading JSON {abs_path}: {e}")
            return {}

    def get_info(self, animal_type, task_type, label):
        key = f"{str(animal_type).strip().lower()}_{str(task_type).strip().lower()}"
        dataset = self.data.get(key, {})

        if not dataset or not label:
            print(f"No dataset found for key: {key}")
            return {}

        normalized_label = str(label).strip().lower().replace(" ", "_").replace("-", "_")

        if normalized_label in dataset:
            return dataset[normalized_label]

        for k, v in dataset.items():
            normalized_key = str(k).strip().lower().replace(" ", "_").replace("-", "_")
            if normalized_key == normalized_label:
                return v

        print(f"Label '{label}' not found in dataset '{key}'")
        print("Available keys:", list(dataset.keys()))
        return {}