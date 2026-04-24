import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# =========================
# MODEL PATHS
# =========================
DOG_BREED_MODEL = os.path.join(MODELS_DIR, "dog_breed_model.keras")
DOG_BREED_CLASSES = os.path.join(MODELS_DIR, "dog_breed_classes.json")

COW_BREED_MODEL = os.path.join(MODELS_DIR, "cow_breed_model.keras")
COW_BREED_CLASSES = os.path.join(MODELS_DIR, "cow_breed_classes.json")

CAT_BREED_MODEL = os.path.join(MODELS_DIR, "cat_breed_model.keras")
CAT_BREED_CLASSES = os.path.join(MODELS_DIR, "cat_breed_classes.json")

ANIMAL_TYPE_MODEL = os.path.join(MODELS_DIR, "animal_type_model.keras")
ANIMAL_TYPE_CLASSES = os.path.join(MODELS_DIR, "animal_type_classes.json")

DOG_AGE_MODEL = os.path.join(MODELS_DIR, "dog_age_model.keras")
DOG_AGE_CLASSES = os.path.join(MODELS_DIR, "dog_age_classes.json")

HORSE_BREED_MODEL = os.path.join(MODELS_DIR, "horse_breed_model.keras")
HORSE_BREED_CLASSES = os.path.join(MODELS_DIR, "horse_breed_classes.json")

BUFFALO_BREED_MODEL = os.path.join(MODELS_DIR, "buffalo_breed_model.keras")
BUFFALO_BREED_CLASSES = os.path.join(MODELS_DIR, "buffalo_breed_classes.json")

# =========================
# INFO JSON PATHS
# =========================
DOG_BREED_INFO = os.path.join(DATA_DIR, "dog_breed.json")
COW_BREED_INFO = os.path.join(DATA_DIR, "cow_breed.json")
CAT_BREED_INFO = os.path.join(DATA_DIR, "cat_breed.json")
HORSE_BREED_INFO = os.path.join(DATA_DIR, "horse_breed.json")
BUFFALO_BREED_INFO = os.path.join(DATA_DIR, "buffalo_breed.json")

DOG_AGE_INFO = os.path.join(DATA_DIR, "dog_age.json")

# =========================
# COMMON SETTINGS
# =========================
IMAGE_SIZE = (224, 224)

ANIMAL_CONFIDENCE_THRESHOLD = 0.70
ANIMAL_MARGIN_THRESHOLD = 0.20

BREED_CONFIDENCE_THRESHOLD = 0.75
BREED_MARGIN_THRESHOLD = 0.15

AGE_CONFIDENCE_THRESHOLD = 0.60
AGE_MARGIN_THRESHOLD = 0.10