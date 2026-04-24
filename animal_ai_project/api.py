from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

from src.predictor_improved import ProductionPredictor
from src.knowledge_base import AnimalKnowledgeBase

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ProductionPredictor(models_dir="models")
kb = AnimalKnowledgeBase()

print("Loading models...")
print("animal_type:", predictor.load_model("animal", "type"))
print("dog_breed:", predictor.load_model("dog", "breed"))
print("cat_breed:", predictor.load_model("cat", "breed"))
print("cow_breed:", predictor.load_model("cow", "breed"))
print("horse_breed:", predictor.load_model("horse", "breed"))
print("buffalo_breed:", predictor.load_model("buffalo", "breed"))
print("dog_age:", predictor.load_model("dog", "age"))


def normalize_label(value: str) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def is_unknown_animal(result: dict) -> bool:
    confidence = float(result.get("confidence", 0) or 0)
    margin = float(result.get("margin", 0) or 0)

    print("Animal confidence:", confidence)
    print("Animal margin:", margin)

    return confidence < 0.55 or margin < 0.10


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    username: str | None = None
    session_id: str | None = None
    history: list | None = None
    context: dict | None = None


@app.get("/")
def home():
    return {"message": "Animal API running"}


@app.post("/login")
def login(data: LoginRequest):
    if data.username == "admin@gmail.com" and data.password == "123":
        return {"success": True, "message": "Login successful"}
    return {"success": False, "message": "Invalid credentials"}


@app.post("/chat")
def chat_api(data: ChatRequest):
    try:
        message = (data.message or "").strip().lower()
        context = data.context or {}

        animal = normalize_label(context.get("animal", ""))
        breed = normalize_label(context.get("breed", ""))
        age = context.get("age", "")

        if not animal or animal == "unknown":
            return {
                "response": "Please upload a supported animal image first so I can answer correctly."
            }

        breed_info = kb.get_info(animal, "breed", breed) if breed else {}

        if not breed_info:
            return {
                "response": "I could not find breed details for the current animal. Please upload the image again."
            }

        if "food" in message or "eat" in message or "diet" in message or "feed" in message:
            return {
                "response": f"**Food:** {breed_info.get('food', 'No food information available.')}"
            }

        if "care" in message or "groom" in message or "maintain" in message:
            return {
                "response": f"**Care:** {breed_info.get('care', 'No care information available.')}"
            }

        if "temperament" in message or "nature" in message or "behavior" in message:
            return {
                "response": f"**Temperament:** {breed_info.get('temperament', 'No temperament information available.')}"
            }

        if "origin" in message or "where" in message:
            return {
                "response": f"**Origin:** {breed_info.get('origin', 'No origin information available.')}"
            }

        if "life" in message or "age" in message or "span" in message:
            response = f"**Life Span:** {breed_info.get('life_span', 'No life span information available.')}"
            if age:
                response += f"\n\n**Predicted Age Group:** {age}"
            return {"response": response}

        if "describe" in message or "about" in message or "what is this" in message:
            return {
                "response": f"**Description:** {breed_info.get('description', 'No description available.')}"
            }

        return {
            "response": (
                f"**Animal:** {animal}\n\n"
                f"**Breed:** {breed}\n\n"
                f"**Description:** {breed_info.get('description', 'No description available.')}\n\n"
                f"You can ask about **food**, **care**, **temperament**, **origin**, or **life span**."
            )
        }

    except Exception as e:
        return {"response": f"Error: {str(e)}"}
    

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        animal_result = predictor.predict(
            image,
            "animal",
            "type",
            return_uncertainty=True
        )

        if "error" in animal_result:
            return {"error": animal_result["error"]}

        animal = normalize_label(
            animal_result.get("predicted_class") or animal_result.get("raw_class")
        )

        if is_unknown_animal(animal_result):
            return {
                "animal": "unknown",
                "breed": None,
                "breed_info": {},
                "message": "Unsupported or unclear animal image",
                "animal_result": animal_result,
                "breed_result": None,
                "age": None,
                "age_result": None,
                "age_info": None,
            }

        predictor.set_thresholds(animal, "breed", 0.75, 0.15)

        breed_result = predictor.predict(
            image,
            animal,
            "breed",
            return_uncertainty=True
        )

        if "error" in breed_result:
            return {"error": breed_result["error"]}

        breed = normalize_label(
            breed_result.get("predicted_class") or breed_result.get("raw_class")
        )

        breed_info = kb.get_info(animal, "breed", breed)

        age_result = None
        age_info = None
        age_label = None

        if animal == "dog":
            predictor.set_thresholds("dog", "age", 0.60, 0.10)
            age_result = predictor.predict(
                image,
                "dog",
                "age",
                return_uncertainty=True
            )

            if "error" not in age_result:
                age_label = normalize_label(
                    age_result.get("predicted_class") or age_result.get("raw_class")
                )
                if age_label:
                    age_info = kb.get_info("dog", "age", age_label)

        return {
            "animal": animal,
            "breed": breed,
            "animal_result": animal_result,
            "breed_result": breed_result,
            "breed_info": breed_info if isinstance(breed_info, dict) else {},
            "age": age_label,
            "age_result": age_result,
            "age_info": age_info if isinstance(age_info, dict) else {},
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}