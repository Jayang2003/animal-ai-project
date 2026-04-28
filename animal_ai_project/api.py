from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import json
import torch
import numpy as np

from src.predictor_improved import ProductionPredictor
from src.knowledge_base import AnimalKnowledgeBase
from src.chatbot import chat as chatbot_chat, is_configured as chatbot_is_configured


# ── Pydantic models MUST be defined before routes ─────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message:    str
    username:   str | None  = None
    session_id: str | None  = None
    history:    list | None = None
    context:    dict | None = None


# ── FastAPI app ────────────────────────────────────────────────────────────
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
print("animal_type:",   predictor.load_model("animal",  "type"))
print("dog_breed:",     predictor.load_model("dog",     "breed"))
print("cat_breed:",     predictor.load_model("cat",     "breed"))
print("cow_breed:",     predictor.load_model("cow",     "breed"))
print("horse_breed:",   predictor.load_model("horse",   "breed"))
print("buffalo_breed:", predictor.load_model("buffalo", "breed"))
print("dog_age:",       predictor.load_model("dog",     "age"))


# ── CLIP setup ────────────────────────────────────────────────────────────
_clip_model      = None
_clip_preprocess = None
_clip_device     = None

SUPPORTED_ANIMALS = ["dog", "cat", "cow", "horse", "buffalo"]


def get_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        try:
            import clip
            _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
            print(f"[CLIP] Loaded successfully on {_clip_device}")
        except Exception as e:
            print(f"[CLIP] Failed to load: {e}")
    return _clip_model, _clip_preprocess, _clip_device


# Load CLIP at startup
get_clip()


# ── helpers ────────────────────────────────────────────────────────────────

def normalize_label(value: str) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def is_unknown_animal(result: dict, image: Image.Image) -> bool:
    """
    Two-stage unknown detection:
    Stage 1 — Confidence threshold
    Stage 2 — CLIP verification
    """
    confidence = float(result.get("confidence", 0) or 0)
    margin     = float(result.get("margin",     0) or 0)

    print(f"[Stage1] confidence={confidence:.2f}  margin={margin:.2f}")

    # Stage 1: Keras model not confident enough
    if confidence < 0.55 or margin < 0.10:
        print("[Stage1] REJECTED — low confidence / margin")
        return True

    # Stage 2: CLIP double-check
    try:
        import clip
        clip_model, clip_preprocess, clip_device = get_clip()

        if clip_model is None:
            print("[Stage2] CLIP not available, falling back to threshold only")
            return False

        animal_labels    = [f"a photo of a {a}" for a in SUPPORTED_ANIMALS]
        non_animal_label = (
            "a photo of a human, person, man, woman, child, "
            "car, vehicle, food, furniture, building, or any non-animal object"
        )
        all_labels = animal_labels + [non_animal_label]

        image_input = clip_preprocess(image).unsqueeze(0).to(clip_device)
        text_input  = clip.tokenize(all_labels).to(clip_device)

        with torch.no_grad():
            logits, _ = clip_model(image_input, text_input)
            probs     = logits.softmax(dim=-1)[0].cpu().numpy()

        best_animal_score = float(probs[:len(SUPPORTED_ANIMALS)].max())
        best_animal_name  = SUPPORTED_ANIMALS[int(probs[:len(SUPPORTED_ANIMALS)].argmax())]
        non_animal_score  = float(probs[-1])

        print(
            f"[Stage2] CLIP best_animal={best_animal_name} ({best_animal_score:.2f})  "
            f"non_animal={non_animal_score:.2f}"
        )

        if non_animal_score > best_animal_score:
            print("[Stage2] REJECTED — CLIP says not a supported animal")
            return True

        print("[Stage2] PASSED — CLIP confirms supported animal")

    except Exception as e:
        print(f"[Stage2] CLIP error (skipping): {e}")

    return False


def is_unknown_breed(result: dict) -> bool:
    confidence = float(result.get("confidence", 0) or 0)
    margin     = float(result.get("margin",     0) or 0)
    entropy    = float(result.get("entropy",    0) or 0)
    top2_ratio = float(result.get("top2_ratio", 0) or 0)

    if confidence < 0.75 or margin < 0.15:
        return True
    if entropy > 0.65 or top2_ratio > 0.65:
        return True
    return False


def _kb_answer(message: str, animal: str, breed: str, age: str, breed_info: dict) -> str | None:
    msg = message.strip().lower()

    if any(w in msg for w in ["food", "eat", "diet", "feed"]):
        return f"**Food:** {breed_info.get('food', 'No food information available.')}"

    if any(w in msg for w in ["care", "groom", "maintain"]):
        return f"**Care:** {breed_info.get('care', 'No care information available.')}"

    if any(w in msg for w in ["temperament", "nature", "behavior"]):
        return f"**Temperament:** {breed_info.get('temperament', 'No temperament information available.')}"

    if any(w in msg for w in ["origin", "where"]):
        return f"**Origin:** {breed_info.get('origin', 'No origin information available.')}"

    if any(w in msg for w in ["life", "span"]):
        resp = f"**Life Span:** {breed_info.get('life_span', 'No life span information available.')}"
        if age:
            resp += f"\n\n**Predicted Age Group:** {age}"
        return resp

    if any(w in msg for w in ["describe", "about", "what is this"]):
        return f"**Description:** {breed_info.get('description', 'No description available.')}"

    return None


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Animal API running"}


@app.post("/login")
def login(data: LoginRequest):
    if data.username == "admin@gmail.com" and data.password == "123":
        return {"success": True,  "message": "Login successful"}
    return      {"success": False, "message": "Invalid credentials"}


@app.post("/chat")
def chat_api(data: ChatRequest):
    try:
        message = (data.message or "").strip()
        context = data.context or {}
        history = data.history or []

        animal = normalize_label(context.get("animal", ""))
        breed  = normalize_label(context.get("breed",  ""))
        age    = context.get("age", "")

        if animal and animal != "unknown" and breed:
            breed_info = kb.get_info(animal, "breed", breed) or {}
            kb_reply   = _kb_answer(message, animal, breed, age, breed_info)
            if kb_reply:
                return {"response": kb_reply}

        if not chatbot_is_configured():
            if not animal or animal == "unknown":
                return {
                    "response": (
                        "I'm ready to help! Please upload an animal image first, "
                        "then ask me questions about food, care, temperament, origin, or life span."
                    )
                }
            return {
                "response": (
                    f"I detected a **{breed.replace('_', ' ').title()}** ({animal}). "
                    f"You can ask me about **food**, **care**, **temperament**, **origin**, or **life span**."
                )
            }

        chat_context = {}
        if animal and animal != "unknown":
            chat_context["animal"] = animal
        if breed:
            chat_context["breed"] = breed
        if age:
            chat_context["age"] = age

        ai_response = chatbot_chat(
            message=message,
            history=history,
            context=chat_context if chat_context else None,
        )
        return {"response": ai_response}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}


@app.post("/predict")
async def predict(
    file:     UploadFile = File(...),
    username: str        = Form(None),
    message:  str        = Form(None),
    history:  str        = Form(None),
):
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")

        # Step 1: Predict animal type
        animal_result = predictor.predict(
            image, "animal", "type", return_uncertainty=True
        )

        if "error" in animal_result:
            return {"error": animal_result["error"]}

        animal = normalize_label(
            animal_result.get("predicted_class") or animal_result.get("raw_class")
        )

        # Step 2: Two-stage unknown check (threshold + CLIP)
        if is_unknown_animal(animal_result, image):
            return {
                "animal":        "unknown",
                "breed":         None,
                "breed_info":    {},
                "message":       "Unsupported or unclear animal image",
                "animal_result": animal_result,
                "breed_result":  None,
                "age":           None,
                "age_result":    None,
                "age_info":      None,
            }

        # Step 3: Predict breed
        predictor.set_thresholds(animal, "breed", 0.75, 0.15)

        breed_result = predictor.predict(
            image, animal, "breed", return_uncertainty=True
        )

        if "error" in breed_result:
            return {"error": breed_result["error"]}

        breed      = normalize_label(
            breed_result.get("predicted_class") or breed_result.get("raw_class")
        )

        if is_unknown_breed(breed_result):
            return {
                "animal":        animal,
                "breed":         "unknown",
                "breed_info":    {},
                "message":       "Dog breed not in the training dataset",
                "animal_result": animal_result,
                "breed_result":  breed_result,
                "age":           None,
                "age_result":    None,
                "age_info":      None,
            }

        breed_info = kb.get_info(animal, "breed", breed)

        # Step 4: Age prediction (dogs only)
        age_result = None
        age_info   = None
        age_label  = None

        if animal == "dog":
            predictor.set_thresholds("dog", "age", 0.60, 0.10)
            age_result = predictor.predict(
                image, "dog", "age", return_uncertainty=True
            )
            if "error" not in age_result:
                age_label = normalize_label(
                    age_result.get("predicted_class") or age_result.get("raw_class")
                )
                if age_label:
                    age_info = kb.get_info("dog", "age", age_label)

        # Step 5: Optional chat response
        chat_response = None
        if message and message.strip():
            parsed_history = []
            if history:
                try:
                    parsed_history = json.loads(history)
                except Exception:
                    parsed_history = []

            context = {"animal": animal, "breed": breed}
            if age_label:
                context["age"] = age_label

            breed_info_dict = breed_info if isinstance(breed_info, dict) else {}
            kb_reply = _kb_answer(message, animal, breed, age_label or "", breed_info_dict)

            if kb_reply:
                chat_response = kb_reply
            elif chatbot_is_configured():
                chat_response = chatbot_chat(
                    message=message.strip(),
                    history=parsed_history,
                    context=context,
                )

        return {
            "animal":        animal,
            "breed":         breed,
            "animal_result": animal_result,
            "breed_result":  breed_result,
            "breed_info":    breed_info if isinstance(breed_info, dict) else {},
            "age":           age_label,
            "age_result":    age_result,
            "age_info":      age_info if isinstance(age_info, dict) else {},
            "chat_response": chat_response,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}