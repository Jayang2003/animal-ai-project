import streamlit as st
from PIL import Image

from src.knowledge_base import AnimalKnowledgeBase
from src.predictor_improved import ProductionPredictor
from src.chatbot import chat, is_configured


st.set_page_config(page_title="Animal Classification Pro", layout="wide")


@st.cache_resource
def load_services():
    predictor = ProductionPredictor(models_dir="models")
    kb = AnimalKnowledgeBase()

    model_status = {
        "animal_type": predictor.load_model("animal", "type"),
        "dog_breed": predictor.load_model("dog", "breed"),
        "cat_breed": predictor.load_model("cat", "breed"),
        "cow_breed": predictor.load_model("cow", "breed"),
        "horse_breed": predictor.load_model("horse", "breed"),
        "buffalo_breed": predictor.load_model("buffalo", "breed"),
        "dog_age": predictor.load_model("dog", "age"),
    }

    return predictor, kb, model_status


def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {}
    if "analysis" not in st.session_state:
        st.session_state.analysis = None


def normalize_label(value: str) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def format_age_label(label: str) -> str:
    if not label:
        return "Unknown"
    return label.replace("_", " to ").replace("plus", "+").title() + " Years"


def detect(image, predictor, kb, model_status):
    animal_result = predictor.predict(image, "animal", "type", return_uncertainty=True)

    if "error" in animal_result:
        return {"error": animal_result["error"]}

    animal = normalize_label(
        animal_result.get("predicted_class") or animal_result.get("raw_class")
    )

    predictor.set_thresholds(animal, "breed", 0.75, 0.15)
    breed_result = predictor.predict(image, animal, "breed", return_uncertainty=True)

    if "error" in breed_result:
        return {"error": breed_result["error"]}

    breed = normalize_label(
        breed_result.get("predicted_class") or breed_result.get("raw_class")
    )

    age_result = None
    age_info = None

    if animal == "dog":
        predictor.set_thresholds("dog", "age", 0.60, 0.10)
        age_result = predictor.predict(image, "dog", "age", return_uncertainty=True)

        age_label = normalize_label(
            age_result.get("predicted_class") or age_result.get("raw_class")
        )

        if age_label:
            age_info = kb.get_info("dog", "age", age_label)

    breed_info = kb.get_info(animal, "breed", breed)

    return {
        "image": image,
        "animal": animal,
        "breed": breed,
        "breed_info": breed_info,
        "age_result": age_result,
        "age_info": age_info,
        "animal_result": animal_result,
        "breed_result": breed_result,
    }


def render_chat(predictor, kb, model_status):
    st.markdown("## 🤖 Ask the Assistant")

    if not is_configured():
        st.warning("Add OPENROUTER_API_KEY in .env")
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Ask or upload image...",
        accept_file=True,
        file_type=["jpg", "png", "jpeg", "webp"]
    )

    if not prompt:
        return

    text = prompt.text if hasattr(prompt, "text") else ""
    files = prompt.files if hasattr(prompt, "files") else []

    if files:
        image = Image.open(files[0]).convert("RGB")

        with st.spinner("Analyzing image..."):
            result = detect(image, predictor, kb, model_status)

        if "error" in result:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["error"]
            })
            st.rerun()

        st.session_state.analysis = result
        st.session_state.chat_context = {
            "animal": result["animal"],
            "breed": result["breed"]
        }

        st.session_state.chat_history = []

        msg = f"""
Image analyzed ✅  
Animal: {result['animal']}  
Breed: {result['breed']}  

Now ask anything about this animal.
"""

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": msg
        })

        if text:
            st.session_state.chat_history.append({"role": "user", "content": text})

            reply = chat(
                message=text,
                history=st.session_state.chat_history[:-1],
                context=st.session_state.chat_context
            )

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        st.rerun()

    if text:
        st.session_state.chat_history.append({"role": "user", "content": text})

        reply = chat(
            message=text,
            history=st.session_state.chat_history[:-1],
            context=st.session_state.chat_context
        )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


def render_age_result(age_result, age_info=None):
    st.markdown("### Age Prediction")

    if not age_result:
        st.info("Age prediction not available.")
        return

    if "error" in age_result:
        st.error(age_result["error"])
        return

    if age_result.get("is_uncertain", False):
        label = age_result.get("raw_class", "Unknown")
        st.warning("Age prediction is uncertain")
        st.markdown(
            f"**Reason:** {age_result.get('reason', 'Confidence is below threshold')}"
        )
        st.markdown(f"**Best Guess:** {format_age_label(label)}")
    else:
        label = age_result.get("predicted_class", "Unknown")
        st.success(f"Estimated Age Range: {format_age_label(label)}")

    if age_info:
        st.markdown("#### Age Stage Details")
        if age_info.get("stage"):
            st.markdown(f"**Stage:** {age_info['stage']}")
        if age_info.get("care"):
            st.markdown(f"**Care:** {age_info['care']}")
        if age_info.get("food"):
            st.markdown(f"**Food Advice:** {age_info['food']}")
        if age_info.get("note"):
            st.markdown(f"**Note:** {age_info['note']}")


def render_result():
    data = st.session_state.analysis
    if not data:
        st.info("Upload image in chat above")
        return

    st.image(data["image"], width=300)
    st.markdown(f"## {data['animal'].title()} - {data['breed'].title()}")

    if data["breed_info"]:
        st.markdown("### Breed Details")

        labels = {
            "animal": "Animal",
            "breed": "Breed",
            "origin": "Origin",
            "life_span": "Life Span",
            "temperament": "Temperament",
            "food": "Food",
            "care": "Care",
            "description": "Description",
        }

        for key, label in labels.items():
            if key in data["breed_info"]:
                st.markdown(f"**{label}:** {data['breed_info'][key]}")
    else:
        st.warning("Breed details not found in knowledge base.")
        with st.expander("Debug info"):
            st.write("Animal:", data["animal"])
            st.write("Breed:", data["breed"])
            st.write("Breed info:", data["breed_info"])

    if data["animal"] == "dog":
        render_age_result(data["age_result"], data["age_info"])


def main():
    init_state()
    st.title("🐾 Animal Classification Pro")

    predictor, kb, model_status = load_services()

    render_chat(predictor, kb, model_status)
    render_result()


if __name__ == "__main__":
    main()