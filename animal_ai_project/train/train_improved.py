"""Streamlit app for automatic animal detection, breed prediction, breed details, and dog age prediction."""

import streamlit as st
from PIL import Image

from src.predictor_improved import ProductionPredictor
from src.knowledge_base import AnimalKnowledgeBase


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Animal Classifier Pro",
    page_icon="🐾",
    layout="wide",
)


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_services():
    predictor = ProductionPredictor(models_dir="models")
    kb = AnimalKnowledgeBase()

    model_status = {
        "dog_breed": predictor.load_model("dog", "breed"),
        "cow_breed": predictor.load_model("cow", "breed"),
        "dog_age": predictor.load_model("dog", "age"),
    }

    return predictor, kb, model_status


# =========================
# UI HELPERS
# =========================
def format_age_label(label: str) -> str:
    if not label:
        return "Unknown"
    return label.replace("_", " to ").replace("plus", "+").title() + " Years"


def render_prediction_block(title, animal_type, result):
    st.subheader(title)

    if "error" in result:
        st.error(result["error"])
        return

    st.markdown(f"**Animal Name:** {animal_type.title()}")

    if result.get("is_uncertain", False):
        label = result.get("raw_class", "Unknown")
        st.warning("Prediction is uncertain")
        st.markdown(f"**Best Guess:** {label}")
    else:
        label = result.get("predicted_class", "Unknown")
        st.success(f"Prediction: {label}")


def render_age_info(age_result, age_info=None):
    st.subheader("Age Prediction")

    if not age_result:
        st.info("Age prediction not available.")
        return

    if "error" in age_result:
        st.error(age_result["error"])
        return

    if age_result.get("is_uncertain", False):
        label = age_result.get("raw_class", "Unknown")
        st.warning("Age prediction is uncertain")
        st.markdown(f"**Reason:** {age_result.get('reason', 'Low confidence')}")
        st.markdown(f"**Best Guess:** {format_age_label(label)}")
    else:
        label = age_result.get("predicted_class", "Unknown")
        st.success(f"Estimated Age Range: {format_age_label(label)}")

    if age_info:
        st.markdown("### Age Stage Details")
        if "stage" in age_info:
            st.markdown(f"**Stage:** {age_info['stage']}")
        if "care" in age_info:
            st.markdown(f"**Care:** {age_info['care']}")
        if "food" in age_info:
            st.markdown(f"**Food Advice:** {age_info['food']}")
        if "note" in age_info:
            st.markdown(f"**Note:** {age_info['note']}")


def render_breed_info(breed_label, info):
    if not breed_label or not info:
        st.warning(f"No breed information found for: {breed_label}")
        return

    breed_name = info.get("breed", breed_label.replace("_", " ").title())
    animal = info.get("animal", "N/A")
    origin = info.get("origin", "N/A")
    life_span = info.get("life_span", info.get("lifespan", "N/A"))
    temperament = info.get("temperament", "N/A")
    food = info.get("food", "N/A")
    care = info.get("care", info.get("care_tips", "N/A"))
    description = info.get("description", "N/A")

    st.markdown("### Breed Details")
    st.markdown(f"**Breed Name:** {breed_name}")

    if animal != "N/A":
        st.markdown(f"**Animal Type:** {animal}")
    if origin != "N/A":
        st.markdown(f"**Origin / Place:** {origin}")
    if life_span != "N/A":
        st.markdown(f"**Life Span:** {life_span}")
    if temperament != "N/A":
        st.markdown(f"**Temperament:** {temperament}")
    if food != "N/A":
        st.markdown(f"**Food:** {food}")
    if care != "N/A":
        st.markdown(f"**Care:** {care}")
    if description != "N/A":
        st.markdown(f"**Description:** {description}")


def show_examples():
    st.info("Upload a dog or cow image")


# =========================
# DETECT ANIMAL
# =========================
def detect_animal_type(image, predictor):
    dog_result = predictor.predict(image, "dog", "breed", return_uncertainty=True)
    cow_result = predictor.predict(image, "cow", "breed", return_uncertainty=True)

    if "error" in dog_result and "error" in cow_result:
        return None, None, {"error": "Dog and cow models failed."}

    dog_conf = dog_result.get("confidence", 0)
    cow_conf = cow_result.get("confidence", 0)

    if dog_conf >= cow_conf:
        return "dog", "Dog", dog_result
    else:
        return "cow", "Cow", cow_result


# =========================
# MAIN APP
# =========================
def main():
    st.title("🐾 Animal Classification Pro")
    st.markdown("Upload image → auto detect animal → show breed details + dog age")

    predictor, kb, model_status = load_services()

    if not any(model_status.values()):
        st.error("No models loaded")
        return

    st.sidebar.title("Settings")

    with st.sidebar.expander("Breed Threshold Settings"):
        breed_conf = st.slider("Breed Confidence Threshold", 0.50, 1.00, 0.75, 0.05)
        breed_margin = st.slider("Breed Margin Threshold", 0.05, 0.50, 0.15, 0.05)

    with st.sidebar.expander("Age Threshold Settings"):
        age_conf = st.slider("Age Confidence Threshold", 0.50, 1.00, 0.60, 0.05)
        age_margin = st.slider("Age Margin Threshold", 0.05, 0.50, 0.10, 0.05)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        show_examples()
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=250)

    with st.spinner("Processing..."):
        animal_type, animal_label, breed_result = detect_animal_type(image, predictor)

        if not animal_type:
            st.error("Could not detect animal")
            return

        predictor.set_thresholds(animal_type, "breed", breed_conf, breed_margin)
        breed_result = predictor.predict(image, animal_type, "breed", return_uncertainty=True)

        age_result = None
        age_info = None

        if animal_type == "dog" and model_status.get("dog_age"):
            predictor.set_thresholds("dog", "age", age_conf, age_margin)
            age_result = predictor.predict(image, "dog", "age", return_uncertainty=True)

            age_label = age_result.get("raw_class") if age_result.get("is_uncertain") else age_result.get("predicted_class")
            if age_label:
                age_info = kb.get_info("dog", "age", age_label)

    st.markdown(f"## Detected Animal: {animal_label}")

    col1, col2 = st.columns(2)

    with col1:
        render_prediction_block("Breed Prediction", animal_type, breed_result)

        breed_label = (
            breed_result.get("raw_class")
            if breed_result.get("is_uncertain", False)
            else breed_result.get("predicted_class")
        )

        if breed_label:
            breed_info = kb.get_info(animal_type, "breed", breed_label)
            render_breed_info(breed_label, breed_info)

    with col2:
        if animal_type == "dog":
            render_age_info(age_result, age_info)
        else:
            st.subheader("Age Prediction")
            st.info("Age prediction is currently available only for dog.")


if __name__ == "__main__":
    main()
