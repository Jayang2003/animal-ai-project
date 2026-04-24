"""
Animal Classifier Chatbot
=========================
Conversational assistant powered by OpenRouter.
API key is loaded from OPENROUTER_API_KEY in your .env file.

OpenRouter is OpenAI-compatible, so we use the openai SDK
pointed at https://openrouter.ai/api/v1
"""

import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ------------------------------------------------------------------ #
# Config from environment                                            #
# ------------------------------------------------------------------ #
_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4-5")
_MAX_TOKENS = int(os.getenv("CHATBOT_MAX_TOKENS", "1024"))

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ------------------------------------------------------------------ #
# System prompt                                                      #
# ------------------------------------------------------------------ #
SYSTEM_PROMPT = """You are an expert Animal Veterinary & Care Assistant integrated into the
Animal Classifier Pro application.

Your role:
- Help users understand the animal detected from the uploaded image.
- Use the provided app context (animal, breed, age) whenever available.
- Answer questions about care, food, health, behavior, temperament, grooming, and training.
- If no app context is available, ask the user to upload an animal image first or ask a general animal-care question.

Guidelines:
- Be warm, friendly, and concise.
- Personalize answers using the detected animal, breed, and age when available.
- When discussing health concerns, always recommend consulting a licensed veterinarian.
- If asked something outside animal care, politely redirect to animal-related help.
- Use simple language unless the user asks for more technical detail.
- Prefer short paragraphs and practical advice.
"""

# ------------------------------------------------------------------ #
# Client factory                                                     #
# ------------------------------------------------------------------ #
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not _API_KEY or _API_KEY == "your_openrouter_api_key_here":
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file and restart the app."
            )
        _client = OpenAI(
            api_key=_API_KEY,
            base_url=_OPENROUTER_BASE_URL,
        )
    return _client


# ------------------------------------------------------------------ #
# Public API                                                         #
# ------------------------------------------------------------------ #
def is_configured() -> bool:
    return bool(_API_KEY) and _API_KEY != "your_openrouter_api_key_here"


def chat(
    message: str,
    history: list[dict],
    context: Optional[dict] = None,
) -> str:
    client = _get_client()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        ctx_parts = []
        if context.get("animal"):
            ctx_parts.append(f"Detected animal: {context['animal'].title()}")
        if context.get("breed"):
            ctx_parts.append(f"Predicted breed: {context['breed']}")
        if context.get("age"):
            age_str = context["age"].replace("_", " to ").replace("plus", "+")
            ctx_parts.append(f"Estimated age range: {age_str} years")

        if ctx_parts:
            ctx_text = "[App context]\n" + "\n".join(ctx_parts)
            messages.append({"role": "user", "content": ctx_text})
            messages.append({
                "role": "assistant",
                "content": "Got it. I will use this detected animal context in my answers."
            })

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        messages=messages,
    )

    return response.choices[0].message.content