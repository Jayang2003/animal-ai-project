import base64
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)

SUPPORTED_ANIMALS = ["dog", "cat", "cow", "horse", "buffalo"]

def check_unknown(image):
    try:
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = f"""
Check the image and answer ONLY one word:
dog, cat, cow, horse, buffalo, unknown

Rules:
- If not in list → unknown
- No explanation
"""

        response = client.chat.completions.create(
            model="meta-llama/llama-3.2-11b-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                    ]
                }
            ]
        )

        return response.choices[0].message.content.strip().lower()

    except Exception as e:
        print("VLM Error:", e)
        return "unknown"