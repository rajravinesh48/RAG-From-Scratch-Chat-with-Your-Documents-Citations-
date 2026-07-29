import os

from dotenv import load_dotenv
from openai import OpenAI


# Force values from this project's .env file.
load_dotenv(override=True)


api_key = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

model = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
).strip()

base_url = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
).strip()


print("Gemini key loaded:", bool(api_key))
print("Gemini key length:", len(api_key))
print("Model:", model)
print("Provider:", base_url)


if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env"
    )

if api_key in {
    "YOUR_GEMINI_API_KEY_HERE",
    "your-gemini-api-key-here",
}:
    raise RuntimeError(
        "Replace the placeholder with the complete Gemini key."
    )


client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=45,
)


try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "Gemini connection successful"
                ),
            }
        ],
        temperature=0,
    )

    print("\nResponse:")
    print(response.choices[0].message.content)

except Exception as exc:
    print("\nGemini request failed")
    print("Error type:", type(exc).__name__)
    print("Error details:", str(exc))