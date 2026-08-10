"""
Optional Gemini connectivity test.

Gemini is NOT required for the core RAG-from-scratch project.
Run this file only when you want to test the optional cloud generator.
"""

import os

from dotenv import load_dotenv


load_dotenv(override=True)

api_key = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()


if not api_key:
    print(
        "SKIPPED: Gemini is optional and GEMINI_API_KEY "
        "is not configured."
    )

else:
    try:
        from openai import OpenAI

        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash",
        ).strip()

        base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip()

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=45,
        )

        response = (
            client.chat.completions.create(
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
        )

        print(
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:
        print(
            "Optional Gemini test failed:",
            type(exc).__name__,
            str(exc),
        )
