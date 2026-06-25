import asyncio
import os

from openai import OpenAIError
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
DEFAULT_CONTEXT_WINDOW = 8192


async def main():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_base = os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE).strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip()
    context_window = int(os.getenv("OPENAI_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW))
    placeholder_keys = {"your_api_key_here", "dein_echter_key_hier"}

    if not api_key or api_key in placeholder_keys:
        raise RuntimeError(
            "OPENAI_API_KEY fehlt oder ist noch ein Platzhalter. "
            "Bitte trage deinen echten API-Key in die .env-Datei ein."
        )

    llm = OpenAILike(
        model=model,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,
        is_chat_model=True,
        is_function_calling_model=False,
    )

    try:
        response = await llm.acomplete(
            "What is the capital of Germany? Answer with one word."
        )
    except OpenAIError as error:
        raise RuntimeError(
            "Der OpenAI-kompatible API-Aufruf ist fehlgeschlagen. "
            "Wenn du OpenRouter nutzt, pruefe OPENAI_MODEL in .env und deine "
            "Privacy-Einstellungen unter https://openrouter.ai/settings/privacy."
        ) from error

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
