import asyncio
import os
from pathlib import Path

import pdfplumber
from openai import OpenAIError
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_PDF_FOLDER = "pdfs"
MAX_CHARS = 6000  # Limit text to avoid exceeding context window


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Seite {i + 1}]\n{page_text}")
    return "\n\n".join(text_parts)


def build_llm() -> OpenAILike:
    """Initialize the LLM from environment variables (same as starter.py)."""
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

    return OpenAILike(
        model=model,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,
        is_chat_model=True,
        is_function_calling_model=False,
    )


async def summarize_text(llm: OpenAILike, text: str, filename: str) -> str:
    """Send extracted PDF text to the LLM and return the summary."""
    truncated = text[:MAX_CHARS]
    if len(text) > MAX_CHARS:
        truncated += "\n\n[... Text wurde gekürzt ...]"

    prompt = (
        f"Hier ist der extrahierte Text aus der PDF-Datei '{filename}':\n\n"
        f"{truncated}\n\n"
        "Bitte schreibe eine klare, strukturierte Zusammenfassung auf Deutsch. "
        "Fasse die wichtigsten Punkte, Themen und Kernaussagen zusammen."
    )

    try:
        response = await llm.acomplete(prompt)
        return str(response)
    except OpenAIError as error:
        raise RuntimeError(
            f"API-Fehler bei '{filename}': {error}. "
            "Prüfe OPENAI_MODEL in .env und deine OpenRouter-Einstellungen."
        ) from error


async def main():
    pdf_folder = Path(os.getenv("PDF_FOLDER", DEFAULT_PDF_FOLDER))

    if not pdf_folder.exists():
        print(f"Ordner '{pdf_folder}' nicht gefunden. Erstelle ihn und lege PDFs hinein.")
        pdf_folder.mkdir(parents=True)
        return

    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"Keine PDF-Dateien in '{pdf_folder}' gefunden.")
        return

    print(f"Gefundene PDFs: {len(pdf_files)}\n")
    llm = build_llm()

    for pdf_path in pdf_files:
        print("=" * 60)
        print(f"📄 Datei: {pdf_path.name}")
        print("=" * 60)

        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print("⚠️  Kein Text extrahierbar (evtl. gescannte PDF ohne OCR).\n")
            continue

        print(f"Extrahiert: {len(text)} Zeichen | Sende an KI...\n")
        summary = await summarize_text(llm, text, pdf_path.name)

        print("📝 Zusammenfassung:")
        print(summary)
        print()


if __name__ == "__main__":
    asyncio.run(main())