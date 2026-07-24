"""Translation via pi subprocess. Async interface for the orchestrator."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from btran.schema import PageResult

TRANSLATION_PROMPT = """Translate this book page from {source_lang} to {target_lang}.
Output ONLY a raw JSON object on a single line — no markdown fences, no backticks, no explanation:
{{"page_text": "<exact original text>", "translated_text": "<translation>", "image_descriptions": ["<description>"]}}"""


class TranslationError(Exception):
    """Raised when translation fails (retryable)."""

    pass


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


async def translate_image(
    image_path: Path,
    source_lang: str,
    target_lang: str,
    model: str,
    sha256: str,
    phash: str,
    page_number: int,
    pi_bin: str = "pi",
    timeout: int = 120,
) -> PageResult:
    """Translate a single book page image using pi.

    Args:
        image_path: Path to the image file.
        source_lang: Source language code.
        target_lang: Target language code.
        model: Vision model ID to pass to pi --model.
        sha256: Pre-computed SHA256 hex digest.
        phash: Pre-computed perceptual hash hex string.
        page_number: 1-based page number in the book.
        pi_bin: Path to pi binary.
        timeout: Max seconds to wait for pi.

    Returns:
        PageResult with translation data.

    Raises:
        TranslationError: If translation fails for any reason.
    """
    prompt = TRANSLATION_PROMPT.format(source_lang=source_lang, target_lang=target_lang)
    # Attach image to the prompt
    full_prompt = f"{prompt} @{image_path}"

    try:
        proc = await asyncio.create_subprocess_exec(
            pi_bin,
            "-p",
            "--model", model,
            "--no-session",
            full_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PI_OFFLINE": "0"},
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TranslationError(
            f"pi timed out after {timeout}s for {image_path}"
        ) from None

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    # Strip markdown code fences if present (models often ignore "no fences")
    stdout = _strip_fences(stdout)

    if proc.returncode != 0:
        raise TranslationError(
            f"pi exited with code {proc.returncode} for {image_path}: {stderr[:500]}"
        )

    # Parse LLM output as JSON
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise TranslationError(
            f"Failed to parse pi JSON output for {image_path}: {e}\nOutput: {stdout[:300]}"
        ) from None

    # Validate required fields
    missing = [k for k in ("page_text", "translated_text", "image_descriptions") if k not in data]
    if missing:
        raise TranslationError(
            f"Missing required fields in pi output for {image_path}: {missing}"
        )

    return PageResult(
        page_number=page_number,
        image_path=str(image_path),
        sha256=sha256,
        phash=phash,
        source_lang=source_lang,
        target_lang=target_lang,
        page_text=data["page_text"],
        translated_text=data["translated_text"],
        image_descriptions=data.get("image_descriptions", []),
        model=model,
    )
