"""Glossary-aware, text-only translation of extracted source blocks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

from btran.schema import PageExtraction, TerminologyMap, TranslatedBlock

TRANSLATION_PROMPT = """Translate the supplied source blocks from {source_lang} to {target_lang}.
Honor the glossary target forms exactly where applicable. Preserve every block ID.
The adjacent source boundaries provide context only; do not translate them separately.
Output ONLY one raw JSON object, with this shape:
{{"blocks": [{{"block_id": "<source id>", "translated_text": "<translation>"}}]}}

Input:
{context}"""


class TranslationError(Exception):
    """Raised when text-block translation cannot produce a valid result."""


def _strip_fences(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _glossary_slice(extraction: PageExtraction, glossary: TerminologyMap) -> list[dict]:
    source_text = "\n".join(block.text for block in extraction.blocks).casefold()
    return [
        entry.to_dict()
        for entry in glossary.entries
        if any(term.casefold() in source_text for term in entry.source_terms)
    ]


def _translation_context(extraction: PageExtraction, glossary: TerminologyMap) -> dict:
    blocks = extraction.blocks
    return {
        "source_blocks": [block.to_dict() for block in blocks],
        "glossary": _glossary_slice(extraction, glossary),
        "adjacent_source_boundaries": [
            {
                "block_id": block.id,
                "previous": blocks[index - 1].text if index else None,
                "next": blocks[index + 1].text if index + 1 < len(blocks) else None,
            }
            for index, block in enumerate(blocks)
        ],
    }


def translation_cache_identity(
    *,
    source_artifact_hash: str,
    glossary_hash: str,
    source_lang: str,
    target_lang: str,
    model: str,
) -> str:
    """Fingerprint a text translation independently from image translation cache keys."""
    context = {
        "source_artifact_hash": source_artifact_hash,
        "glossary_hash": glossary_hash,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model,
        "prompt": TRANSLATION_PROMPT,
    }
    encoded = json.dumps(context, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _translated_blocks(data: object, source_ids: list[str]) -> list[TranslatedBlock]:
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), list):
        raise TranslationError("Missing required blocks array in pi output")

    try:
        returned = [TranslatedBlock.from_dict(block) for block in data["blocks"]]
    except (TypeError, KeyError) as exc:
        raise TranslationError(f"Invalid translated block in pi output: {exc}") from None

    returned_ids = [block.block_id for block in returned]
    source_set, returned_set = set(source_ids), set(returned_ids)
    missing = source_set - returned_set
    extra = returned_set - source_set
    duplicate = len(returned_ids) != len(returned_set)
    if missing or extra or duplicate:
        details = []
        if missing:
            details.append(f"missing block IDs: {sorted(missing)}")
        if extra:
            details.append(f"extra block IDs: {sorted(extra)}")
        if duplicate:
            details.append("duplicate block IDs")
        raise TranslationError("; ".join(details))

    by_id = {block.block_id: block for block in returned}
    return [by_id[block_id] for block_id in source_ids]


async def translate_blocks(
    extraction: PageExtraction,
    glossary: TerminologyMap,
    *,
    model: str,
    pi_bin: str = "pi",
    timeout: int = 120,
) -> list[TranslatedBlock]:
    """Translate an extracted page with a relevant glossary slice, never an image."""
    context = _translation_context(extraction, glossary)
    prompt = TRANSLATION_PROMPT.format(
        source_lang=extraction.source_lang,
        target_lang=glossary.target_lang,
        context=json.dumps(context, ensure_ascii=False),
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            pi_bin,
            "-p",
            "--model",
            model,
            "--no-session",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PI_OFFLINE": "0"},
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TranslationError(f"pi timed out after {timeout}s for page {extraction.page_number}") from None

    stdout = _strip_fences(stdout_bytes.decode("utf-8", errors="replace").strip())
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise TranslationError(f"pi exited with code {proc.returncode}: {stderr[:500]}")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TranslationError(f"Failed to parse pi JSON output: {exc}") from None

    return _translated_blocks(data, [block.id for block in extraction.blocks])


async def translate_image(*args: object, **kwargs: object) -> None:
    """Temporary import-compatible boundary until WP-7 wires extraction to blocks.

    This deliberately performs no vision call: image translation was replaced
    by :func:`translate_blocks` and cannot produce an unstructured result.
    """
    raise TranslationError("image translation was replaced by text-block translation")
