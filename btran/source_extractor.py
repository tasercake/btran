"""Extract typed source content from one page image with a vision Pi call."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from btran.schema import PageExtraction, SourceBlock, TermMention


BLOCK_TYPES = frozenset({
    "heading", "paragraph", "list_item", "table", "caption", "footnote",
    "pull_quote", "illustration",
})

EXTRACTION_PROMPT = """Extract the source content from this book page in {source_lang}.
Output ONLY one raw JSON object, without markdown or explanation. Use this schema:
{{
  "blocks": [
    {{"id": "model-local-id", "type": "heading|paragraph|list_item|table|caption|footnote|pull_quote|illustration", "text": "source text or illustration description", "reading_order": 0}}
  ],
  "term_mentions": [{{"term": "source term", "block_id": "model-local-id"}}],
  "illustrations": ["illustration description"]
}}
Every block needs all four fields. Assign unique non-negative reading_order values
in natural reading order. Keep source text verbatim; do not translate it. Include
illustrations as blocks and also list their descriptions in illustrations."""


class ExtractionError(Exception):
    """Raised when a source extraction cannot produce a valid artifact."""


def _strip_fences(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExtractionError(f"{name} must be an object")
    return value


def _validate_blocks(raw_blocks: Any, page_number: int) -> tuple[list[SourceBlock], dict[str, str]]:
    if not isinstance(raw_blocks, list):
        raise ExtractionError("blocks must be a list")

    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    parsed: list[tuple[str, SourceBlock]] = []
    for index, raw in enumerate(raw_blocks):
        block = _require_mapping(raw, f"blocks[{index}]")
        missing = {"id", "type", "text", "reading_order"} - block.keys()
        if missing:
            raise ExtractionError(f"blocks[{index}] missing required fields: {sorted(missing)}")

        raw_id = block["id"]
        block_type = block["type"]
        text = block["text"]
        reading_order = block["reading_order"]
        if not isinstance(raw_id, str) or not raw_id.strip() or raw_id in seen_ids:
            raise ExtractionError(f"blocks[{index}].id must be a unique non-empty string")
        if not isinstance(block_type, str) or block_type not in BLOCK_TYPES:
            raise ExtractionError(f"blocks[{index}].type must be one of {sorted(BLOCK_TYPES)}")
        if not isinstance(text, str):
            raise ExtractionError(f"blocks[{index}].text must be a string")
        if isinstance(reading_order, bool) or not isinstance(reading_order, int) or reading_order < 0:
            raise ExtractionError(f"blocks[{index}].reading_order must be a non-negative integer")
        if reading_order in seen_orders:
            raise ExtractionError("blocks must have unique reading_order values")

        seen_ids.add(raw_id)
        seen_orders.add(reading_order)
        parsed.append((raw_id, SourceBlock(
            id=f"page_{page_number}_block_{reading_order}",
            type=block_type,
            text=text,
            reading_order=reading_order,
        )))

    parsed.sort(key=lambda item: item[1].reading_order)
    return [block for _, block in parsed], {raw_id: block.id for raw_id, block in parsed}


def _validate_mentions(raw_mentions: Any, id_map: dict[str, str]) -> list[TermMention]:
    if not isinstance(raw_mentions, list):
        raise ExtractionError("term_mentions must be a list")

    mentions: list[TermMention] = []
    for index, raw in enumerate(raw_mentions):
        mention = _require_mapping(raw, f"term_mentions[{index}]")
        missing = {"term", "block_id"} - mention.keys()
        if missing:
            raise ExtractionError(f"term_mentions[{index}] missing required fields: {sorted(missing)}")
        term = mention["term"]
        raw_block_id = mention["block_id"]
        if not isinstance(term, str) or not term.strip():
            raise ExtractionError(f"term_mentions[{index}].term must be a non-empty string")
        if not isinstance(raw_block_id, str) or raw_block_id not in id_map:
            raise ExtractionError(f"term_mentions[{index}].block_id must reference a block")
        mentions.append(TermMention(term=term, block_id=id_map[raw_block_id]))
    return mentions


def parse_extraction(
    data: Any,
    *,
    image_path: Path,
    source_lang: str,
    model: str,
    sha256: str,
    phash: str,
    page_number: int,
) -> PageExtraction:
    """Validate Pi JSON and construct a PageExtraction with canonical block IDs."""
    output = _require_mapping(data, "Pi output")
    missing = {"blocks", "term_mentions", "illustrations"} - output.keys()
    if missing:
        raise ExtractionError(f"Pi output missing required fields: {sorted(missing)}")

    blocks, id_map = _validate_blocks(output["blocks"], page_number)
    mentions = _validate_mentions(output["term_mentions"], id_map)
    illustrations = output["illustrations"]
    if not isinstance(illustrations, list) or not all(isinstance(item, str) for item in illustrations):
        raise ExtractionError("illustrations must be a list of strings")

    return PageExtraction(
        page_number=page_number,
        image_path=str(image_path),
        sha256=sha256,
        phash=phash,
        source_lang=source_lang,
        model=model,
        blocks=blocks,
        term_mentions=mentions,
        illustrations=illustrations,
    )


async def extract_page(
    image_path: Path,
    source_lang: str,
    model: str,
    sha256: str,
    phash: str,
    page_number: int,
    pi_bin: str = "pi",
    timeout: int = 120,
) -> PageExtraction:
    """Run exactly one bounded vision Pi invocation and validate its extraction."""
    prompt = EXTRACTION_PROMPT.format(source_lang=source_lang)
    try:
        proc = await asyncio.create_subprocess_exec(
            pi_bin,
            "-p",
            "--model", model,
            "--no-session",
            f"{prompt} @{image_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PI_OFFLINE": "0"},
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ExtractionError(f"pi timed out after {timeout}s for {image_path}") from None
    except OSError as error:
        raise ExtractionError(f"failed to start pi for {image_path}: {error}") from error

    stdout = _strip_fences(stdout_bytes.decode("utf-8", errors="replace").strip())
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise ExtractionError(f"pi exited with code {proc.returncode} for {image_path}: {stderr[:500]}")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ExtractionError(f"failed to parse pi JSON output for {image_path}: {error}") from error

    return parse_extraction(
        data,
        image_path=image_path,
        source_lang=source_lang,
        model=model,
        sha256=sha256,
        phash=phash,
        page_number=page_number,
    )


def legacy_page_text(extraction: PageExtraction) -> str:
    """Derive legacy flat page text from non-illustration blocks in reading order."""
    return "\n\n".join(
        block.text for block in sorted(extraction.blocks, key=lambda block: block.reading_order)
        if block.type != "illustration" and block.text
    )


def to_file(extraction: PageExtraction, path: Path) -> None:
    """Atomically write a PageExtraction JSON artifact at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as artifact:
            temp_name = artifact.name
            json.dump(extraction.to_dict(), artifact, indent=2, ensure_ascii=False)
            artifact.write("\n")
            artifact.flush()
            os.fsync(artifact.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise


def extraction_cache_identity(sha256: str, source_lang: str, model: str) -> str:
    """Return a source-extraction-only cache identity, separate from translations."""
    semantic_inputs = json.dumps(
        {
            "kind": "source-extraction",
            "image_sha256": sha256,
            "source_lang": source_lang,
            "model": model,
            "prompt": EXTRACTION_PROMPT,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "extraction:" + hashlib.sha256(semantic_inputs.encode("utf-8")).hexdigest()
