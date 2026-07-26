# BTRAN — Implementation Plan

## Overview

A Python CLI tool that takes a folder of book photos in a source language, translates them to a target language using a vision-capable LLM via `pi`, and outputs an EPUB file. Each photo is processed independently via a `pi -p` subprocess with controlled concurrency, retries, and hash-based caching for resumeability.

---

## Architecture

```
btran/
├── pyproject.toml
├── .env.example
├── README.md
└── btran/
    ├── __init__.py
    ├── __main__.py          # python -m btran entry
    ├── cli.py               # argparse entry point
    ├── config.py            # .env + CLI merging (python-dotenv)
    ├── schema.py            # JSON schema / dataclass for intermediate format
    ├── hasher.py            # Image hashing + cache (SHA256, SQLite)
    ├── translator.py        # pi subprocess invocation per image
    ├── orchestrator.py      # Main loop: scan → hash → translate → assemble
    └── epub_builder.py      # Compile intermediate JSON → EPUB via ebooklib
```

### Data flow

```
photos/          →  [hasher.py]   →  cache.sqlite (lookup)
                                 →  uncached → [translator.py] → pi -p --model X @img.jpg
                                 →  intermediate/page_001.json
                                 →  intermediate/page_002.json
                                 →  ...
                                 →  [epub_builder.py] → output.epub
```

---

## Dependencies

| Package         | Purpose                            |
|-----------------|------------------------------------|
| `ebooklib`      | EPUB generation                    |
| `imagehash`     | Perceptual hashing (phash)         |
| `Pillow`        | Image loading (imagehash dep)      |
| `python-dotenv` | .env file loading                  |
| stdlib only     | asyncio, argparse, sqlite3, json, subprocess, hashlib, pathlib |

No heavy frameworks. All pip-installable.

---

## Component Details

### 1. Config (`config.py`)

- Load from `.env` file via `python-dotenv`
- Override with CLI flags (argparse)
- Config keys:
  - `MODEL` — vision model ID (default: `gemini-2.5-flash`)
  - Source language is detected from each page during extraction; it is not configurable.
  - `TARGET_LANG` — target language (required, no default)
  - `CONCURRENCY` — max parallel pi processes (default: `4`)
  - `MAX_RETRIES` — max retries per image (default: `3`)
  - `INPUT_DIR` — path to photos folder
  - `OUTPUT_EPUB` — output epub path
  - `INTERMEDIATE_DIR` — intermediate JSON output dir (default: `./intermediate`)
  - `CACHE_DB` — SQLite cache path (default: `./cache.sqlite`)
  - `PI_BIN` — path to pi binary (default: `pi`)
  - `TIMEOUT` — per-image pi timeout in seconds (default: `0`, no timeout)
  - `TITLE` — EPUB book title
  - `AUTHOR` — EPUB book author

### 2. Intermediate JSON Schema (`schema.py`)

Each page produces one JSON file in `intermediate/`; `source_lang` is detected
from the page image, never supplied as configuration:

```json
{
  "page_number": 1,
  "image_path": "photos/page_01.jpg",
  "sha256": "abc123def456...",
  "phash": "a1b2c3d4e5f6...",
  "source_lang": "ja",
  "target_lang": "en",
  "original_text": "こんにちは世界",
  "translated_text": "Hello world",
  "image_descriptions": [],
  "model": "gemini-2.5-flash",
  "timestamp": "2026-07-24T06:22:57Z",
  "retry_count": 2
}
```

### 3. Image Hashing & Cache (`hasher.py`)

Two-tier caching:
1. **SHA256** (exact match) — if same file bytes, skip immediately
2. **phash** (perceptual) — if a visually near-identical page was already processed (Hamming distance ≤ 5), reuse result

Storage: SQLite database with columns: `sha256`, `phash`, `page_number`, `image_path`, `result_json`

Cache workflow:
- Compute SHA256 of image bytes
- If SHA256 in DB → cached hit, write intermediate JSON from stored result
- Compute phash of image
- Query all stored phashes, compute Hamming distance
- If distance ≤ threshold → near-match hit (same page, different photo)
- Otherwise → miss, queue for translation

### 4. pi Translator (`translator.py`)

Each image is translated by spawning:
```bash
pi -p --model <model> --no-session \
  "System: Translate this book page from <source> to <target>.
   Output ONLY a JSON object:
   {\"page_text\": \"...\", \"translated_text\": \"...\", \"image_descriptions\": []}
   Do not include any other text. @<image_path>"
```

Key design points:
- `--no-session` avoids session pollution (no sessions directory)
- `--mode json` NOT used — we want raw text output that contains only our JSON
- Prompt explicitly constrains output format
- Response parsed from stdout, validated against schema
- On parse failure → retry (counts as failure)
- Timeout enforced via `asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)`

### 5. Orchestrator (`orchestrator.py`)

Main async pipeline:

```
1. Scan INPUT_DIR → sorted list of image files (jpg, jpeg, png, webp, heic)
2. For each image:
   a. Compute SHA256, check cache → hit → write intermediate, skip
   b. Compute phash, check cache → near hit → write intermediate, skip
   c. Miss → add to pending queue
3. Process pending queue with asyncio.Semaphore(CONCURRENCY):
   a. Acquire semaphore slot
   b. Retry loop (MAX_RETRIES with exponential backoff + jitter)
   c. On success → save intermediate JSON + update cache
   d. On exhaust retries → save error JSON, continue
4. Progress reporting: print "✓ page 3/42 (7%)" per completion
5. Handle SIGINT: cancel pending, let running finish gracefully
```

Concurrency pattern:
```python
sem = asyncio.Semaphore(concurrency)

async def process_one(image_path, page_num):
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                result = await translate_image(image_path, ...)
                save_intermediate(page_num, result)
                update_cache(sha256, phash, result)
                return
            except Exception as e:
                if attempt == max_retries:
                    save_error(page_num, str(e))
                    return
                await asyncio.sleep(base_delay * (2 ** attempt) + jitter)
```

### 6. EPUB Builder (`epub_builder.py`)

- Load all intermediate JSON files, sorted by page_number
- Create `epub.EpubBook` with metadata (title, author, languages)
- Each page → one `epub.EpubHtml` chapter
- Chapter content: original text + translated text, side by side or sequential
- Include optional images if `--embed-images` flag set
- Generate TOC, spine, NCX, NAV
- Write `.epub` file

---

## CLI Interface

```
usage: btran [OPTIONS] INPUT_DIR OUTPUT_EPUB

Book photo → translated EPUB using vision LLM via pi.

positional arguments:
  INPUT_DIR             Folder containing book photos (sorted alphabetically)
  OUTPUT_EPUB           Output EPUB file path

options:
  --target-lang LANG    Target language code (required)
  --model MODEL         Vision model ID (default: gemini-2.5-flash)
  --concurrency N       Max parallel translations (default: 4)
  --max-retries N       Max retries per image (default: 3)
  --timeout SECONDS     Per-image timeout (default: 0, no timeout)
  --intermediate-dir DIR  Intermediate JSON directory (default: ./intermediate)
  --cache-db PATH       SQLite cache path (default: ./cache.sqlite)
  --title TITLE         EPUB book title
  --author AUTHOR       EPUB book author
  --embed-images        Embed original photos in EPUB
  --pi-bin PATH         Path to pi binary (default: pi)
  --resume              Resume from cache (default: True)
  --no-resume           Force re-translate all pages
```

All options readable from `.env` with `BTRAN_` prefix.

---

## Implementation Order

| Phase | Files | Effort |
|-------|-------|--------|
| 1. Scaffold | `pyproject.toml`, `__init__.py`, `__main__.py` | Small |
| 2. Config | `config.py` | Small |
| 3. Schema | `schema.py` | Tiny |
| 4. Hasher/Cache | `hasher.py` | Medium |
| 5. Translator | `translator.py` | Medium |
| 6. Orchestrator | `orchestrator.py` | Large |
| 7. EPUB Builder | `epub_builder.py` | Medium |
| 8. CLI | `cli.py` | Small |
| 9. Integration test | End-to-end with test images | Medium |

---

## Key Design Decisions

1. **pi subprocess, not pi-subagents extension**: The extension had `spawn pi ENOENT` failures. Direct `pi -p` subprocess calls are simpler, more reliable, and don't depend on pi's internal spawning.

2. **SHA256 + phash dual caching**: SHA256 catches exact re-runs (same file), phash catches re-photographed same page (different lighting/angle).

3. **One JSON per page, not one big file**: Enables partial resume, parallel writes, and easy debugging. Each page is an independent unit of work.

4. **No `--mode json`**: pi's JSON mode wraps the full agent response (tool calls, etc.). We want just the model's output, so we use `-p` (print) mode with a constrained prompt.

5. **Asyncio over multiprocessing**: Subprocess calls are I/O-bound (waiting for LLM API). Asyncio handles hundreds of concurrent waits efficiently without process overhead.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM hallucinates JSON format | Retry on parse failure; strict prompt engineering |
| Large images exhaust API context | Option to resize images before sending (future feature) |
| API rate limiting | Built-in retries with backoff handle transient rate limits |
| pi binary not found | Configurable `--pi-bin` path; validate at startup |
| Partial output on timeout | asyncio.wait_for kills subprocess; retry from scratch |
