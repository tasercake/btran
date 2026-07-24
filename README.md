# btran

Translate book photos to EPUB using vision LLMs via [pi](https://pi.dev).

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
btran ./photos/ output.epub --target-lang ja --model gemini-2.5-flash
```

## Config

All options via CLI or `.env` (prefix `BTRAN_`). See `.env.example`.

## How it works

1. Scans input folder for images (`.jpg`, `.png`, `.webp`)
2. Hashes each image (SHA256 + perceptual phash) → skips cached
3. Spawns `pi -p --model <model> @image.jpg` per uncached image (concurrency=4)
4. Parses structured JSON from each pi response
5. Compiles all results into EPUB with `ebooklib`
