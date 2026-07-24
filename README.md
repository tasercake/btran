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

The CLI requires `INPUT_DIR`, `OUTPUT_EPUB`, and a target language (flag or
`BTRAN_TARGET_LANG`). CLI options override `.env` values. See `.env.example`
for the complete supported production configuration surface.

## Integrated pipeline

Each run loads or creates a manifest, preflights every listed page, extracts
source blocks, freezes a glossary, translates, performs the fixed single
reconciliation pass, validates the results, and writes the EPUB. Any terminal
page or gate failure prevents EPUB output and returns a nonzero exit status.
Terminal page failures stream to stderr once as they occur; the CLI emits one
final failure summary.

Preflight is mandatory and unresolved glossary or terminology review items
block the run until their resolution artifacts are supplied in the work
directory. These are pipeline gates, not optional CLI modes. Reconciliation is
fixed to one pass and the eval corpus is a developer harness, not a production
control.

## Production controls

- `--manifest-path PATH` selects a manifest explicitly. Without it, the runner
  creates or loads `INPUT_DIR/manifest.json`. Relative manifest paths are
  resolved beneath `INPUT_DIR`, never from the caller's cwd.
- `--glossary-budget N` controls the terminology consolidation budget. It
  defaults to 100,000 and is capped at 120,000.
- `--epub-check [--epub-check-path PATH]` strictly validates the generated
  EPUB with the named checker; supply a path when it is not on `PATH`.
- `--embed-images` embeds original page images in the generated EPUB.
- `--no-resume` bypasses the work-owned translation cache for this run.

Resource bounds: concurrency is 1–32, `--max-retries` is 1–10 total attempts,
and timeout is 1–3,600 seconds. `--pi-bin` selects the `pi` executable used by
the model leaves.
