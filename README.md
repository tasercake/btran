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
for the complete production configuration surface.

Useful final-run controls (consumed by the Gate 1 orchestration integration):

- `--manifest-path PATH` selects a manifest explicitly. The default is
  `INPUT_DIR/manifest.json`; relative paths also resolve beneath `INPUT_DIR`,
  never from the caller's cwd.
- `--epub-check [--epub-check-path PATH]` strictly validates the generated
  EPUB; the checker must be resolvable on `PATH` (or supplied explicitly).
- `--review` enables the one allowed review/resolution pass.
- `--preflight-only` runs input validation without requiring `pi`.
- `--glossary-budget N` defaults to 100,000 and is capped at 120,000.

Resource bounds: concurrency is 1–32, `--max-retries` is 1–10 total attempts
(the existing pipeline meaning of that flag), and timeout is 1–3,600 seconds.

Preflight is always enabled. Reconciliation is fixed to one round, and the eval
corpus remains a developer harness rather than a production CLI control.

This CLI/configuration boundary intentionally does not alter orchestration.
The WP-7 runner consumes these controls; until that runner is integrated, the
legacy runner does not implement manifest, preflight-only, review, or EPUBCheck
behavior.

## Pipeline behavior after WP-7 integration

1. Builds or loads an input manifest and preflights every page
2. Translates pages through `pi` with bounded concurrency and attempts
3. Streams terminal page errors to stderr and fails the run if any page fails
4. Compiles all successful results into an EPUB with `ebooklib`
