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
`BTRAN_TARGET_LANG`). It detects the source language for every page during
extraction. CLI options override `.env` values. See `.env.example`
for the complete supported production configuration surface.

## Integrated pipeline

Each run discovers listed pages, extracts source blocks, freezes a glossary,
translates, performs the fixed single reconciliation pass, validates results,
and writes the EPUB. Page input or model failures retain diagnostic source
content so the EPUB remains inspectable. Findings are nonblocking; terminal
invocation failures prevent EPUB output and return a nonzero exit status.

Reconciliation is fixed to one pass and the eval corpus is a developer harness,
not a production control.

## Timing reports

Every DAG stage is measured with a monotonic clock. On completion, the CLI prints
one `btran timing_ms` line containing the aggregate and per-stage durations. The
same values are persisted as each stage record's `duration_ms` plus
`total_stage_duration_ms` in `WORKSPACE/reports/RUN_ID.json`. Timings are
execution metadata: they do not affect cache keys, content artifacts, or revision
selection.

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

`--max-retries` is 0–5; zero still performs the initial model attempt.
`--timeout` is 1–3600 seconds and bounds terminology consolidation and
EPUBCheck only. Source extraction and translation Pi calls have no execution
deadline; they run until exit, failure, or parent-task cancellation. `--pi-bin`
selects the `pi` executable used by model leaves.

## Deterministic eval corpus

`eval_corpus/case_*/` contains locally authored, synthetic regression fixtures;
it contains no scanned book pages, private inputs, or copyrighted source text.
Each directory has a locally generated `page.png` and a `config.json` that
supplies the source artifact, translated artifact, glossary, expected result
for every validator stage, and non-empty `risk_tags`. The source and
translation artifacts record the fixture's actual SHA-256 and perceptual hash;
the harness rejects a config whose identities do not match its PNG. Tags
describe the reviewed hard page condition (for example `tables`,
`low-resolution-risk`, or `block-id-mismatch`), rather than a claim about model
quality.

Run the corpus with:

```bash
pytest btran/tests/test_eval_harness.py -q
# or write an inspectable report
python -c 'from pathlib import Path; from btran.eval_harness import run_corpus; print(run_corpus(Path("eval_corpus"), Path("eval-report.json")))'
```

To add a reviewed failure, create one case directory with a synthetic fixture
that you generated yourself, a complete `config.json`, at least one meaningful
risk tag, and explicit `true`/`false` expectations for every validation stage.
Keep the corpus between 20 and 50 cases; extend category coverage instead of
adding near-duplicates. The repository test gates the required categories,
valid and deliberately invalid outcomes for every validator stage, and fixture
size (16 KiB each / 200 KiB total). A deliberately invalid artifact belongs in
the corpus only when its expected validator failure is recorded and reviewed.

This corpus exercises JSON artifacts and deterministic validators only. It does
not measure live OCR, translation fluency, semantic adequacy, visual extraction
accuracy, or provider reliability; it has no LLM judge and makes no live
provider calls. In particular, terminology context-variant cases are
preclassified glossary artifacts: they demonstrate permitted entries, not
semantic sense disambiguation by the validator.
