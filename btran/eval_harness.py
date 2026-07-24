"""JSON-backed regression runner for deterministic translation validations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from btran.schema import PageExtraction, PageResult, TerminologyMap
from btran.validators import VALIDATION_STAGES, validate_page


@dataclass(frozen=True)
class EvalCase:
    """One self-contained corpus case loaded from ``case_*/config.json``."""

    name: str
    fixture_image: Path
    source: PageExtraction
    translation: PageResult
    glossary: TerminologyMap
    expected: dict[str, bool]


def load_eval_cases(corpus_dir: Path) -> list[EvalCase]:
    """Load all case configs, requiring each declared fixture image to exist."""
    cases: list[EvalCase] = []
    for config_path in sorted(Path(corpus_dir).glob("*/config.json")):
        config = json.loads(config_path.read_text())
        case_dir = config_path.parent
        image = case_dir / config["fixture_image"]
        if not image.is_file():
            raise FileNotFoundError(f"fixture image missing for {config['name']}: {image}")
        expected = config["expected"]
        if not isinstance(expected, dict) or set(expected) != set(VALIDATION_STAGES):
            raise ValueError(
                f"case {config['name']} expected stages must exactly match "
                f"{', '.join(VALIDATION_STAGES)}"
            )
        if not all(isinstance(value, bool) for value in expected.values()):
            raise ValueError(f"case {config['name']} expected stages must be booleans")
        cases.append(EvalCase(
            name=config["name"],
            fixture_image=image,
            source=PageExtraction.from_dict(config["source"]),
            translation=PageResult.from_dict(config["translation"]),
            glossary=TerminologyMap.from_dict(config["glossary"]),
            expected=expected,
        ))
    return cases


def run_corpus(corpus_dir: Path, report_path: Path | None = None) -> dict:
    """Run validation stages for every case and return a JSON-serializable report."""
    results = [_run_case(case) for case in load_eval_cases(corpus_dir)]
    report = {
        "cases": results,
        "passed": sum(case["passed"] for case in results),
        "failed": sum(not case["passed"] for case in results),
        "all_passed": all(case["passed"] for case in results),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _run_case(case: EvalCase) -> dict:
    errors = validate_page(case.source, case.translation, case.glossary)
    stages = {
        name: {
            "expected": expected,
            "actual": not bool(errors[name]),
            "passed": (not bool(errors[name])) == expected,
            "errors": errors[name],
        }
        for name, expected in case.expected.items()
    }
    return {
        "name": case.name,
        "passed": all(stage["passed"] for stage in stages.values()),
        "stages": stages,
    }
