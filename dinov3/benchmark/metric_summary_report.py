#!/usr/bin/env python3
"""Summarize all standardized segmentation metric results."""

from __future__ import annotations

import json
from pathlib import Path

from metric_common import OUTPUT_ROOT, SUMMARY_DIR, ensure_dir, save_json

RESULT_FILE = "metric_summary_report.json"

SEM_ACC_RESULT = OUTPUT_ROOT / "metric_semantic_segmentation_accuracy" / "semantic_segmentation_accuracy_result.json"
LATENCY_RESULT = OUTPUT_ROOT / "metric_real_time_segmentation_latency" / "real_time_segmentation_latency_result.json"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    report = {
        "semantic_segmentation_accuracy": read_json(SEM_ACC_RESULT),
        "real_time_segmentation_latency": read_json(LATENCY_RESULT),
    }

    summary = {
        "report_type": "metric_summary",
        "source_files": {
            "semantic_segmentation_accuracy": str(SEM_ACC_RESULT),
            "real_time_segmentation_latency": str(LATENCY_RESULT),
        },
        "report": report,
    }

    ensure_dir(SUMMARY_DIR)
    out = SUMMARY_DIR / RESULT_FILE
    save_json(out, summary)
    print(f"[INFO] summary JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
