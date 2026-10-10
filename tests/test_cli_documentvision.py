"""
Unit Tests for DocumentVision CLI Commands.
"""

import os
import json
import subprocess
import sys
import pytest


def test_cli_benchmark_command(tmp_path):
    mini_data = [
        {
            "sample_id": "mini_01",
            "task_type": "key_value_extraction",
            "domain": "test",
            "image_path": "fake.png",
            "ground_truth": {"name": "Alice", "total": "100"},
            "model_predictions": {
                "Tesseract_OCR_Pipeline": {"name": "Alice", "total": "100"},
                "DocOwl_1.5B": {"name": "Alice", "total": "100"},
                "Qwen2_VL_7B": {"name": "Alice", "total": "100"},
                "GPT4o_Vision": {"name": "Alice", "total": "100"},
            },
        }
    ]
    ds_path = tmp_path / "mini.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for d in mini_data:
            f.write(json.dumps(d) + "\n")

    out_dir = tmp_path / "results"
    rep_path = tmp_path / "report.md"

    cmd = [
        sys.executable,
        "-m",
        "documentvision.cli",
        "benchmark",
        "--dataset",
        str(ds_path),
        "--output-dir",
        str(out_dir),
        "--report-path",
        str(rep_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Benchmark completed successfully" in res.stdout
    assert os.path.exists(out_dir / "documentvision_summary.json")
    assert os.path.exists(out_dir / "documentvision_predictions.csv")
    assert os.path.exists(rep_path)


def test_cli_evaluate_teds_command():
    pred_html = "<table><tr><td>A</td></tr></table>"
    gt_html = "<table><tr><td>A</td></tr></table>"

    cmd = [
        sys.executable,
        "-m",
        "documentvision.cli",
        "evaluate-teds",
        "--pred",
        pred_html,
        "--gt",
        gt_html,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "100.00%" in res.stdout


def test_cli_synthesize_docs_command(tmp_path):
    out_dir = tmp_path / "imgs"

    cmd = [
        sys.executable,
        "-m",
        "documentvision.cli",
        "synthesize-docs",
        "--output-dir",
        str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists(out_dir / "invoice_001.png")
