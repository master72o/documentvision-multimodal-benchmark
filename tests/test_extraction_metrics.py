"""
Unit Tests for Key-Value Extraction (KIE) Metrics.
"""

import pytest
from documentvision.metrics.extraction_metrics import ExtractionMetricsCalculator


def test_normalized_levenshtein_similarity():
    sim_exact = ExtractionMetricsCalculator.normalized_levenshtein_similarity("INV-100", "INV-100")
    assert sim_exact == 1.0

    sim_typo = ExtractionMetricsCalculator.normalized_levenshtein_similarity("INV-100", "INV-10O")
    assert sim_typo > 0.80

    sim_diff = ExtractionMetricsCalculator.normalized_levenshtein_similarity("Apple", "Banana")
    assert sim_diff < 0.50


def test_evaluate_key_values_perfect():
    gt = {"invoice_id": "123", "total": "$100.00"}
    pred = {"invoice_id": "123", "total": "$100.00"}

    res = ExtractionMetricsCalculator.evaluate_key_values(pred, gt)
    assert res["exact_match"] == 1.0
    assert res["field_f1"] == 1.0
    assert res["mean_levenshtein"] == 1.0


def test_evaluate_key_values_partial():
    gt = {"invoice_id": "123", "total": "$100.00", "date": "2026-10-01"}
    pred = {"invoice_id": "123", "total": "$100.0O"}  # Typo in total, missing date

    res = ExtractionMetricsCalculator.evaluate_key_values(pred, gt)
    assert res["exact_match"] == pytest.approx(1 / 3, abs=1e-3)
    assert 0.0 < res["field_f1"] < 1.0


def test_compute_anls_and_numeric_value_match():
    # Exact or minor OCR typo stays above 0.5 threshold
    assert ExtractionMetricsCalculator.compute_anls("INV-2026-99", ["inv-2026-99", "2026-99"]) == 1.0
    assert ExtractionMetricsCalculator.compute_anls("INV-2026-9O", ["INV-2026-99"]) > 0.85
    # Completely wrong answer below 0.5 threshold clips to 0.0
    assert ExtractionMetricsCalculator.compute_anls("Unrelated text", ["INV-2026-99"]) == 0.0

    # Numeric value matching with currency/commas
    assert ExtractionMetricsCalculator.numeric_value_match("$1,250.00", "1250.0") is True
    assert ExtractionMetricsCalculator.numeric_value_match("12.5%", "12.50") is True
    assert ExtractionMetricsCalculator.numeric_value_match("$900.00", "$1,250.00") is False
