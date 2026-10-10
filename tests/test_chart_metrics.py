"""
Unit Tests for ChartQA & Quantitative Figure Reasoning Metrics.
"""

import pytest
from documentvision.metrics.chart_metrics import ChartMetricsCalculator


def test_extract_first_number():
    assert ChartMetricsCalculator.extract_first_number("$4,821.50") == 4821.50
    assert ChartMetricsCalculator.extract_first_number("approx 94.8%") == 94.8
    assert ChartMetricsCalculator.extract_first_number("no number") == 0.0


def test_is_numerical_match():
    assert ChartMetricsCalculator.is_numerical_match("100.0", "102.0", tolerance=0.05) is True
    assert ChartMetricsCalculator.is_numerical_match("100.0", "110.0", tolerance=0.05) is False
    assert ChartMetricsCalculator.is_numerical_match("$94.8M", "94.8", tolerance=0.01) is True


def test_evaluate_chart_qa():
    gt = [
        {"question_id": "q1", "question": "What is revenue?", "answer": "100.0", "question_type": "numerical"},
        {"question_id": "q2", "question": "What is trend?", "answer": "increasing", "question_type": "trend"},
    ]
    pred_good = [
        {"question_id": "q1", "answer": "$101.5M"},
        {"question_id": "q2", "answer": "clearly increasing over time"},
    ]
    res = ChartMetricsCalculator.evaluate_chart_qa(pred_good, gt)
    assert res["qa_accuracy"] == 1.0
    assert res["numerical_accuracy"] == 1.0
    assert res["trend_accuracy"] == 1.0
