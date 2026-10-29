"""
Unit Tests for Layout Grounding & Bounding Box Metrics.
"""

import pytest
from documentvision.core.schema import BoundingBox
from documentvision.metrics.layout_metrics import LayoutMetricsCalculator


def test_bounding_box_iou():
    box_a = BoundingBox(ymin=0, xmin=0, ymax=100, xmax=100)
    box_b = BoundingBox(ymin=0, xmin=0, ymax=100, xmax=100)
    assert box_a.iou(box_b) == 1.0

    box_c = BoundingBox(ymin=200, xmin=200, ymax=300, xmax=300)
    assert box_a.iou(box_c) == 0.0

    # 50% overlap along one axis
    box_d = BoundingBox(ymin=0, xmin=0, ymax=100, xmax=50)
    assert box_a.iou(box_d) == pytest.approx(0.5, abs=1e-3)


def test_evaluate_layout_boxes():
    gt = [
        {"ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100, "label": "header"},
        {"ymin": 200, "xmin": 0, "ymax": 300, "xmax": 100, "label": "footer"},
    ]
    pred = [
        {"ymin": 5, "xmin": 5, "ymax": 95, "xmax": 95, "label": "header"},
        {"ymin": 205, "xmin": 5, "ymax": 295, "xmax": 95, "label": "footer"},
    ]
    res = LayoutMetricsCalculator.evaluate_layout_boxes(pred, gt)
    assert res["mean_iou"] > 0.70
    assert res["ap_50"] == 1.0


def test_compute_map_50_95_and_reading_order():
    gt = [
        {"ymin": 10, "xmin": 10, "ymax": 60, "xmax": 200, "label": "title"},
        {"ymin": 80, "xmin": 10, "ymax": 180, "xmax": 200, "label": "table"},
        {"ymin": 200, "xmin": 10, "ymax": 240, "xmax": 200, "label": "footer"},
    ]
    pred_exact = list(gt)
    assert LayoutMetricsCalculator.compute_map_50_95(pred_exact, gt) == 1.0
    assert LayoutMetricsCalculator.reading_order_concordance(pred_exact, gt) == 1.0

    # Reversed order
    pred_reversed = list(reversed(gt))
    assert LayoutMetricsCalculator.reading_order_concordance(pred_reversed, gt) == 0.0
