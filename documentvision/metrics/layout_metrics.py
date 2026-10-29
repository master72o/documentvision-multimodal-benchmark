"""
Visual Document Layout Grounding & Bounding Box Metrics: IoU, AP@50, and AP@75.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from documentvision.core.schema import BoundingBox


class LayoutMetricsCalculator:
    """Evaluates document object detection, layout parsing, and bounding box grounding."""

    @classmethod
    def evaluate_layout_boxes(
        cls,
        predicted_boxes: List[Dict[str, Any]],
        ground_truth_boxes: List[Dict[str, Any]],
        iou_threshold_50: float = 0.50,
        iou_threshold_75: float = 0.75,
    ) -> Dict[str, float]:
        """Matches predicted bounding boxes against ground truth using greedy bipartite matching."""
        if not ground_truth_boxes and not predicted_boxes:
            return {"mean_iou": 1.0, "ap_50": 1.0, "ap_75": 1.0, "layout_f1": 1.0}
        if not ground_truth_boxes or not predicted_boxes:
            return {"mean_iou": 0.0, "ap_50": 0.0, "ap_75": 0.0, "layout_f1": 0.0}

        gt_objs = [
            BoundingBox(
                ymin=float(b["ymin"]),
                xmin=float(b["xmin"]),
                ymax=float(b["ymax"]),
                xmax=float(b["xmax"]),
                label=str(b.get("label", "")).lower(),
            )
            for b in ground_truth_boxes
        ]

        pred_objs = [
            BoundingBox(
                ymin=float(b["ymin"]),
                xmin=float(b["xmin"]),
                ymax=float(b["ymax"]),
                xmax=float(b["xmax"]),
                label=str(b.get("label", "")).lower(),
                confidence=float(b.get("confidence", 1.0)),
            )
            for b in predicted_boxes
        ]

        matched_ious = []
        tp_50 = 0
        tp_75 = 0
        matched_gt = set()

        for pred in pred_objs:
            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt in enumerate(gt_objs):
                if idx in matched_gt:
                    continue
                # Optional label match requirement
                if pred.label and gt.label and pred.label != gt.label:
                    continue

                curr_iou = pred.iou(gt)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = idx

            if best_gt_idx >= 0 and best_iou > 0.0:
                matched_ious.append(best_iou)
                matched_gt.add(best_gt_idx)
                if best_iou >= iou_threshold_50:
                    tp_50 += 1
                if best_iou >= iou_threshold_75:
                    tp_75 += 1

        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        n_pred = len(pred_objs)
        n_gt = len(gt_objs)

        prec_50 = float(tp_50 / n_pred) if n_pred > 0 else 0.0
        rec_50 = float(tp_50 / n_gt) if n_gt > 0 else 0.0
        f1_50 = float(2 * prec_50 * rec_50 / (prec_50 + rec_50)) if (prec_50 + rec_50) > 0 else 0.0

        ap_75 = float(tp_75 / max(n_pred, n_gt)) if max(n_pred, n_gt) > 0 else 0.0

        return {
            "mean_iou": round(mean_iou, 4),
            "ap_50": round(f1_50, 4),
            "ap_75": round(ap_75, 4),
            "layout_f1": round(f1_50, 4),
        }

    @classmethod
    def compute_map_50_95(
        cls,
        predicted_boxes: List[Dict[str, Any]],
        ground_truth_boxes: List[Dict[str, Any]],
    ) -> float:
        """
        Computes COCO-style mAP@[.50:.95] by averaging layout F1/AP across 10 IoU thresholds (0.50 to 0.95).
        """
        if not ground_truth_boxes and not predicted_boxes:
            return 1.0
        if not ground_truth_boxes or not predicted_boxes:
            return 0.0

        thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
        scores = []
        for thresh in thresholds:
            res = cls.evaluate_layout_boxes(
                predicted_boxes, ground_truth_boxes, iou_threshold_50=thresh
            )
            scores.append(res["ap_50"])
        return round(float(np.mean(scores)), 4)

    @classmethod
    def reading_order_concordance(
        cls,
        predicted_boxes: List[Dict[str, Any]],
        ground_truth_boxes: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluates whether predicted layout blocks preserve canonical top-to-bottom, left-to-right reading order.
        """
        if len(ground_truth_boxes) < 2 or len(predicted_boxes) < 2:
            return 1.0

        # Canonical reading key: (ymin, xmin)
        gt_sorted_labels = [
            str(b.get("label", idx)).lower()
            for idx, b in enumerate(
                sorted(ground_truth_boxes, key=lambda x: (float(x["ymin"]), float(x["xmin"])))
            )
        ]
        pred_order_labels = [
            str(b.get("label", idx)).lower() for idx, b in enumerate(predicted_boxes)
        ]

        common = [lbl for lbl in gt_sorted_labels if lbl in pred_order_labels]
        if len(common) < 2:
            return 0.0

        concordant = 0
        total_pairs = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                total_pairs += 1
                if pred_order_labels.index(common[i]) < pred_order_labels.index(common[j]):
                    concordant += 1

        return round(float(concordant / total_pairs), 4) if total_pairs > 0 else 1.0
