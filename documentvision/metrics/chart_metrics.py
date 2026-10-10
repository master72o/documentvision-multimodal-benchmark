"""
ChartQA & Quantitative Figure Reasoning Evaluation Metrics.
"""

from typing import List, Dict, Any
import re
import numpy as np


class ChartMetricsCalculator:
    """Evaluates question answering and trend extraction over charts, plots, and infographics."""

    @staticmethod
    def extract_first_number(s: str) -> float:
        """Extracts first numeric value from a string (handles currency, percentages, commas)."""
        clean = re.sub(r"[\$,%]", "", str(s))
        match = re.search(r"[-+]?\d*\.?\d+", clean)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return 0.0
        return 0.0

    @classmethod
    def is_numerical_match(cls, pred: Any, gt: Any, tolerance: float = 0.05) -> bool:
        """Checks if prediction is within relative tolerance of ground truth."""
        try:
            val_pred = cls.extract_first_number(str(pred))
            val_gt = cls.extract_first_number(str(gt))

            denom = max(1e-5, abs(val_gt))
            rel_error = abs(val_pred - val_gt) / denom
            return bool(rel_error <= tolerance)
        except Exception:
            return False

    @classmethod
    def evaluate_chart_qa(
        cls,
        predicted_answers: List[Dict[str, Any]],
        ground_truth_answers: List[Dict[str, Any]],
        tolerance: float = 0.05,
    ) -> Dict[str, float]:
        """Compares list of predicted question-answer pairs against ground truth."""
        if not ground_truth_answers:
            return {"qa_accuracy": 0.0, "numerical_accuracy": 0.0, "trend_accuracy": 0.0}

        correct_total = 0
        num_count = 0
        num_correct = 0
        trend_count = 0
        trend_correct = 0

        # Map predictions by question_id or question
        pred_map = {
            p.get("question_id", p.get("question", "")): p.get("answer", "")
            for p in predicted_answers
        }

        for gt_item in ground_truth_answers:
            q_key = gt_item.get("question_id", gt_item.get("question", ""))
            gt_ans = str(gt_item.get("answer", "")).strip()
            pred_ans = str(pred_map.get(q_key, "")).strip()

            q_type = gt_item.get("question_type", "numerical")

            if q_type == "numerical":
                num_count += 1
                match = cls.is_numerical_match(pred_ans, gt_ans, tolerance=tolerance)
                if match:
                    num_correct += 1
                    correct_total += 1
            else:
                trend_count += 1
                match = (gt_ans.lower() in pred_ans.lower()) or (pred_ans.lower() in gt_ans.lower())
                if match:
                    trend_correct += 1
                    correct_total += 1

        total_q = len(ground_truth_answers)
        overall_acc = float(correct_total / total_q) if total_q > 0 else 0.0
        num_acc = float(num_correct / num_count) if num_count > 0 else 0.0
        trend_acc = float(trend_correct / trend_count) if trend_count > 0 else 0.0

        return {
            "qa_accuracy": round(overall_acc, 4),
            "numerical_accuracy": round(num_acc, 4),
            "trend_accuracy": round(trend_acc, 4),
        }
