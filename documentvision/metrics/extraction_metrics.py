"""
Key-Value Extraction (KIE) Metrics: Exact Match, Normalized Levenshtein, and Field F1.
"""

from typing import Dict, Any, Tuple
from typing import List, Sequence
import re
import numpy as np
from documentvision.metrics.teds import levenshtein_distance


class ExtractionMetricsCalculator:
    """Calculates field-level extraction accuracy, similarity, and F1 score."""

    @staticmethod
    def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        s1_clean = str(s1).strip()
        s2_clean = str(s2).strip()
        if s1_clean == s2_clean:
            return 1.0
        max_len = max(len(s1_clean), len(s2_clean))
        if max_len == 0:
            return 1.0
        dist = levenshtein_distance(s1_clean, s2_clean)
        return float(max(0.0, 1.0 - (dist / max_len)))

    @classmethod
    def evaluate_key_values(
        cls,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
        similarity_threshold: float = 0.85,
    ) -> Dict[str, float]:
        """Compares predicted key-value dictionary against ground truth dictionary."""
        if not ground_truth and not predicted:
            return {"exact_match": 1.0, "mean_levenshtein": 1.0, "field_precision": 1.0, "field_recall": 1.0, "field_f1": 1.0}
        if not ground_truth or not predicted:
            return {"exact_match": 0.0, "mean_levenshtein": 0.0, "field_precision": 0.0, "field_recall": 0.0, "field_f1": 0.0}

        exact_matches = 0
        lev_scores = []
        true_positives = 0

        for key, gt_val in ground_truth.items():
            if key in predicted:
                pred_val = predicted[key]
                is_exact = str(gt_val).strip().lower() == str(pred_val).strip().lower()
                if is_exact:
                    exact_matches += 1

                sim = cls.normalized_levenshtein_similarity(str(gt_val), str(pred_val))
                lev_scores.append(sim)

                if sim >= similarity_threshold:
                    true_positives += 1
            else:
                lev_scores.append(0.0)

        total_gt = len(ground_truth)
        total_pred = len(predicted)

        em_rate = float(exact_matches / total_gt)
        mean_lev = float(np.mean(lev_scores)) if lev_scores else 0.0

        precision = float(true_positives / total_pred) if total_pred > 0 else 0.0
        recall = float(true_positives / total_gt) if total_gt > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "exact_match": round(em_rate, 4),
            "mean_levenshtein": round(mean_lev, 4),
            "field_precision": round(precision, 4),
            "field_recall": round(recall, 4),
            "field_f1": round(f1, 4),
        }

    @classmethod
    def compute_anls(
        cls,
        predicted_answer: str,
        ground_truth_answers: Sequence[str],
        threshold: float = 0.5,
    ) -> float:
        """
        ICDAR DocVQA Average Normalized Levenshtein Similarity (ANLS).
        Computes max normalized similarity across human reference answers and clips to 0.0 if < threshold.
        """
        if not ground_truth_answers:
            return 1.0 if not str(predicted_answer).strip() else 0.0

        pred_norm = " ".join(str(predicted_answer).strip().lower().split())
        best_sim = 0.0
        for gt in ground_truth_answers:
            gt_norm = " ".join(str(gt).strip().lower().split())
            sim = cls.normalized_levenshtein_similarity(pred_norm, gt_norm)
            if sim > best_sim:
                best_sim = sim

        return round(float(best_sim), 4) if best_sim >= threshold else 0.0

    @staticmethod
    def numeric_value_match(
        predicted_val: str,
        ground_truth_val: str,
        rel_tol: float = 0.01,
    ) -> bool:
        """
        Normalizes currency/formatting symbols ($, commas, %) and checks numeric equivalence within rel_tol.
        """
        def _parse_num(val: str):
            cleaned = re.sub(r"[$€£,\s%]", "", str(val).strip())
            match = re.search(r"[-+]?\d*\.?\d+", cleaned)
            return float(match.group(0)) if match else None

        p_num = _parse_num(predicted_val)
        g_num = _parse_num(ground_truth_val)
        if p_num is None or g_num is None:
            return False
        if g_num == 0.0:
            return abs(p_num) <= rel_tol
        return abs(p_num - g_num) / abs(g_num) <= rel_tol
