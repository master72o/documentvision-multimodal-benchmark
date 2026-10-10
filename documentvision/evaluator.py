"""
Unified Benchmark Evaluator for DocumentVision.
Orchestrates TEDS table evaluation, KIE metrics, ChartQA reasoning, and layout grounding.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from documentvision.core.schema import (
    DocumentTaskType,
    DocumentSample,
    EvaluationResult,
)
from documentvision.metrics.teds import TEDS
from documentvision.metrics.extraction_metrics import ExtractionMetricsCalculator
from documentvision.metrics.chart_metrics import ChartMetricsCalculator
from documentvision.metrics.layout_metrics import LayoutMetricsCalculator


class DocumentVisionEvaluator:
    """Benchmark orchestrator evaluating multimodal document models across diverse tasks."""

    MODELS = [
        ("Tesseract_OCR_Pipeline", "Tesseract OCR + Layout Heuristics (Baseline)"),
        ("DocOwl_1.5B", "mPLUG-DocOwl 1.5B (Specialized VLM)"),
        ("Qwen2_VL_7B", "Qwen2-VL 7B (Open-Weights Vision-Language Model)"),
        ("GPT4o_Vision", "GPT-4o Vision (Frontier Multimodal API)"),
    ]

    def __init__(self, samples: List[DocumentSample]):
        self.samples = samples
        self.teds_calc = TEDS(structure_only=False)
        self.teds_struct_calc = TEDS(structure_only=True)

    @classmethod
    def from_jsonl(cls, file_path: str) -> "DocumentVisionEvaluator":
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(DocumentSample.from_dict(json.loads(line)))
        return cls(samples)

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Runs complete benchmark suite across all models and tasks."""
        results_by_model: Dict[str, Dict[str, Any]] = {}

        for model_id, model_label in self.MODELS:
            kie_f1s = []
            kie_ems = []
            kie_levs = []

            teds_scores = []
            teds_struct_scores = []

            chart_accs = []
            chart_num_accs = []

            layout_ious = []
            layout_ap50s = []

            for s in self.samples:
                pred_dict = s.model_predictions.get(model_id, {})
                gt_dict = s.ground_truth

                if s.task_type == DocumentTaskType.KEY_VALUE_EXTRACTION:
                    metrics = ExtractionMetricsCalculator.evaluate_key_values(pred_dict, gt_dict)
                    kie_f1s.append(metrics["field_f1"])
                    kie_ems.append(metrics["exact_match"])
                    kie_levs.append(metrics["mean_levenshtein"])

                elif s.task_type == DocumentTaskType.TABLE_RECONSTRUCTION:
                    pred_html = str(pred_dict.get("html", ""))
                    gt_html = str(gt_dict.get("html", ""))
                    score = self.teds_calc.evaluate(pred_html, gt_html)
                    struct_score = self.teds_struct_calc.evaluate(pred_html, gt_html)
                    teds_scores.append(score)
                    teds_struct_scores.append(struct_score)

                elif s.task_type == DocumentTaskType.CHART_REASONING:
                    pred_answers = pred_dict.get("answers", [])
                    gt_questions = gt_dict.get("questions", [])
                    chart_res = ChartMetricsCalculator.evaluate_chart_qa(pred_answers, gt_questions)
                    chart_accs.append(chart_res["qa_accuracy"])
                    chart_num_accs.append(chart_res["numerical_accuracy"])

                elif s.task_type == DocumentTaskType.LAYOUT_GROUNDING:
                    pred_boxes = pred_dict.get("boxes", [])
                    gt_boxes = gt_dict.get("boxes", [])
                    layout_res = LayoutMetricsCalculator.evaluate_layout_boxes(pred_boxes, gt_boxes)
                    layout_ious.append(layout_res["mean_iou"])
                    layout_ap50s.append(layout_res["ap_50"])

            # Compute aggregated averages
            avg_kie_f1 = float(np.mean(kie_f1s)) if kie_f1s else 0.0
            avg_kie_em = float(np.mean(kie_ems)) if kie_ems else 0.0
            avg_kie_lev = float(np.mean(kie_levs)) if kie_levs else 0.0

            avg_teds = float(np.mean(teds_scores)) if teds_scores else 0.0
            avg_teds_struct = float(np.mean(teds_struct_scores)) if teds_struct_scores else 0.0

            avg_chart_acc = float(np.mean(chart_accs)) if chart_accs else 0.0
            avg_chart_num = float(np.mean(chart_num_accs)) if chart_num_accs else 0.0

            avg_layout_iou = float(np.mean(layout_ious)) if layout_ious else 0.0
            avg_layout_ap50 = float(np.mean(layout_ap50s)) if layout_ap50s else 0.0

            composite = (avg_kie_f1 + avg_teds + avg_chart_acc + avg_layout_ap50) / 4.0

            results_by_model[model_id] = {
                "label": model_label,
                "composite_score": round(composite, 4),
                "kie_field_f1": round(avg_kie_f1, 4),
                "kie_exact_match": round(avg_kie_em, 4),
                "kie_levenshtein": round(avg_kie_lev, 4),
                "teds": round(avg_teds, 4),
                "teds_struct": round(avg_teds_struct, 4),
                "chart_qa_acc": round(avg_chart_acc, 4),
                "chart_num_acc": round(avg_chart_num, 4),
                "layout_miou": round(avg_layout_iou, 4),
                "layout_ap50": round(avg_layout_ap50, 4),
            }

        return {
            "total_samples": len(self.samples),
            "models": results_by_model,
        }

    def export_summary_json(self, summary: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def export_predictions_csv(self, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        rows = []
        for s in self.samples:
            row = {
                "sample_id": s.sample_id,
                "task_type": s.task_type.value,
                "domain": s.domain,
                "image_path": s.image_path,
            }
            for model_id, _ in self.MODELS:
                pred = s.model_predictions.get(model_id, {})
                row[f"{model_id}_pred"] = json.dumps(pred)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)

    def generate_markdown_report(self, summary: Dict[str, Any], output_path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        rows = []
        for model_id, _ in self.MODELS:
            m = summary["models"][model_id]
            rows.append(
                f"| `{model_id}` | **{m['composite_score']*100:.1f}%** | {m['teds']*100:.1f}% | "
                f"{m['teds_struct']*100:.1f}% | {m['kie_field_f1']*100:.1f}% | {m['kie_exact_match']*100:.1f}% | "
                f"{m['chart_qa_acc']*100:.1f}% | {m['layout_ap50']*100:.1f}% | {m['layout_miou']*100:.1f}% |"
            )
        table_content = "\n".join(rows)

        report = f"""# DocumentVision: Multimodal Document Intelligence Benchmark Report

## Executive Summary
This benchmark evaluates **OCR Pipelines vs Vision-Language Models** across 4 dense multimodal document intelligence tasks:
1. **Dense Key-Value Information Extraction (KIE)**
2. **Table Structure & Cell Reconstruction (TEDS)**
3. **Chart & Figure Trend Reasoning (ChartQA)**
4. **Visual Layout Bounding Box Grounding (AP@50 & mIoU)**

---

## Controlled Model Leaderboard

| Model Checkpoint | Composite Score | TEDS (Full) | TEDS (Struct) | KIE Field F1 | KIE Exact Match | ChartQA Acc | Layout AP@50 | Layout mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_content}

---

## Key Experimental Findings

1. **Table Structure Collapse in Traditional OCR**:
   - The Tesseract pipeline achieves only **{summary['models']['Tesseract_OCR_Pipeline']['teds']*100:.1f}% TEDS**, frequently merging adjacent column cells when borders are light or absent.
   - Modern VLMs (Qwen2-VL & GPT-4o) retain near-perfect table hierarchies (**>{summary['models']['Qwen2_VL_7B']['teds']*100:.1f}% TEDS**).

2. **Complex Key-Value Association**:
   - On financial invoices and clinical lab panels, rule-based OCR achieves **{summary['models']['Tesseract_OCR_Pipeline']['kie_exact_match']*100:.1f}% Exact Match**, struggling with disconnected keys and values across horizontal white space.
   - Vision-language models achieve **>{summary['models']['Qwen2_VL_7B']['kie_exact_match']*100:.1f}% Exact Match** by leveraging spatial positional embeddings.

3. **Chart & Trend Reasoning Gap**:
   - Traditional OCR pipelines cannot perform quantitative trend reasoning ({summary['models']['Tesseract_OCR_Pipeline']['chart_qa_acc']*100:.1f}% accuracy), confusing bar chart series with axis coordinates.
   - Frontier VLMs solve chart reasoning at **{summary['models']['GPT4o_Vision']['chart_qa_acc']*100:.1f}% accuracy**.

---
*Report auto-generated by DocumentVision Multimodal Benchmark v1.0.0*
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return report
