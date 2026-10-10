"""
DocumentVision: Multimodal Document Intelligence Benchmark Studio.
Interactive Streamlit Application for OCR vs VLM Evaluation, TEDS Table Inspection, and Layout Grounding.
"""

import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from documentvision.core.schema import DocumentTaskType, DocumentSample
from documentvision.evaluator import DocumentVisionEvaluator
from documentvision.metrics.teds import TEDS
from documentvision.metrics.extraction_metrics import ExtractionMetricsCalculator
from documentvision.metrics.chart_metrics import ChartMetricsCalculator
from documentvision.metrics.layout_metrics import LayoutMetricsCalculator

# Streamlit Page Config
st.set_page_config(
    page_title="DocumentVision | Multimodal Document Benchmark",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .delta-good { color: #10B981; }
    .delta-bad { color: #EF4444; }
</style>
""", unsafe_allow_html=True)

# Load Benchmark
@st.cache_data
def load_evaluator(dataset_path: str = "data/ground_truth.jsonl"):
    if not os.path.exists(dataset_path):
        alt_path = os.path.join("repos", "documentvision-multimodal-benchmark", dataset_path)
        if os.path.exists(alt_path):
            dataset_path = alt_path
    return DocumentVisionEvaluator.from_jsonl(dataset_path)

DATASET_PATH = "data/ground_truth.jsonl"
try:
    evaluator = load_evaluator(DATASET_PATH)
except Exception as e:
    st.error(f"Error loading benchmark: {e}")
    st.stop()

# Run Evaluation
summary = evaluator.run_full_benchmark()

# Sidebar
st.sidebar.title("📄 DocumentVision")
st.sidebar.caption("Multimodal Document Intelligence Benchmark (OCR vs VLM)")

st.sidebar.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/master72o/documentvision-multimodal-benchmark)
    [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
    """
)

selected_model = st.sidebar.selectbox(
    "Active Model Checkpoint",
    options=[m[0] for m in DocumentVisionEvaluator.MODELS],
    format_func=lambda x: dict(DocumentVisionEvaluator.MODELS)[x],
)

st.sidebar.divider()
st.sidebar.info(
    f"📊 **Benchmark Dataset**: {len(evaluator.samples)} multimodal document tasks across "
    "Key-Value Extraction, Table Reconstruction, ChartQA, and Layout Grounding."
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Leaderboard & KPIs",
    "🖼️ Visual Grounding Inspector",
    "📊 Table Structure & TEDS",
    "📈 ChartQA Reasoning",
    "⚠️ OCR Failure Mode Gallery",
])

# ----------------- TAB 1: Leaderboard & KPIs -----------------
with tab1:
    st.subheader("Multimodal Document Intelligence Leaderboard")
    st.markdown(
        "Benchmarking traditional OCR pipelines (Tesseract + rule-based layout heuristics) against specialized "
        "and frontier Vision-Language Models across dense document understanding tasks."
    )

    curr_m = summary["models"][selected_model]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Composite Score</div>
            <div class="metric-value">{curr_m['composite_score']*100:.1f}%</div>
            <div class="metric-delta delta-good">Average across 4 tasks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">TEDS Table Score</div>
            <div class="metric-value">{curr_m['teds']*100:.1f}%</div>
            <div class="metric-delta {'delta-good' if curr_m['teds'] > 0.8 else 'delta-bad'}">IBM PubTabNet standard</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">KIE Field F1</div>
            <div class="metric-value">{curr_m['kie_field_f1']*100:.1f}%</div>
            <div class="metric-delta delta-good">Exact Match: {curr_m['kie_exact_match']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ChartQA Accuracy</div>
            <div class="metric-value">{curr_m['chart_qa_acc']*100:.1f}%</div>
            <div class="metric-delta delta-good">Numerical & Trend QA</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Layout AP@50</div>
            <div class="metric-value">{curr_m['layout_ap50']*100:.1f}%</div>
            <div class="metric-delta delta-good">mIoU: {curr_m['layout_miou']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Comparative Model Leaderboard")
    leaderboard_data = []
    for m_id, label in DocumentVisionEvaluator.MODELS:
        m = summary["models"][m_id]
        leaderboard_data.append({
            "Model Checkpoint": m_id,
            "Architecture": "VLM" if "VLM" in label or "GPT" in label or "Owl" in label else "OCR Pipeline",
            "Composite Score": f"{m['composite_score']*100:.1f}%",
            "TEDS (Full)": f"{m['teds']*100:.1f}%",
            "TEDS (Struct)": f"{m['teds_struct']*100:.1f}%",
            "KIE Field F1": f"{m['kie_field_f1']*100:.1f}%",
            "KIE Exact Match": f"{m['kie_exact_match']*100:.1f}%",
            "ChartQA Acc": f"{m['chart_qa_acc']*100:.1f}%",
            "Layout AP@50": f"{m['layout_ap50']*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)

# ----------------- TAB 2: Visual Grounding Inspector -----------------
with tab2:
    st.subheader("🖼️ Visual Grounding & Layout Bounding Box Inspector")
    st.markdown("Inspect visual localization of document entities (headers, tables, totals, metadata).")

    layout_samples = [s for s in evaluator.samples if s.task_type == DocumentTaskType.LAYOUT_GROUNDING]
    if layout_samples:
        sample = layout_samples[0]
        img_p = sample.image_path
        if not os.path.exists(img_p):
            img_p = os.path.join("repos", "documentvision-multimodal-benchmark", sample.image_path)

        if os.path.exists(img_p):
            base_img = Image.open(img_p)

            # Draw Ground Truth Boxes
            gt_img = base_img.copy()
            draw_gt = ImageDraw.Draw(gt_img)
            w, h = gt_img.size

            gt_boxes = sample.ground_truth.get("boxes", [])
            for b in gt_boxes:
                ymin, xmin, ymax, xmax = b["ymin"] * h / 1000, b["xmin"] * w / 1000, b["ymax"] * h / 1000, b["xmax"] * w / 1000
                draw_gt.rectangle([xmin, ymin, xmax, ymax], outline=(16, 185, 129), width=4)
                draw_gt.text((xmin + 8, ymin + 6), b.get("label", ""), fill=(16, 185, 129))

            # Draw Model Predicted Boxes
            pred_img = base_img.copy()
            draw_pred = ImageDraw.Draw(pred_img)
            pred_boxes = sample.model_predictions.get(selected_model, {}).get("boxes", [])
            for b in pred_boxes:
                ymin, xmin, ymax, xmax = b["ymin"] * h / 1000, b["xmin"] * w / 1000, b["ymax"] * h / 1000, b["xmax"] * w / 1000
                draw_pred.rectangle([xmin, ymin, xmax, ymax], outline=(67, 97, 238), width=4)
                draw_pred.text((xmin + 8, ymin + 6), b.get("label", ""), fill=(67, 97, 238))

            col_gt, col_pred = st.columns(2)
            with col_gt:
                st.markdown("**Ground Truth Annotations** (Green)")
                st.image(gt_img, use_container_width=True)
            with col_pred:
                st.markdown(f"**Predicted Bounding Boxes: {selected_model}** (Blue)")
                st.image(pred_img, use_container_width=True)

            # Box metrics
            box_metrics = LayoutMetricsCalculator.evaluate_layout_boxes(pred_boxes, gt_boxes)
            st.markdown(f"**Layout Localization Quality**: mIoU = `{box_metrics['mean_iou']*100:.1f}%` | AP@50 = `{box_metrics['ap_50']*100:.1f}%` | AP@75 = `{box_metrics['ap_75']*100:.1f}%`")
        else:
            st.warning("Document image file not found.")

# ----------------- TAB 3: Table Structure & TEDS -----------------
with tab3:
    st.subheader("📊 Table Structure Reconstruction & TEDS Analysis")
    st.markdown(
        "Tree Edit Distance-based Similarity (TEDS) measures both HTML structure alignment and cell content precision. "
        "Traditional OCR pipelines collapse when table lines are subtle or multi-line cells wrap."
    )

    table_samples = [s for s in evaluator.samples if s.task_type == DocumentTaskType.TABLE_RECONSTRUCTION]
    tbl_idx = st.selectbox("Select Table Document", range(len(table_samples)), format_func=lambda i: f"{table_samples[i].sample_id}: {table_samples[i].domain}")
    t_sample = table_samples[tbl_idx]

    gt_html = t_sample.ground_truth.get("html", "")
    pred_html = t_sample.model_predictions.get(selected_model, {}).get("html", "")

    teds_engine = TEDS(structure_only=False)
    teds_struct_engine = TEDS(structure_only=True)
    score_full = teds_engine.evaluate(pred_html, gt_html)
    score_struct = teds_struct_engine.evaluate(pred_html, gt_html)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.metric("Full TEDS (Content + Layout)", f"{score_full*100:.2f}%")
    with t_col2:
        st.metric("Structural TEDS (Layout Only)", f"{score_struct*100:.2f}%")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Ground Truth Table HTML**")
        st.components.v1.html(gt_html, height=350, scrolling=True)
    with col_t2:
        st.markdown(f"**{selected_model} Reconstructed Table**")
        st.components.v1.html(pred_html, height=350, scrolling=True)

# ----------------- TAB 4: ChartQA Reasoning -----------------
with tab4:
    st.subheader("📈 Chart & Quantitative Trend Reasoning (ChartQA)")
    st.markdown("Testing whether models can parse data plots, identify numeric points, and reason about growth trends.")

    chart_samples = [s for s in evaluator.samples if s.task_type == DocumentTaskType.CHART_REASONING]
    if chart_samples:
        c_sample = chart_samples[0]
        c_img_p = c_sample.image_path
        if not os.path.exists(c_img_p):
            c_img_p = os.path.join("repos", "documentvision-multimodal-benchmark", c_sample.image_path)

        c_col_img, c_col_qa = st.columns([1, 1])
        with c_col_img:
            if os.path.exists(c_img_p):
                st.image(c_img_p, caption="Evaluated Chart: Enterprise AI Cloud Quarterly Revenue", use_container_width=True)

        with c_col_qa:
            st.markdown(f"**Question-Answer Verification: {selected_model}**")
            pred_answers = c_sample.model_predictions.get(selected_model, {}).get("answers", [])
            gt_questions = c_sample.ground_truth.get("questions", [])

            pred_ans_map = {p.get("question_id"): p.get("answer") for p in pred_answers}

            for q in gt_questions:
                qid = q.get("question_id")
                q_text = q.get("question")
                gt_ans = q.get("answer")
                pred_ans = pred_ans_map.get(qid, "N/A")

                q_type = q.get("question_type", "numerical")
                if q_type == "numerical":
                    is_match = ChartMetricsCalculator.is_numerical_match(pred_ans, gt_ans)
                else:
                    is_match = gt_ans.lower() in str(pred_ans).lower()

                icon = "✅" if is_match else "❌"
                with st.expander(f"{icon} {q_text}", expanded=True):
                    st.write(f"- **Ground Truth**: `{gt_ans}`")
                    st.write(f"- **Model Prediction**: `{pred_ans}`")

# ----------------- TAB 5: OCR Failure Mode Gallery -----------------
with tab5:
    st.subheader("⚠️ OCR Failure Mode Gallery & Structural Collapse Diagnostics")
    st.markdown("Why traditional OCR pipelines fail in high-stakes enterprise document intelligence:")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.error("**1. Column Merging & Desynchronization**")
        st.write(
            "Tesseract and standard OCR rely on whitespace projection profiles. When tables lack explicit vertical gridlines, "
            "adjacent numerical columns get merged into a single cell string (e.g. `$4,821.5 $3,940.2`), causing severe downstream financial data errors."
        )
    with f_col2:
        st.warning("**2. Isolated Key-Value Disconnect**")
        st.write(
            "In modern invoices, keys (`TOTAL DUE:`) and values (`$70,986.25`) are separated by wide horizontal whitespace. "
            "Rule-based OCR links values to the wrong keys or drops labels completely without 2D spatial attention."
        )
    with f_col3:
        st.info("**3. Blindness to Infographics & Trends**")
        st.write(
            "Traditional OCR treats plots as raw text tokens, confusing bar values with tick mark coordinates. "
            "Only Vision-Language Models with patch embeddings can correlate bar heights to axes and infer trends."
        )
