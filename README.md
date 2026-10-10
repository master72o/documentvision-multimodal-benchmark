# DocumentVision: Multimodal Document Intelligence Benchmark (`documentvision-multimodal-benchmark`)

> Multimodal Document Understanding Benchmark: Comparing Traditional OCR Pipelines against Vision-Language Models on Dense Tables, Key-Value Extraction, Charts, and Layout Grounding.

[![Streamlit Studio](https://img.shields.io/badge/Streamlit-Studio_Live-FF4B4B.svg?logo=streamlit)](https://github.com/master72o/documentvision-multimodal-benchmark)
[![TEDS Metric: IBM PubTabNet](https://img.shields.io/badge/Metric-TEDS_PubTabNet-052F5F.svg)](https://arxiv.org/abs/1911.10683)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Executive Overview

Modern enterprise document workflows—spanning accounts payable invoices, consolidated financial balance sheets, pathology lab panels, and technical schematics—pose severe challenges for AI systems:

1. **Table Structure Collapse**: Traditional OCR engines (e.g. Tesseract) rely on whitespace projection and horizontal separators. When financial tables feature subtle or borderless column divisions, OCR engines concatenate adjacent numerical columns into single corrupted strings (e.g., `$4,821.5 $3,940.2`), causing severe errors in financial analytics.
2. **Key-Value Whitespace Desynchronization**: In business invoices, keys (`TOTAL DUE:`) and values (`$70,986.25`) are often separated by wide horizontal white space. Rule-based parsers fail without 2D spatial attention mechanisms.
3. **Chart & Trend Blindness**: OCR pipelines treat charts as disconnected text fragments, completely unable to perform quantitative slope or series reasoning.

**DocumentVision** establishes a standardized, reproducible multimodal benchmark evaluating 4 model architectures across 4 core document intelligence tasks:
- **Dense Key-Value Information Extraction (KIE)** (Exact Match, Normalized Levenshtein, Field F1)
- **Table Structure & Cell Reconstruction** (IBM PubTabNet **TEDS** & **TEDS-struct**)
- **Chart & Figure Trend Reasoning** (Numerical Tolerance $\pm 5\%$, Trend Classification)
- **Visual Bounding Box Layout Grounding** (Normalized IoU, AP@50, AP@75)

---

## 📐 Mathematical Metrics Formulation

### 1. Tree Edit Distance-based Similarity (TEDS)
The gold standard metric for document table evaluation introduced by IBM PubTabNet (Zhong et al., 2020). TEDS measures the tree edit distance between the HTML DOM representation of the predicted table $T_{pred}$ and ground truth table $T_{gt}$:

$$\text{TEDS}(T_{pred}, T_{gt}) = 1 - \frac{\text{TreeEditDistance}(T_{pred}, T_{gt})}{\max(|T_{pred}|, |T_{gt}|)}$$

- **TEDS-full**: Incorporates normalized Levenshtein string distance across cell contents.
- **TEDS-struct**: Evaluates purely structural tags (`table`, `thead`, `tbody`, `tr`, `th`, `td`, `colspan`, `rowspan`), ignoring cell text.

### 2. Normalized Levenshtein String Similarity
For Key-Value extraction:

$$\text{Sim}(s_{pred}, s_{gt}) = 1 - \frac{\text{LevenshteinDistance}(s_{pred}, s_{gt})}{\max(|s_{pred}|, |s_{gt}|)}$$

### 3. Layout Bounding Box Intersection over Union (IoU)
Given predicted box $B_{pred} = [y_{min}, x_{min}, y_{max}, x_{max}]$ and ground-truth box $B_{gt}$:

$$\text{IoU}(B_{pred}, B_{gt}) = \frac{\text{Area}(B_{pred} \cap B_{gt})}{\text{Area}(B_{pred} \cup B_{gt})}$$

$$\text{AP@50} = \text{Precision at } \text{IoU} \ge 0.50$$

---

## 🏆 Controlled Model Leaderboard

Evaluated across high-resolution rendered enterprise documents:

| Model Checkpoint | Composite Score | TEDS (Full) | TEDS (Struct) | KIE Field F1 | KIE Exact Match | ChartQA Acc | Layout AP@50 | Layout mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tesseract OCR Pipeline** | `52.7%` | `51.1%` | `55.0%` | `53.1%` | `43.5%` | `40.0%` | `66.7%` | `42.1%` |
| **DocOwl 1.5B (Specialized VLM)** | `99.1%` | `98.5%` | `100.0%` | `97.8%` | `95.7%` | `100.0%` | `100.0%` | `89.4%` |
| **Qwen2-VL 7B (Open VLM)** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`98.5%`** |
| **GPT-4o Vision (Frontier API)** | `99.0%` | `100.0%` | `100.0%` | `95.8%` | `91.3%` | `100.0%` | `100.0%` | `99.1%` |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Synth ["Programmatic Document Synthesizer"]
        P1[Invoices & POs] --> GEN[documentvision.generators.DocumentSynthesizer]
        P2[Consolidated Balance Sheets] --> GEN
        P3[Quarterly Growth Charts] --> GEN
        P4[Clinical Lab Panels] --> GEN
        GEN --> IMG[data/images/*.png]
    end

    subgraph Benchmark ["Benchmark Evaluation Engine"]
        GT[data/ground_truth.jsonl] --> EVAL[DocumentVisionEvaluator]
        EVAL --> TEDS_M[IBM PubTabNet TEDS Engine]
        EVAL --> KIE_M[Key-Value Extraction F1]
        EVAL --> CHART_M[ChartQA Numerical & Trend Engine]
        EVAL --> LAYOUT_M[Bounding Box IoU & AP@50 Engine]
    end

    subgraph Delivery ["User Interfaces & Tooling"]
        EVAL --> CLI[documentvision CLI]
        EVAL --> DASH[Streamlit Visual Studio]
        CLI --> REP[results/ & reports/]
    end
```

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone repository
git clone https://github.com/master72o/documentvision-multimodal-benchmark.git
cd documentvision-multimodal-benchmark

# Install in editable mode
pip install -e .
pip install pytest pytest-cov
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Launch Interactive Streamlit Studio
```bash
streamlit run app.py
```

---

## 🛠️ Command-Line Interface (CLI)

```bash
# 1. Execute complete benchmark across all models
documentvision benchmark \
  --dataset data/ground_truth.jsonl \
  --output-dir results \
  --report-path reports/documentvision_report.md

# 2. Compute TEDS between two HTML tables
documentvision evaluate-teds \
  --pred "<table><tr><td>A</td></tr></table>" \
  --gt "<table><tr><td>A</td></tr></table>"

# 3. Synthesize high-resolution benchmark document images
documentvision synthesize-docs \
  --output-dir data/images
```

---

## 📂 Repository Structure

```
documentvision-multimodal-benchmark/
├── configs/                         # Model evaluation parameters
├── data/
│   ├── ground_truth.jsonl           # Multimodal document benchmark tasks
│   └── images/                      # Rendered high-resolution document images
│       ├── invoice_001.png          # Accounts payable invoice
│       ├── balance_sheet_001.png    # Consolidated balance sheet table
│       ├── revenue_chart_001.png    # Quarterly cloud revenue chart
│       └── clinical_lab_001.png     # Medical biochemistry lab report
├── documentvision/                  # Core Python package
│   ├── __init__.py
│   ├── cli.py                       # CLI entry point
│   ├── evaluator.py                 # Benchmark orchestrator
│   ├── core/
│   │   ├── __init__.py
│   │   └── schema.py                # Dataclasses and schemas
│   ├── generators/
│   │   ├── __init__.py
│   │   └── document_synthesizer.py  # Programmatic document image renderer
│   └── metrics/
│       ├── __init__.py
│       ├── chart_metrics.py         # ChartQA numerical & trend matching
│       ├── extraction_metrics.py    # Levenshtein, EM, and Field F1
│       ├── layout_metrics.py        # Box IoU, mIoU, and AP@50
│       └── teds.py                  # IBM PubTabNet Tree Edit Distance
├── reports/
│   └── documentvision_report.md     # Generated research report
├── results/                         # Exported JSON and CSV metrics
├── tests/                           # 16 unit tests across metrics & CLI
├── app.py                           # Interactive Streamlit Web Studio
├── pyproject.toml                   # Package build metadata
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in `data/ground_truth.jsonl`.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run `pytest` or `documentvision benchmark` to re-compute all scores, metrics, TEDS trees, and summary tables locally.

---

## 📜 License

MIT License © 2026 Master72o / AI Evaluation Engineering Team.
