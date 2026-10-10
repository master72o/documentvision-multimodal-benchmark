"""
DocumentVision: Multimodal Document Intelligence Benchmark.
Evaluating OCR Pipelines vs Vision-Language Models on Dense Tables, Charts, and Layout Grounding.
"""

from documentvision.core.schema import (
    DocumentTaskType,
    BoundingBox,
    TableCell,
    TableStructure,
    ChartSeries,
    DocumentSample,
    EvaluationResult,
)
from documentvision.metrics.teds import TEDS
from documentvision.metrics.extraction_metrics import ExtractionMetricsCalculator
from documentvision.metrics.chart_metrics import ChartMetricsCalculator
from documentvision.metrics.layout_metrics import LayoutMetricsCalculator
from documentvision.evaluator import DocumentVisionEvaluator

__version__ = "1.0.0"

__all__ = [
    "DocumentTaskType",
    "BoundingBox",
    "TableCell",
    "TableStructure",
    "ChartSeries",
    "DocumentSample",
    "EvaluationResult",
    "TEDS",
    "ExtractionMetricsCalculator",
    "ChartMetricsCalculator",
    "LayoutMetricsCalculator",
    "DocumentVisionEvaluator",
    "__version__",
]
