"""
Data Schemas for DocumentVision Multimodal Benchmark.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional
import json


class DocumentTaskType(str, Enum):
    KEY_VALUE_EXTRACTION = "key_value_extraction"
    TABLE_RECONSTRUCTION = "table_reconstruction"
    CHART_REASONING = "chart_reasoning"
    LAYOUT_GROUNDING = "layout_grounding"


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates in [0, 1000] scale."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: Optional[str] = None
    confidence: float = 1.0

    @property
    def area(self) -> float:
        h = max(0.0, self.ymax - self.ymin)
        w = max(0.0, self.xmax - self.xmin)
        return h * w

    def iou(self, other: "BoundingBox") -> float:
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymax = min(self.ymax, other.ymax)
        inter_xmax = min(self.xmax, other.xmax)

        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_area = inter_h * inter_w

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)


@dataclass
class TableCell:
    row: int
    col: int
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass
class TableStructure:
    cells: List[TableCell] = field(default_factory=list)
    html_representation: str = ""

    def to_html(self) -> str:
        if self.html_representation:
            return self.html_representation
        # Construct basic HTML table representation
        if not self.cells:
            return "<table></table>"
        max_row = max(c.row for c in self.cells)
        rows_html = []
        for r in range(max_row + 1):
            row_cells = sorted([c for c in self.cells if c.row == r], key=lambda c: c.col)
            row_strs = []
            for c in row_cells:
                tag = "th" if c.is_header else "td"
                span_str = ""
                if c.rowspan > 1:
                    span_str += f' rowspan="{c.rowspan}"'
                if c.colspan > 1:
                    span_str += f' colspan="{c.colspan}"'
                row_strs.append(f"<{tag}{span_str}>{c.text}</{tag}>")
            rows_html.append(f"<tr>{''.join(row_strs)}</tr>")
        return f"<table>{''.join(rows_html)}</table>"


@dataclass
class ChartSeries:
    name: str
    labels: List[str]
    values: List[float]


@dataclass
class DocumentSample:
    sample_id: str
    task_type: DocumentTaskType
    domain: str
    image_path: str
    ground_truth: Dict[str, Any]
    model_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSample":
        return cls(
            sample_id=str(data["sample_id"]),
            task_type=DocumentTaskType(data["task_type"]),
            domain=str(data.get("domain", "general")),
            image_path=str(data.get("image_path", "")),
            ground_truth=dict(data.get("ground_truth", {})),
            model_predictions=dict(data.get("model_predictions", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type.value,
            "domain": self.domain,
            "image_path": self.image_path,
            "ground_truth": self.ground_truth,
            "model_predictions": self.model_predictions,
        }


@dataclass
class EvaluationResult:
    model_name: str
    task_type: DocumentTaskType
    primary_metric_name: str
    primary_metric_value: float
    detailed_metrics: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
