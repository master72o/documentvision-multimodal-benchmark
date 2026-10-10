"""
Tree Edit Distance-based Similarity (TEDS) for Table Structure and Content Evaluation.
Implements the IBM PubTabNet (Zhong et al., 2020) evaluation standard for document tables.
"""

from html.parser import HTMLParser
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein string edit distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    return int(dp[m][n])


class TableNode:
    def __init__(self, tag: str, text: str = "", attrs: Optional[Dict[str, str]] = None):
        self.tag = tag.lower()
        self.text = text.strip()
        self.attrs = attrs or {}
        self.children: List["TableNode"] = []

    @property
    def colspan(self) -> int:
        return int(self.attrs.get("colspan", 1))

    @property
    def rowspan(self) -> int:
        return int(self.attrs.get("rowspan", 1))

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)


class TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root: Optional[TableNode] = None
        self.stack: List[TableNode] = []
        self.current_text: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in ["table", "thead", "tbody", "tr", "th", "td"]:
            attr_dict = {k.lower(): v for k, v in attrs}
            node = TableNode(tag_lower, attrs=attr_dict)
            if self.root is None and tag_lower == "table":
                self.root = node
            elif self.stack:
                self.stack[-1].children.append(node)
            self.stack.append(node)
            self.current_text = []

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if self.stack and self.stack[-1].tag == tag_lower:
            popped = self.stack.pop()
            popped.text = "".join(self.current_text).strip()
            self.current_text = []

    def handle_data(self, data):
        self.current_text.append(data)


class TEDS:
    """Tree Edit Distance-based Similarity Calculator for Tables."""

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only

    def parse_html(self, html_str: str) -> TableNode:
        parser = TableHTMLParser()
        parser.feed(html_str)
        return parser.root if parser.root is not None else TableNode("table")

    def _cell_edit_distance(self, c1: TableNode, c2: TableNode) -> float:
        """Edit distance between two table cells."""
        cost = 0.0
        # Tag mismatch (e.g. th vs td)
        if c1.tag != c2.tag:
            cost += 0.5

        # Span mismatch
        if c1.colspan != c2.colspan or c1.rowspan != c2.rowspan:
            cost += 0.5

        if not self.structure_only:
            max_len = max(len(c1.text), len(c2.text))
            if max_len > 0:
                lev = levenshtein_distance(c1.text, c2.text)
                cost += float(lev / max_len)

        return cost

    def _row_edit_distance(self, r1: TableNode, r2: TableNode) -> float:
        """Sequence alignment edit distance between two table rows."""
        cells1 = [c for c in r1.children if c.tag in ["td", "th"]]
        cells2 = [c for c in r2.children if c.tag in ["td", "th"]]

        n1, n2 = len(cells1), len(cells2)
        dp = np.zeros((n1 + 1, n2 + 1), dtype=float)

        for i in range(n1 + 1):
            dp[i][0] = i * 1.0
        for j in range(n2 + 1):
            dp[0][j] = j * 1.0

        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                cell_cost = self._cell_edit_distance(cells1[i - 1], cells2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,         # Delete cell
                    dp[i][j - 1] + 1.0,         # Insert cell
                    dp[i - 1][j - 1] + cell_cost # Modify cell
                )

        return float(dp[n1][n2])

    def evaluate(self, pred_html: str, gt_html: str) -> float:
        """Computes TEDS score in [0.0, 1.0] between predicted and ground-truth HTML table."""
        if not pred_html.strip() and not gt_html.strip():
            return 0.0
        if not pred_html.strip() or not gt_html.strip():
            return 0.0
        if pred_html.strip() == gt_html.strip():
            return 1.0

        tree_pred = self.parse_html(pred_html)
        tree_gt = self.parse_html(gt_html)

        # Extract all rows from both trees (flattening any thead/tbody)
        def get_rows(node: TableNode) -> List[TableNode]:
            rows = []
            for child in node.children:
                if child.tag == "tr":
                    rows.append(child)
                elif child.tag in ["thead", "tbody", "table"]:
                    rows.extend(get_rows(child))
            return rows

        rows_pred = get_rows(tree_pred)
        rows_gt = get_rows(tree_gt)

        m1, m2 = len(rows_pred), len(rows_gt)
        if m1 == 0 and m2 == 0:
            return 1.0
        if m1 == 0 or m2 == 0:
            return 0.0

        # Dynamic programming row sequence alignment
        dp = np.zeros((m1 + 1, m2 + 1), dtype=float)

        for i in range(m1 + 1):
            dp[i][0] = sum(1.0 + len(rows_pred[k].children) for k in range(i))
        for j in range(m2 + 1):
            dp[0][j] = sum(1.0 + len(rows_gt[k].children) for k in range(j))

        for i in range(1, m1 + 1):
            for j in range(1, m2 + 1):
                del_cost = 1.0 + len(rows_pred[i - 1].children)
                ins_cost = 1.0 + len(rows_gt[j - 1].children)
                sub_cost = self._row_edit_distance(rows_pred[i - 1], rows_gt[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + del_cost,
                    dp[i][j - 1] + ins_cost,
                    dp[i - 1][j - 1] + sub_cost,
                )

        tree_dist = float(dp[m1][m2])
        size_pred = tree_pred.node_count()
        size_gt = tree_gt.node_count()
        max_size = max(size_pred, size_gt, 1)

        teds_score = max(0.0, 1.0 - (tree_dist / max_size))
        return round(float(teds_score), 4)
