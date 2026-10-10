"""
Unit Tests for Tree Edit Distance-based Similarity (TEDS).
"""

import pytest
from documentvision.metrics.teds import TEDS, levenshtein_distance


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("same", "same") == 0


def test_teds_identical_tables():
    html = "<table><tr><th>Header</th></tr><tr><td>Value</td></tr></table>"
    teds = TEDS()
    assert teds.evaluate(html, html) == 1.0


def test_teds_empty_tables():
    teds = TEDS()
    assert teds.evaluate("", "") == 0.0
    assert teds.evaluate("<table><tr><td>A</td></tr></table>", "") == 0.0


def test_teds_structure_only():
    html1 = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
    html2 = "<table><tr><th>Title</th><th>Years</th></tr><tr><td>Bob</td><td>40</td></tr></table>"

    teds_full = TEDS(structure_only=False)
    teds_struct = TEDS(structure_only=True)

    score_full = teds_full.evaluate(html1, html2)
    score_struct = teds_struct.evaluate(html1, html2)

    # Structure is identical (2x2 table with th and td), so structure TEDS is 1.0
    assert score_struct == 1.0
    # Content is different, so full TEDS is < 1.0
    assert score_full < 1.0


def test_teds_structural_difference():
    html_2x2 = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    html_3x3 = "<table><tr><td>A</td><td>B</td><td>X</td></tr><tr><td>C</td><td>D</td><td>Y</td></tr><tr><td>Z1</td><td>Z2</td><td>Z3</td></tr></table>"

    teds = TEDS()
    score = teds.evaluate(html_2x2, html_3x3)
    assert 0.0 < score < 1.0
