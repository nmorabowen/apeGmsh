"""AABB quotation (cota) text for ModelViewer labels."""
from apeGmsh.viewers.ui._filter_view_tabs import quotation_text


def test_quotation_text_solid() -> None:
    text = quotation_text("footing", (0.0, 0.0, 0.0, 1.0, 1.0, 0.4))
    assert text == "footing\n1 × 1 × 0.4 m"


def test_quotation_text_drops_flat_axis() -> None:
    text = quotation_text("ColumnTop", (-0.2, -0.2, 3.425, 0.2, 0.2, 3.425))
    assert text == "ColumnTop\n0.4 × 0.4 m"


def test_quotation_text_none_is_name_only() -> None:
    assert quotation_text("col", None) == "col"
