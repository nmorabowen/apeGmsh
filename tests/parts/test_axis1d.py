"""Unit tests for :class:`apeGmsh.parts.Axis1D`."""
from __future__ import annotations

import pytest

from apeGmsh import Axis1D


class TestSymmetricLayered:
    def test_break_layout(self):
        ax = Axis1D.symmetric_layered(
            "x",
            inner=(605.0, 10),
            layer=(10.0, 1),
            outer=(20.0, 2),
        )
        # 5 segments laid as outer | layer | inner | layer | outer,
        # symmetric about 0, with inner centred on the origin.
        breaks = ax.breaks
        assert len(breaks) == 6
        assert breaks[0] == pytest.approx(-(605.0 / 2 + 10 + 20))
        assert breaks[1] == pytest.approx(-(605.0 / 2 + 10))
        assert breaks[2] == pytest.approx(-605.0 / 2)
        assert breaks[3] == pytest.approx(+605.0 / 2)
        assert breaks[4] == pytest.approx(+(605.0 / 2 + 10))
        assert breaks[5] == pytest.approx(+(605.0 / 2 + 10 + 20))

    def test_size_and_endpoints(self):
        ax = Axis1D.symmetric_layered(
            "y",
            inner=(100.0, 4),
            layer=(20.0, 2),
            outer=(50.0, 5),
        )
        assert ax.lo == pytest.approx(-(100.0 / 2 + 20 + 50))
        assert ax.hi == pytest.approx(+(100.0 / 2 + 20 + 50))
        assert ax.size == pytest.approx(100.0 + 2 * 20.0 + 2 * 50.0)

    def test_slice_offsets_are_interior(self):
        ax = Axis1D.symmetric_layered(
            "x", inner=(10.0, 1), layer=(2.0, 1), outer=(3.0, 1),
        )
        assert ax.slice_offsets() == pytest.approx(ax.breaks[1:-1])
        assert len(ax.slice_offsets()) == 4

    def test_witnesses_midpoints(self):
        ax = Axis1D.symmetric_layered(
            "x", inner=(10.0, 1), layer=(2.0, 1), outer=(3.0, 1),
        )
        ws = ax.witnesses()
        assert [r for r, _ in ws] == ["outer", "layer", "inner", "layer", "outer"]
        # Inner segment is centred on the origin.
        assert ws[2][1] == pytest.approx(0.0)

    def test_region_of_segment_interiors(self):
        ax = Axis1D.symmetric_layered(
            "x", inner=(10.0, 4), layer=(2.0, 1), outer=(3.0, 2),
        )
        # inner size 10 ⇒ inner range [-5, +5]; layers extend ±2 ⇒
        # [-7, -5] and [+5, +7]; outers extend ±3 ⇒ [-10, -7] and
        # [+7, +10].
        assert ax.region_of(-8.5) == "outer"
        assert ax.region_of(-6.0) == "layer"
        assert ax.region_of(0.0) == "inner"
        assert ax.region_of(+6.0) == "layer"
        assert ax.region_of(+8.5) == "outer"

    def test_region_of_breakpoints(self):
        ax = Axis1D.symmetric_layered(
            "x", inner=(10.0, 4), layer=(2.0, 1), outer=(3.0, 2),
        )
        # Interior break belongs to the segment to its right.
        assert ax.region_of(-7.0) == "layer"
        assert ax.region_of(-5.0) == "inner"
        assert ax.region_of(+5.0) == "layer"
        assert ax.region_of(+7.0) == "outer"
        # Endpoints belong to the outer segments.
        assert ax.region_of(-10.0) == "outer"
        assert ax.region_of(+10.0) == "outer"

    def test_region_of_outside_range_raises(self):
        ax = Axis1D.symmetric_layered(
            "x", inner=(10.0, 4), layer=(2.0, 1), outer=(3.0, 2),
        )
        with pytest.raises(ValueError):
            ax.region_of(-100.0)
        with pytest.raises(ValueError):
            ax.region_of(+100.0)

    def test_count_for(self):
        ax = Axis1D.symmetric_layered(
            "x", inner=(10.0, 4), layer=(2.0, 1), outer=(3.0, 2),
        )
        # See test_region_of_segment_interiors for the segment ranges.
        assert ax.count_for(0.0) == 4
        assert ax.count_for(-6.0) == 1
        assert ax.count_for(+6.0) == 1
        assert ax.count_for(-8.5) == 2
        assert ax.count_for(+8.5) == 2


class TestDownwardLayered:
    def test_break_layout(self):
        ax = Axis1D.downward_layered(
            "z",
            top=(50.0, 5),
            mid=(50.0, 5),
            bottom=(200.0, 20),
        )
        # bottom | mid | top, with hi == 0
        breaks = ax.breaks
        assert len(breaks) == 4
        assert breaks[0] == pytest.approx(-(50.0 + 50.0 + 200.0))
        assert breaks[1] == pytest.approx(-(50.0 + 50.0))
        assert breaks[2] == pytest.approx(-50.0)
        assert breaks[3] == pytest.approx(0.0)

    def test_regions_descending(self):
        ax = Axis1D.downward_layered(
            "z", top=(50.0, 5), mid=(50.0, 5), bottom=(200.0, 20),
        )
        # Free surface (z=0) belongs to top.
        assert ax.region_of(0.0) == "top"
        assert ax.region_of(-25.0) == "top"
        assert ax.region_of(-75.0) == "mid"
        assert ax.region_of(-200.0) == "bottom"

    def test_count_for_z(self):
        ax = Axis1D.downward_layered(
            "z", top=(50.0, 5), mid=(50.0, 5), bottom=(200.0, 20),
        )
        assert ax.count_for(-25.0) == 5     # top
        assert ax.count_for(-75.0) == 5     # mid
        assert ax.count_for(-200.0) == 20   # bottom


class TestValidation:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError):
            Axis1D(name="x", segments=())

    def test_non_contiguous_segments_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            Axis1D(
                name="x",
                segments=(
                    ("a", 0.0, 1.0, 1),
                    ("b", 2.0, 3.0, 1),  # gap from 1.0 to 2.0
                ),
            )

    def test_zero_count_raises(self):
        with pytest.raises(ValueError, match="count"):
            Axis1D(name="x", segments=(("a", 0.0, 1.0, 0),))

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="hi"):
            Axis1D(name="x", segments=(("a", 0.0, 0.0, 1),))

    def test_symmetric_layered_zero_size_raises(self):
        with pytest.raises(ValueError, match="size"):
            Axis1D.symmetric_layered(
                "x", inner=(0.0, 1), layer=(1.0, 1), outer=(1.0, 1),
            )
