"""#1326: ``FieldHelper._resolve_at_dim`` accepts numpy integer tags.

Tags straight from ``g.physical.entities`` are ``np.int32``; the field
builders used to reject them (``isinstance(r, int)``).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh


@pytest.fixture
def helper():
    with apeGmsh(model_name="np_tags") as g:
        yield g.mesh.field


def test_numpy_scalar_and_list(helper):
    assert helper._resolve_at_dim([np.int32(4), np.int64(5)], 1, "w") == [4, 5]
    assert helper._resolve_at_dim(np.int32(4), 1, "w") == [4]


def test_numpy_array(helper):
    out = helper._resolve_at_dim(np.array([1, 2], dtype=np.int32), 1, "w")
    assert out == [1, 2]
    assert all(type(t) is int for t in out)


def test_numpy_dimtag_tuple(helper):
    assert helper._resolve_at_dim((np.int32(1), np.int32(4)), 1, "w") == [4]
    with pytest.raises(ValueError):
        helper._resolve_at_dim((np.int32(2), np.int32(4)), 1, "w")


def test_bool_still_rejected(helper):
    with pytest.raises(TypeError):
        helper._resolve_at_dim([True], 1, "w")


def test_physical_entities_tags_feed_distance():
    with apeGmsh(model_name="np_dist") as g:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
        e = 1e-6
        g.model.select(None, dim=1).in_box(
            (-e, -e, -e), (e, 1 + e, e)).to_physical("Left")
        tags = g.physical.entities("Left", dim=1)
        g.mesh.field.distance(curves=tags)
