"""``NodeSlab.item()`` (ADR 0115 D3, R2-c #1704): the one value of a
one-node, one-step slab, and a refusal on anything larger."""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results._slabs import NodeSlab


def slab(values: list[list[float]]) -> NodeSlab:
    v = np.asarray(values, dtype=np.float64)
    return NodeSlab(
        component="displacement_y", values=v,
        node_ids=np.arange(1, v.shape[1] + 1), time=np.arange(v.shape[0]),
    )


def test_item_returns_the_one_value_as_a_float() -> None:
    got = slab([[-0.028125]]).item()
    assert type(got) is float and got == -0.028125


@pytest.mark.parametrize("values", [
    [[1.0, 2.0]],          # one step, two nodes
    [[1.0], [2.0]],        # two steps, one node
    [[1.0, 2.0], [3.0, 4.0]],
])
def test_item_refuses_more_than_one_value(values: list[list[float]]) -> None:
    n = np.asarray(values).size
    with pytest.raises(ValueError, match=rf"holds {n} values.*time=-1"):
        slab(values).item()


def test_item_refuses_an_empty_slab() -> None:
    empty = NodeSlab(component="displacement_y", values=np.empty((0, 1)),
                     node_ids=np.array([1]), time=np.empty(0))
    with pytest.raises(ValueError, match=r"holds 0 values"):
        empty.item()
