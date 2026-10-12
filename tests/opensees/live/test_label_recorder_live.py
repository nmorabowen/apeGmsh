"""``recorder.Element(label=)`` records the labelled element under both
tag modes (ADR 0115 amendment R2-f, #1735), on stock openseespy.

The Pratt truss of ``test_readback_live.py`` with its element families
declared in reverse, so under ``element_tags="sequential"`` an element's
OpenSees tag is not its FEM id. ``label="diag2"`` resolves to the FEM id
at the call; the build translates it. The oracle is the method of
sections: the panel-2 diagonal carries ``F_DIAG``. Recording the FEM id
verbatim as a tag would record another bar (a top chord here), whose
force differs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.opensees.live.test_readback_live import (
    F_DIAG,
    build_pratt_fem,
    pratt_bridge,
)

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


@pytest.fixture(scope="module")
def pratt() -> Any:
    return build_pratt_fem()


@pytest.mark.parametrize("element_tags", ["sequential", "fem"])
def test_label_recorder_records_the_labelled_bar(
    pratt: Any, tmp_path: Path, element_tags: str,
) -> None:
    from apeGmsh.opensees.emitter.live import get_ops

    out = tmp_path / f"diag2_{element_tags}.out"
    ops = pratt_bridge(pratt, element_tags=element_tags, reverse=True)
    rec = ops.recorder.Element(file=str(out), response=("axialForce",),
                               label="diag2")
    eid = ops.elements.one(label="diag2").id
    assert rec.elements == (eid,)
    assert ops.analyze(steps=1) == 0
    read = float(ops.elements.one(label="diag2").read("axial_force"))
    get_ops().remove("recorders")  # close the file so it is flushed
    recorded = np.loadtxt(out, ndmin=1)
    assert recorded.shape == (1,)
    # A text recorder writes 6 significant digits.
    assert recorded[0] == pytest.approx(F_DIAG, rel=1e-5)
    assert recorded[0] == pytest.approx(read, rel=1e-5)
