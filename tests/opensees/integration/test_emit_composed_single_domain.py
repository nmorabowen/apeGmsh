"""Regression — a composed multi-module FEM solves in one process.

``g.compose(...)`` auto-assigns one OpenSeesMP rank per module (ADR 0038
§"Rank model": host on rank 0, each composed module on rank 1, 2, …), so a
composed model reports as *partitioned* even though it is one logical
structure.  Driving it through ``apeSees`` with a single-process emitter
(:class:`LiveOpsEmitter`) must therefore **flatten** the partitions into one
domain — otherwise the live emitter no-ops every non-rank-0 block and the
composed modules' nodes are silently dropped (the bug this test locks).

Gated on ``openseespy`` (the ``live`` path); ``gmsh`` is a hard dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.mesh.FEMData import FEMData  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees._internal.build import is_partitioned  # noqa: E402
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402


def _build_bar_module(path: Path) -> None:
    """Save a one-element cantilever-bar module to ``path`` (mm, N)."""
    with apeGmsh(model_name="bar", save_to=str(path), overwrite=True) as g:
        a = g.model.geometry.add_point(0.0, 0.0, 0.0)
        b = g.model.geometry.add_point(1000.0, 0.0, 0.0)
        line = g.model.geometry.add_line(a, b)
        g.model.sync()
        g.physical.add(1, [line], name="Bar")
        g.physical.add(0, [a], name="Fixed")
        g.physical.add(0, [b], name="Tip")
        g.mesh.sizing.set_global_size(250.0)   # ~4 elements
        g.mesh.generation.generate(1)


def test_composed_two_modules_emits_all_nodes_and_analyzes(tmp_path: Path) -> None:
    module = tmp_path / "bar.h5"
    _build_bar_module(module)

    # Compose a second copy: host (rank 0) + module "B" (rank 1).
    g = apeGmsh.from_h5(str(module))
    g.compose(str(module), label="B", translate=(0.0, 2000.0, 0.0))
    composed = tmp_path / "two_bars.h5"
    g.save(str(composed))

    fem = FEMData.from_h5(str(composed))
    # Precondition: compose auto-partitions one rank per module.
    assert is_partitioned(fem), "expected the composed FEM to be partitioned"
    assert len(fem.partitions) == 2

    # Drive the whole composed model through the live (single-process) bridge.
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    for pg in ("Bar", "B.Bar"):
        ops.element.elasticBeamColumn(
            pg=pg, transf=transf, A=4.0e4, E=2.0e5, Iz=1.333e8,
        )
    ops.fix(pg="Fixed", dofs=(1, 1, 1))
    ops.fix(pg="B.Fixed", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(pg="Tip", forces=(0.0, -1.0e3, 0.0))
        pat.load(pg="B.Tip", forces=(0.0, -1.0e3, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    bm = ops.build()
    bm.emit(emitter)
    ret = emitter.analyze(steps=1)

    # EVERY composed module's nodes must reach the single live domain ...
    domain_tags = openseespy.getNodeTags()
    assert len(domain_tags) == fem.info.n_nodes, (
        f"composed model dropped modules: {len(domain_tags)} of "
        f"{fem.info.n_nodes} nodes in the OpenSees domain"
    )
    # ... and both module B's node ids (tag-shifted to ~1_000_000+) are present.
    all_ids = {int(x) for x in fem.nodes.ids}
    assert all_ids == {int(t) for t in domain_tags}
    assert max(all_ids) > 1_000_000, "module B should be tag-shifted by compose"

    # ... and the model actually solves, with module B carrying its load.
    assert ret == 0, f"analyze returned non-zero ({ret}) on the composed model"
    b_tip = int(fem.nodes.select(pg="B.Tip").ids[0])
    assert b_tip > 1_000_000, "module B's tip should be tag-shifted by compose"
    tip_uy = openseespy.nodeDisp(b_tip, 2)
    assert abs(tip_uy) > 0.0, "module B carried no load — it was not emitted"
