"""Phase 11b Step 3a — full cycle: real ElasticBeam3d .mpco read.

Builds a 3D ElasticBeam3d cantilever under a transverse tip load,
runs OpenSees through the El Ladruno Tcl launcher to produce a real
``.mpco`` file with both ``globalForce`` and ``localForce`` buckets,
then reads them back through ``MPCOReader.read_elements``. Verifies
physical correctness (axial / shear / moment equilibrium at end nodes)
and that frame-distinguished canonical names route to the right buckets.

The ``model_h5=`` is this cantilever's own bridge archive, not the
session stub. The stub pairs fem element 1 with ops tag 2, so an
``ids=[1]`` filter on a deck whose element is ops tag 1 was translated
to ops tag 2 and selected nothing (#1954). The deck now takes its element
tag from the archive's pairing. The filter is checked through the
fem_eid <-> ops-tag translation in both directions, with a filter of
paired ids only.

Two more filters on a real archive are pinned at the end of this file
(#2066): a filter that mixes paired and unpaired ids, and an id the file
did not record.

Skipped if the launcher is not available.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import _open_model_from_h5


_DEFAULT_TCL_LAUNCHERS = [
    r"C:\Program Files\El Ladruno OpenSees\opensees_ladruno.bat",
]


def _resolve_tcl_launcher() -> Path | None:
    env = os.environ.get("APEGMSH_OPENSEES_TCL")
    if env:
        p = Path(env)
        if p.exists():
            return p
    for cand in _DEFAULT_TCL_LAUNCHERS:
        p = Path(cand)
        if p.exists():
            return p
    return None


def _cantilever_model_h5(
    tmp_path: Path, L: float, n_elem: int = 1,
) -> "tuple[Path, dict[int, int]]":
    """Write the bridge archive of a cantilever along +x with ``n_elem``
    elements of length ``L`` (nodes 1..n_elem+1, fem elements
    1..n_elem) and return ``(model.h5, {fem element id: ops tag})``.

    The decks below are hand-written, so they declare each element with
    the ops tag the archive pairs with its fem id; otherwise the archive
    does not describe the file and an id filter is meaningless.
    """
    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite, FEMData, MeshInfo, NodeComposite,
    )
    from apeGmsh.opensees import apeSees

    node_ids = np.arange(1, n_elem + 2, dtype=np.int64)
    coords = np.zeros((n_elem + 1, 3), dtype=np.float64)
    coords[:, 0] = L * np.arange(n_elem + 1)
    elem_ids = np.arange(1, n_elem + 1, dtype=np.int64)
    line = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=n_elem,
    )
    group = ElementGroup(
        element_type=line, ids=elem_ids,
        connectivity=np.stack([node_ids[:-1], node_ids[1:]], axis=1),
    )
    pg = {(1, 100): {
        "name": "Beam", "node_ids": node_ids, "node_coords": coords,
        "element_ids": elem_ids,
    }}
    fem = FEMData(
        nodes=NodeComposite(
            node_ids=node_ids, node_coords=coords,
            physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        ),
        elements=ElementComposite(
            groups={1: group},
            physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        ),
        info=MeshInfo(
            n_nodes=n_elem + 1, n_elems=n_elem, bandwidth=1, types=[line],
        ),
        model_name="cantilever",
    )
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf,
        A=0.01, E=200e9, G=80e9, Iz=1.0e-4, Iy=1.0e-4, J=1.5e-4,
    )
    path = tmp_path / "model.h5"
    ops.h5(str(path))
    pairs = {
        int(rec.fem_eid): int(rec.tag)
        for rec in _open_model_from_h5(path).elements()
    }
    assert sorted(pairs) == elem_ids.tolist(), pairs
    return path, pairs


def _run_deck(launcher: Path, tmp_path: Path, lines: "list[str]") -> Path:
    """Run a Tcl deck that ends in ``puts "DONE"`` and return its
    ``run.mpco``. OpenSees.exe exits 0 on a Tcl error, so only the
    trailing marker proves the deck ran."""
    script = tmp_path / "model.tcl"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [str(launcher), str(script)],
        capture_output=True, text=True, timeout=60,
        cwd=str(tmp_path),
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "DONE" in combined, combined
    mpco_path = tmp_path / "run.mpco"
    assert mpco_path.exists()
    return mpco_path


def test_elastic_beam_3d_global_and_local_force(tmp_path: Path) -> None:
    """Cantilever ElasticBeam3d with combined transverse + axial tip load.

    With the beam along the global x axis and a load
    ``P = (Fx=1000, Fy=500, Fz=0, Mx=My=Mz=0)`` at node 2 (free end):

    - The cantilever's reaction at node 1 must balance the load.
      In global frame: F_n1 = (-Fx, -Fy, 0, 0, 0, -Fy*L), and
      F_n2 = (+Fx, +Fy, 0, 0, 0, 0).
    - In element-local frame (local x = global x for this geometry):
      N=Fx (axial), Vy=Fy (transverse shear), Vz=0, T=My=0,
      Mz at node 1 = -Fy*L (cantilever bending moment).
    """
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip("No OpenSees Tcl launcher found.")

    L = 5.0
    Fx, Fy = 1.0e3, 5.0e2
    model_h5, pairs = _cantilever_model_h5(tmp_path, L)
    ((fem_eid, ops_tag),) = pairs.items()
    # The filter check below proves the translation only while the two
    # id spaces differ (the bridge allocates fem element 1 as ops tag 2).
    assert ops_tag != fem_eid, (ops_tag, fem_eid)

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "node 1 0.0 0.0 0.0",
        f"node 2 {L} 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        # ElasticBeam3d: A E G Iz Iy J transfTag.
        f"element elasticBeamColumn {ops_tag} 1 2 0.01 200e9 80e9 1.0e-4 1.0e-4 1.5e-4 1",
        "recorder mpco run.mpco -N displacement -E globalForce localForce -T dt 1.0",
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        f"    load 2 {Fx} {Fy} 0.0 0.0 0.0 0.0",
        "}",
        "system ProfileSPD",
        "numberer RCM",
        "constraints Plain",
        "algorithm Linear",
        "integrator LoadControl 1.0",
        "analysis Static",
        "analyze 1",
        "wipe",
        'puts "DONE"',
    ]
    mpco_path = _run_deck(launcher, tmp_path, tcl_lines)

    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path, model_h5=model_h5) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.available_components())

        # Both global and local frames are exposed.
        for n in (
            "nodal_resisting_force_x", "nodal_resisting_force_y",
            "nodal_resisting_force_z", "nodal_resisting_moment_x",
            "nodal_resisting_moment_y", "nodal_resisting_moment_z",
            "nodal_resisting_force_local_x",
            "nodal_resisting_force_local_y",
            "nodal_resisting_force_local_z",
            "nodal_resisting_moment_local_x",
            "nodal_resisting_moment_local_y",
            "nodal_resisting_moment_local_z",
        ):
            assert n in comps, f"missing {n} in {sorted(comps)}"

        # ── Global frame: Fx at end nodes ────────────────────────
        slab_fx = s.elements.get(component="nodal_resisting_force_x")
        # 1 element × 2 nodes × 1 step.
        assert slab_fx.values.shape == (1, 1, 2)
        # Internal forces: at node 1 (fixed), the element pulls inward by
        # -Fx; at node 2 (loaded), it pushes outward by +Fx (the
        # signs depend on OpenSees's sign convention but magnitudes
        # equal Fx).
        np.testing.assert_allclose(
            np.abs(slab_fx.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )

        # ── Global frame: Fy at end nodes ────────────────────────
        slab_fy = s.elements.get(component="nodal_resisting_force_y")
        np.testing.assert_allclose(
            np.abs(slab_fy.values[0, 0]), [Fy, Fy], rtol=1e-9, atol=1e-6,
        )

        # ── Local frame: axial = Fx (since beam is along global x) ──
        slab_n = s.elements.get(component="nodal_resisting_force_local_x")
        np.testing.assert_allclose(
            np.abs(slab_n.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )

        # ── Mz at node 1 should equal -Fy*L (cantilever moment); at node 2 = 0
        slab_mz = s.elements.get(component="nodal_resisting_moment_z")
        node1_mz = slab_mz.values[0, 0, 0]
        node2_mz = slab_mz.values[0, 0, 1]
        np.testing.assert_allclose(
            abs(node1_mz), Fy * L, rtol=1e-9, atol=1e-6,
        )
        assert abs(node2_mz) < 1e-6, (
            f"Mz at free end must be ~0; got {node2_mz}"
        )

        # ── Element ID filter survives the read ────────────────
        # The file keys the element by its ops tag; the results API
        # speaks fem_eid, unfiltered and through a filter alike.
        assert slab_fx.element_ids.tolist() == [fem_eid]
        slab_filtered = s.elements.get(
            component="nodal_resisting_force_x", ids=np.array([fem_eid]),
        )
        assert slab_filtered.element_ids.tolist() == [fem_eid]
        np.testing.assert_array_equal(slab_filtered.values, slab_fx.values)
        # A filter that mixes paired and unpaired ids, or names an element
        # the file did not record, is pinned below (#2066).


# ── Element-id filters on a real archive (#2066) ────────────────────────
#
# A two-element cantilever along +x, fem elements 1 and 2, written through
# the bridge, so the archive pairs (fem, ops) = (1, 2), (2, 3). Tip load
# Fx2 at node 3 and Fx1 at node 2: fem 1 carries N1 = Fx1 + Fx2 = 1500 and
# fem 2 carries N2 = Fx2 = 1000 (statics), at rtol 1e-10. Each test asserts
# the right answer; before #2066 the reader mislabelled or dropped the rows.

_L2 = 2.0
_FX1, _FX2 = 500.0, 1000.0
_N = {1: _FX1 + _FX2, 2: _FX2}


def _two_element_mpco(
    tmp_path: Path, *, recorded_fem: "tuple[int, ...]" = (1, 2),
) -> "tuple[Path, Path, dict[int, int]]":
    """Run the two-element deck and return ``(run.mpco, model.h5, pairs)``.

    ``recorded_fem`` names the fem elements the MPCO recorder covers
    (through a ``region``); the archive always describes both.
    """
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip("No OpenSees Tcl launcher found.")
    # Setup failures raise RuntimeError, so a broken deck or archive is
    # never mistaken for a wrong read.
    try:
        model_h5, pairs = _cantilever_model_h5(tmp_path, _L2, n_elem=2)
    except AssertionError as exc:
        raise RuntimeError(f"archive setup failed: {exc}") from exc
    if pairs[1] == 1 or pairs[2] == 2:
        raise RuntimeError(f"fem ids equal ops tags, nothing to translate: {pairs}")
    region = " ".join(str(pairs[f]) for f in recorded_fem)
    lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "node 1 0.0 0.0 0.0",
        f"node 2 {_L2} 0.0 0.0",
        f"node 3 {2 * _L2} 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        f"element elasticBeamColumn {pairs[1]} 1 2 "
        "0.01 200e9 80e9 1.0e-4 1.0e-4 1.5e-4 1",
        f"element elasticBeamColumn {pairs[2]} 2 3 "
        "0.01 200e9 80e9 1.0e-4 1.0e-4 1.5e-4 1",
        f"region 1 -ele {region}",
        "recorder mpco run.mpco -E globalForce -R 1 -T dt 1.0",
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        f"    load 2 {_FX1} 0.0 0.0 0.0 0.0 0.0",
        f"    load 3 {_FX2} 0.0 0.0 0.0 0.0 0.0",
        "}",
        "system ProfileSPD",
        "numberer RCM",
        "constraints Plain",
        "algorithm Linear",
        "integrator LoadControl 1.0",
        "analysis Static",
        "analyze 1",
        "wipe",
        'puts "DONE"',
    ]
    try:
        mpco_path = _run_deck(launcher, tmp_path, lines)
    except AssertionError as exc:
        raise RuntimeError(f"deck failed: {exc}") from exc
    return mpco_path, model_h5, pairs


def _axial(slab: Any) -> "dict[int, float]":
    """``{element id: |N|}`` from a one-step ``nodal_resisting_force_x``
    slab (both end nodes carry the axial force)."""
    out = {}
    for k, eid in enumerate(np.asarray(slab.element_ids).tolist()):
        ends = np.abs(np.asarray(slab.values)[0, k])
        np.testing.assert_allclose(ends, [ends[0], ends[0]], rtol=1e-10)
        out[int(eid)] = float(ends[0])
    return out


def test_two_element_unfiltered_read_is_the_oracle(tmp_path: Path) -> None:
    """The unfiltered read relabels both rows and matches statics; the
    pins below rely on this deck and this oracle."""
    from apeGmsh.results import Results

    mpco_path, model_h5, _ = _two_element_mpco(tmp_path)
    with Results.from_mpco(mpco_path, model_h5=model_h5) as r:
        s = r.stage(r.stages[0].id)
        got = _axial(s.elements.get(component="nodal_resisting_force_x"))
    assert sorted(got) == [1, 2]
    for fem, n in _N.items():
        np.testing.assert_allclose(got[fem], n, rtol=1e-10)


@pytest.mark.parametrize("fem", [1, 2])
def test_mixed_id_filter_returns_the_paired_element(
    tmp_path: Path, fem: int,
) -> None:
    """``ids=[fem, 999]``: 999 is no element; ``fem`` must come back as
    itself with its own axial force. Before #2066 ``[2, 999]`` returned
    fem 1's row (ops tag 2) labelled 2, and ``[1, 999]`` returned nothing."""
    from apeGmsh.results import Results

    mpco_path, model_h5, _ = _two_element_mpco(tmp_path)
    with Results.from_mpco(mpco_path, model_h5=model_h5) as r:
        s = r.stage(r.stages[0].id)
        got = _axial(s.elements.get(
            component="nodal_resisting_force_x", ids=np.array([fem, 999]),
        ))
    assert sorted(got) == [fem]
    np.testing.assert_allclose(got[fem], _N[fem], rtol=1e-10)


def test_filter_on_an_unrecorded_element_is_not_silent(tmp_path: Path) -> None:
    """The recorder covers fem 2 only; ``ids=[1]`` names an element the
    archive describes but the file does not hold. The read must warn or
    raise, not return an empty slab silently."""
    import warnings

    from apeGmsh.results import Results

    mpco_path, model_h5, _ = _two_element_mpco(tmp_path, recorded_fem=(2,))
    with Results.from_mpco(mpco_path, model_h5=model_h5) as r:
        s = r.stage(r.stages[0].id)
        # The recorded element reads correctly, so the region took effect.
        # A setup failure raises RuntimeError, not an assertion.
        got = _axial(s.elements.get(component="nodal_resisting_force_x"))
        if sorted(got) != [2] or not np.isclose(got[2], _N[2], rtol=1e-10):
            raise RuntimeError(f"region setup failed: read {got}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                s.elements.get(
                    component="nodal_resisting_force_x", ids=np.array([1]),
                )
            except (ValueError, LookupError) as exc:
                assert _names_element_1(str(exc)), (
                    f"the refusal does not name element 1: {exc}"
                )
                return
    assert any(
        issubclass(w.category, UserWarning) and _names_element_1(str(w.message))
        for w in caught
    ), (
        "an unrecorded fem id read back empty with no warning naming it; "
        f"warnings: {[str(w.message) for w in caught]}"
    )


def _names_element_1(message: str) -> bool:
    """True when ``message`` speaks of an element and names id 1, so an
    unrelated warning or error cannot satisfy the pin above."""
    import re

    return ("element" in message.lower()
            and re.search(r"(?<![\d.])1(?![\d.])", message) is not None)
