"""Fork lane: fibres and layered shells read back from ``.ladruno`` and MPCO.

Gap G4 of the DomainCapture parity table (#1767, #1945). ``.out`` cannot carry
fibres or layers, and on MPCO only the emission was tested
(``tests/test_results_live_mpco.py``); the value-level reader tests skip
without the STKO example fixtures. Here both kinds are built by the bridge
(``apeSees(fem)``, so element ids go through the fem -> ops translation),
recorded by the fork's own recorders, read back by the public factories and
checked against the live domain at ``rtol=1e-10``:

* **Fibres.** A two-element ``forceBeamColumn`` cantilever with a five-fibre
  section of two materials. Oracle: ``eleResponse(e, "section", ip, "fiber",
  y, z, "stressStrain")`` per fibre and step, and the ``FiberPoint``
  definitions for ``(y, z, area, material)``.
* **Layers.** Two ``ASDShellQ4`` with a four-layer ``LayeredShell`` (two
  materials, four thicknesses), in a plane tilted 30 deg about ``x`` so the
  local frame is not the identity, and the same strip for every other
  layered-shell class the bridge emits (``LAYER_CLASSES``). Oracle:
  ``eleResponse(e, "material", gp, "fiber", k, "stress"|"strain")`` per
  layer, the ``ShellLayer`` thicknesses, and the quaternion C7 writes
  (``results/_shell_geometry.py``). That oracle is itself checked twice:
  against the closed form ``(cos 15, -sin 15, 0, 0)`` (the rotation global ->
  local, rows = local axes), and against the quaternion the fork's MPCO
  writer stores for a beam with the same frame.

Every kind x route cell is pinned, so a reader change flips something:

=================== ======================= ===============================
Kind                ``.ladruno``            MPCO
=================== ======================= ===============================
Fibre values        pass                    pass
Fibre geometry      pass                    pass
Layer values,       pass (#1995)            pass
thickness
Layer quaternion    pass: no stored shell   pass: no stored shell frame,
                    frame, computed from    computed from the file's
                    the file's coordinates  coordinates (#1996)
                    (#1995)
=================== ======================= ===============================

Every layered bridge shell has a ``LAYER_CATALOG`` row. ShellMITC3, the one
that had none, left the bridge (#2092): no build has the element (the pinned
fork's element table answers ``element type ShellMITC3 is unknown``). Both
readers still warn by class on such a bucket in another writer's file
(``tests/test_results_layer_catalog_shellmitc3.py``, #2090).

Fields a format does not carry (reported on #1767, never synthesised here):
``LayerSlab`` has no layer-material field on either format, although both
files store ``FIBER_MATERIALS``; neither file stores local axes for a
shell, so both readers compute the frame from node coordinates.

Pip ``openseespy`` 3.8.0 ships no MPCO recorder and no ``.ladruno`` recorder,
and ``.out`` carries neither kind, so on a pip install fibres and layers have
no route at all. Both formats therefore run here, on the fork's in-process
module (``get_ops`` via ``LiveOpsEmitter``), gated by the ``ladruno_fork``
marker that the nightly ``live-fork`` lane runs against the pinned build.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees import element as _bridge_elements
from apeGmsh.opensees._response_catalog import LAYER_CATALOG, RESPONSE_CATALOG
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.section.fiber import FiberPoint
from apeGmsh.opensees.section.plate import ShellLayer
from apeGmsh.results import Results
from apeGmsh.results._shell_geometry import shell_quaternion

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
KINDS = ("Ladruno", "MPCO")
N_STEPS = 2
#: Gauss-Lobatto, three points: the stations at the ends and the middle.
LOBATTO_3 = (-1.0, 0.0, 1.0)
#: PlateFiber answers five components per layer (11, 22, 12, 23, 13).
N_PLATE = 5
#: The shell plane is the xy plane turned this far about x.
TILT_DEG = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _solve(ops: apeSees, kind: str, tmp_path: Path, stem: str) -> tuple[
    LiveOpsEmitter, str, str, dict[int, int],
]:
    """Record both fibre tokens, persist the model, build it live.

    The nodal channel is on: the MPCO reader takes a stage's time vector
    from the node results (#1970).
    """
    path = str(tmp_path / f"{stem}.{'ladruno' if kind == 'Ladruno' else 'mpco'}")
    getattr(ops.recorder, kind)(
        file=path,
        nodal_responses=("displacement",),
        elem_responses=("section.fiber.stress", "section.fiber.strain"),
    )
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / N_STEPS)
    ops.analysis.Static()
    model_h5 = str(tmp_path / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    f2o = _fem_to_ops(model_h5)
    assert all(t != e for e, t in f2o.items())    # the translation is exercised
    return emitter, path, model_h5, f2o


def _open(kind: str, path: str, fem: Any, model_h5: str) -> Results:
    factory: Callable[..., Results] = (
        Results.from_ladruno if kind == "Ladruno" else Results.from_mpco
    )
    return factory(path, fem=fem, model_h5=model_h5)


def _assert_close(got: Any, want: Any, what: str) -> None:
    want = np.asarray(want, dtype=float)
    scale = float(np.abs(want).max())
    assert scale > 0.0, what
    np.testing.assert_allclose(
        np.asarray(got, dtype=float), want, rtol=RTOL, atol=RTOL * scale,
        err_msg=what,
    )


# ---------------------------------------------------------------------------
# Fibres: a forceBeamColumn cantilever
# ---------------------------------------------------------------------------


def _beam_fem(g: Any) -> Any:
    G = g.model.geometry
    a = G.add_point(0.0, 0.0, 0.0)
    b = G.add_point(2.0, 0.0, 0.0)
    line = G.add_line(a, b)
    g.model.sync()
    g.physical.add(1, [line], name="Beam")
    g.physical.add(0, [a], name="Base")
    g.physical.add(0, [b], name="Tip")
    g.mesh.structured.set_transfinite_curve(line, 3)     # two elements
    g.mesh.generation.generate(1)
    return g.mesh.queries.get_fem_data(dim=1)


def _fiber_run(g: Any, tmp_path: Path, kind: str) -> tuple[
    Results, dict[tuple[float, float], tuple[float, int]],
    dict[tuple[int, int, float, float], np.ndarray],
]:
    """Solve the cantilever; return results, the section definition, live.

    The definition is ``{(y, z): (area, ops material tag)}``; the live
    answer is ``{(fem_eid, gp, y, z): array (N_STEPS, 2)}`` of
    ``(stress, strain)`` per step.
    """
    fem = _beam_fem(g)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    conc = ops.uniaxialMaterial.ElasticMaterial(E=30e9)
    fibers = (
        FiberPoint(material=steel, y=0.10, z=0.05, area=1e-3),
        FiberPoint(material=steel, y=-0.10, z=0.05, area=2e-3),
        FiberPoint(material=conc, y=0.10, z=-0.05, area=3e-3),
        FiberPoint(material=conc, y=-0.10, z=-0.05, area=4e-3),
        FiberPoint(material=conc, y=0.0, z=0.0, area=5e-3),
    )
    sec = ops.section.Fiber(fibers=fibers, GJ=1e8)
    ops.element.forceBeamColumn(
        pg="Beam", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
        integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tip", forces=(1e4, 2e3, 1e3, 0.0, 0.0, 0.0))
    emitter, path, model_h5, f2o = _solve(ops, kind, tmp_path, "fiber")
    live: dict[tuple[int, int, float, float], list[list[float]]] = {}
    for _ in range(N_STEPS):
        assert emitter.analyze(steps=1) == 0
        for eid, tag in f2o.items():
            for ip in (1, 2, 3):
                for f in fibers:
                    ss = emitter.ops.eleResponse(
                        tag, "section", ip, "fiber", f.y, f.z, "stressStrain",
                    )
                    live.setdefault((eid, ip - 1, f.y, f.z), []).append(
                        [float(v) for v in ss])
    emitter.ops.remove("recorders")
    definition = {(f.y, f.z): (f.area, ops.tag_for(f.material)) for f in fibers}
    assert len(set(m for _a, m in definition.values())) == 2
    return (_open(kind, path, fem, model_h5), definition,
            {k: np.asarray(v) for k, v in live.items()})


def _fiber_rows(slab: Any) -> list[tuple[int, int, float, float]]:
    return [
        (int(e), int(gp), float(y), float(z))
        for e, gp, y, z in zip(slab.element_index, slab.gp_index, slab.y, slab.z)
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_fiber_stress_and_strain_equal_live(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    r, _definition, live = _fiber_run(g, tmp_path, kind)
    for col, comp in enumerate(("fiber_stress", "fiber_strain")):
        slab = r.elements.fibers.get(component=comp)
        rows = _fiber_rows(slab)
        assert sorted(rows) == sorted(live), comp
        assert np.asarray(slab.values).shape == (N_STEPS, len(live)), comp
        want = np.stack([live[row][:, col] for row in rows], axis=1)
        _assert_close(slab.values, want, f"{comp} on {kind}")


@pytest.mark.parametrize("kind", KINDS)
def test_fiber_geometry_equals_the_section_definition(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    """``(y, z, area, material)`` per fibre, and the Lobatto stations."""
    r, definition, live = _fiber_run(g, tmp_path, kind)
    slab = r.elements.fibers.get(component="fiber_stress")
    rows = _fiber_rows(slab)
    assert sorted(rows) == sorted(live)
    np.testing.assert_allclose(
        slab.area, [definition[(y, z)][0] for _e, _gp, y, z in rows],
        rtol=RTOL)
    assert [int(m) for m in slab.material_tag] == [
        definition[(y, z)][1] for _e, _gp, y, z in rows]
    assert slab.station_natural_coord is not None
    np.testing.assert_allclose(
        slab.station_natural_coord, [LOBATTO_3[gp] for _e, gp, _y, _z in rows],
        atol=1e-12)


# ---------------------------------------------------------------------------
# Layers: ASDShellQ4 + LayeredShell, in a tilted plane
# ---------------------------------------------------------------------------


def _shell_fem(g: Any, quads: bool = True) -> Any:
    """A 2 x 1 strip, meshed as two quads (or four triangles), in the xy
    plane turned about x."""
    c = math.cos(math.radians(TILT_DEG))
    s = math.sin(math.radians(TILT_DEG))
    G = g.model.geometry
    pts = [G.add_point(x, y * c, y * s) for x, y in ((0, 0), (2, 0), (2, 1), (0, 1))]
    lines = [G.add_line(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    surf = G.add_plane_surface([G.add_curve_loop(lines)])
    g.model.sync()
    g.physical.add(2, [surf], name="Slab")
    g.physical.add(1, [lines[3]], name="Fixed")
    g.physical.add(1, [lines[1]], name="Free")
    st = g.mesh.structured
    for ln, n in zip(lines, (3, 2, 3, 2)):
        st.set_transfinite_curve(ln, n)
    st.set_transfinite_surface(surf)
    if quads:
        st.set_recombine(surf)
    g.mesh.generation.generate(2)
    return g.mesh.queries.get_fem_data(dim=2)


THICKNESS = (0.08, 0.004, 0.05, 0.006)

#: The layered-shell classes the bridge emits (``apeSees.element``) that the
#: response catalog gives a layer layout (``LAYER_CATALOG``), with the
#: surface Gauss point count of their rule.
LAYER_CLASSES = {
    cls: RESPONSE_CATALOG[(
        cls, LAYER_CATALOG[(cls, "fiber_stress")].surface_int_rule, "stress",
    )].n_gauss_points
    for cls in sorted({c for c, _token in LAYER_CATALOG})
    if hasattr(_bridge_elements, cls)
}
_TRIANGLES = {"ASDShellT3"}


def test_layer_classes_are_the_bridge_shells() -> None:
    """The roster this file covers, so a new bridge shell shows up here."""
    assert LAYER_CLASSES == {
        "ASDShellQ4": 4, "ASDShellT3": 3, "ShellDKGQ": 4, "ShellMITC4": 4,
    }
    # ShellMITC3, the layered shell no build had, left the bridge (#2092).
    assert not hasattr(_bridge_elements, "ShellMITC3")


def _layer_run(
    g: Any, tmp_path: Path, kind: str, cls: str = "ASDShellQ4",
) -> tuple[
    Results, dict[tuple[int, int, int], np.ndarray], dict[int, np.ndarray],
]:
    """Solve the shell strip; return results, live layer state, frames.

    Live is ``{(fem_eid, gp, layer): array (N_STEPS, 2, N_PLATE)}`` of
    ``(stress, strain)``; frames is ``{fem_eid: quaternion}`` from the
    live node coordinates through C7's ``shell_quaternion``.
    """
    fem = _shell_fem(g, quads=cls not in _TRIANGLES)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    steel = ops.nDMaterial.ElasticIsotropic(E=200e9, nu=0.3)
    sec = ops.section.LayeredShell(layers=tuple(
        ShellLayer(material=m, thickness=t)
        for m, t in zip((conc, steel, conc, steel), THICKNESS)
    ))
    getattr(ops.element, cls)(pg="Slab", section=sec)
    ops.fix(pg="Fixed", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Free", forces=(1e4, 2e3, 5e3, 0.0, 0.0, 0.0))
    emitter, path, model_h5, f2o = _solve(ops, kind, tmp_path, "layer")
    live: dict[tuple[int, int, int], list[list[list[float]]]] = {}
    for _ in range(N_STEPS):
        assert emitter.analyze(steps=1) == 0
        for eid, tag in f2o.items():
            for gp in range(1, LAYER_CLASSES[cls] + 1):
                for k in range(1, len(THICKNESS) + 1):
                    pair = [
                        [float(v) for v in emitter.ops.eleResponse(
                            tag, "material", gp, "fiber", k, resp)]
                        for resp in ("stress", "strain")
                    ]
                    assert all(len(v) == N_PLATE for v in pair)
                    live.setdefault((eid, gp - 1, k - 1), []).append(pair)
    frames = {
        eid: shell_quaternion(
            np.array([emitter.ops.nodeCoord(n) for n in emitter.ops.eleNodes(tag)]),
            cls,
        )
        for eid, tag in f2o.items()
    }
    emitter.ops.remove("recorders")
    return (_open(kind, path, fem, model_h5),
            {k: np.asarray(v) for k, v in live.items()}, frames)


def _assert_layers_equal_live(
    r: Results, live: dict[tuple[int, int, int], np.ndarray], kind: str,
) -> None:
    """Every layer component, row set, value, thickness and sub-GP. The
    fiber level lists nothing: a shell's layer state reads as layers."""
    assert r.elements.fibers.available_components() == [], kind
    assert r.elements.layers.available_components() == sorted(
        f"{parent}_{c}" for parent in ("fiber_strain", "fiber_stress")
        for c in range(N_PLATE)
    ), kind
    for col, parent in enumerate(("fiber_stress", "fiber_strain")):
        for c in range(N_PLATE):
            comp = f"{parent}_{c}"
            slab = r.elements.layers.get(component=comp)
            rows = [
                (int(e), int(gp), int(k))
                for e, gp, k in zip(slab.element_index, slab.gp_index,
                                    slab.layer_index)
            ]
            assert sorted(rows) == sorted(live), comp
            want = np.stack([live[row][:, col, c] for row in rows], axis=1)
            if np.abs(want).max() == 0.0:
                # A Kirchhoff shell (ShellDKGQ) answers zero transverse
                # shear in every layer: the file must read zero too.
                np.testing.assert_array_equal(
                    slab.values, want, err_msg=f"{comp} on {kind}")
            else:
                _assert_close(slab.values, want, f"{comp} on {kind}")
            np.testing.assert_allclose(
                slab.thickness, [THICKNESS[k] for _e, _gp, k in rows],
                rtol=RTOL)
            assert not np.asarray(slab.sub_gp_index).any()


def _assert_quaternions_are_the_frames(
    r: Results, live: dict[tuple[int, int, int], np.ndarray],
    frames: dict[int, np.ndarray], kind: str,
) -> None:
    slab = r.elements.layers.get(component="fiber_stress_0")
    assert sorted(
        (int(e), int(gp), int(k))
        for e, gp, k in zip(slab.element_index, slab.gp_index, slab.layer_index)
    ) == sorted(live)
    q = np.asarray(slab.local_axes_quaternion, dtype=float)
    want = np.array([frames[int(e)] for e in slab.element_index])
    # A unit quaternion and its negative are one rotation.
    np.testing.assert_allclose(
        np.abs(np.einsum("ij,ij->i", q, want)), 1.0, rtol=0, atol=1e-12,
        err_msg=f"quaternion on {kind}")


@pytest.mark.parametrize("kind", KINDS)
def test_layer_stress_strain_and_thickness_equal_live(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    r, live, _frames = _layer_run(g, tmp_path, kind)
    _assert_layers_equal_live(r, live, kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "cls", [c for c in LAYER_CLASSES if c != "ASDShellQ4"],
)
def test_every_bridge_layered_shell_reads_back_live(
    g: Any, tmp_path: Path, kind: str, cls: str,
) -> None:
    """The other layered-shell classes the bridge emits: values, thickness,
    sub-GP and frame, the same oracles as the ASDShellQ4 cases."""
    r, live, frames = _layer_run(g, tmp_path, kind, cls)
    _assert_layers_equal_live(r, live, f"{cls} on {kind}")
    _assert_quaternions_are_the_frames(r, live, frames, f"{cls} on {kind}")


def _tilted_quad() -> np.ndarray:
    c = math.cos(math.radians(TILT_DEG))
    s = math.sin(math.radians(TILT_DEG))
    return np.array([[0, 0, 0], [1, 0, 0], [1, c, s], [0, c, s]], dtype=float)


def test_c7_quaternion_is_the_tilted_frame() -> None:
    """The oracle, closed form: local axes ``x = (1, 0, 0)``, ``y = (0, c,
    s)``, ``z = (0, -s, c)``. The stored quaternion is the rotation global
    -> local (rows = local axes), a turn of -30 deg about x."""
    half = math.radians(TILT_DEG) / 2
    np.testing.assert_allclose(
        shell_quaternion(_tilted_quad(), "ASDShellQ4"),
        [math.cos(half), -math.sin(half), 0.0, 0.0], atol=1e-14)


def test_c7_quaternion_matches_the_mpco_writer_convention(tmp_path: Path) -> None:
    """The oracle, cross-engine: a beam whose frame is the tilted shell's
    (``vecxz = z``) is stored by the fork's MPCO writer as the quaternion C7
    computes for the shell. A shell answers no ``localAxes``, so a beam is
    the only element the writer gives a frame for."""
    from apeGmsh.opensees.emitter.live import get_ops

    c = math.cos(math.radians(TILT_DEG))
    s = math.sin(math.radians(TILT_DEG))
    path = str(tmp_path / "frame.mpco")
    ops = get_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, 1.0, 0.0, 0.0)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 0.0, -s, c)
    ops.element("elasticBeamColumn", 1, 1, 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1)
    np.testing.assert_allclose(
        ops.eleResponse(1, "localAxes"), [1, 0, 0, 0, c, s, 0, -s, c],
        atol=1e-14)
    ops.recorder("mpco", path, "-N", "displacement")
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    ops.remove("recorders")
    ops.wipe()
    with h5py.File(path, "r") as f:
        stage = next(k for k in f if k.startswith("MODEL_STAGE"))
        axes = f[f"{stage}/MODEL/LOCAL_AXES"]
        assert [int(i) for i in axes["ID"][...]] == [1]
        stored = np.asarray(axes["LOCAL_AXES"][...], dtype=float).reshape(4)
    np.testing.assert_allclose(
        stored, shell_quaternion(_tilted_quad(), "ASDShellQ4"), atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_layer_quaternion_is_the_shell_frame(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    r, live, frames = _layer_run(g, tmp_path, kind)
    _assert_quaternions_are_the_frames(r, live, frames, kind)
