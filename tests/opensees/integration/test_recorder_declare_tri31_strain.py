"""#2892: a bridge ``recorder.declare`` gauss-strain record on Tri31.

Tri31 has no element-level ``strains`` response (``PER_MATERIAL_STRAIN_CLASSES``
in the response catalog): a ``recorder Element ... strains`` on it writes the
time column alone, and nothing warned. The bridge now reads the record's
classes from its element plan and records Tri31 strain per Gauss point
(``material <gp> strain``, the catalog's point count), refusing a record that
mixes Tri31 with a class that answers ``strains`` and an explicit
``element_class_name`` that contradicts the groups.

Emit lanes run on a hand-rolled FEM stub. The ``live`` lane runs a real Gmsh
plate in uniaxial plane stress, whose closed form is exact on constant-strain
triangles: every element holds ``eps_xx = q/E``, ``eps_yy = -nu q/E``,
``gamma_xy = 0``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import ModelData, apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

STRAIN = ("strain_xx", "strain_yy", "strain_xy")


def _fem() -> FEMStub:
    """Two unit squares side by side.

    * ``TriA``: Tri31 elements 1 ``(1, 2, 3)`` and 2 ``(1, 3, 4)``.
    * ``TriB``: Tri31 element 3 ``(2, 5, 6)``.
    * ``Quad``: FourNodeQuad element 4 ``(2, 5, 6, 3)``.
    * ``Empty``: no elements.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6],
        coords=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (2, 0, 0), (2, 1, 0)],
        node_pgs={"Base": [1, 2, 5]},
    )
    elements = _ElementsStub(elem_pgs={
        "TriA": _ElementGroupView(ids=(1, 2),
                                  connectivity=((1, 2, 3), (1, 3, 4))),
        "TriB": _ElementGroupView(ids=(3,), connectivity=((2, 5, 6),)),
        "Quad": _ElementGroupView(ids=(4,), connectivity=((2, 5, 6, 3),)),
        "Empty": _ElementGroupView(ids=(), connectivity=cast(
            "tuple[tuple[int, ...], ...]", np.empty((0, 3), dtype=np.int64))),
    })
    return FEMStub(nodes=nodes, elements=elements)


def _ops(*, quad: bool = False, tri_b: bool = True) -> apeSees:
    ops = apeSees(cast("object", _fem()))
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1000.0, nu=0.25)
    ops.element.Tri31(pg="TriA", material=mat, thickness=1.0)
    if tri_b:
        ops.element.Tri31(pg="TriB", material=mat, thickness=1.0)
    if quad:
        ops.element.FourNodeQuad(pg="Quad", material=mat, thickness=1.0)
    return ops


def _recorders(ops: apeSees) -> list[tuple]:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return [args for name, args, _kw in rec.calls if name == "recorder"]


def _element_tags(ops: apeSees) -> dict[str, list[int]]:
    """``{element token: emitted tags}``, read from the emitted deck."""
    rec = RecordingEmitter()
    ops.build().emit(rec)
    out: dict[str, list[int]] = {}
    for name, args, _kw in rec.calls:
        if name == "element":
            out.setdefault(str(args[0]), []).append(int(args[1]))
    return out


def _ele(args: tuple) -> list[int]:
    """The ``-ele`` tags of one ``recorder Element`` call."""
    i = args.index("-ele") + 1
    out: list[int] = []
    while i < len(args) and isinstance(args[i], int):
        out.append(args[i])
        i += 1
    return out


# ---------------------------------------------------------------------------
# Emit: the per-material route
# ---------------------------------------------------------------------------

class TestTri31StrainEmit:
    def test_pg_strain_declare_emits_per_material_strain(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg="TriA", name="s")
        (args,) = _recorders(ops)
        assert args[0] == "Element"
        assert args[args.index("-file") + 1] == "./s__default__gauss_material1.out"
        # Tri31 is one Gauss point (Triangle_GL_1): one line, point 1.
        assert args[-3:] == ("material", "1", "strain")
        assert "strains" not in args
        assert _ele(args) == _element_tags(ops)["tri31"][:2]

    def test_explicit_tri31_class_matching_the_groups_is_honoured(
        self,
    ) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg="TriA", name="s",
                             element_class_name="Tri31")
        (args,) = _recorders(ops)
        assert args[-3:] == ("material", "1", "strain")

    def test_dt_is_carried_on_the_per_material_line(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg="TriA", name="s", dt=0.5)
        (args,) = _recorders(ops)
        assert args[args.index("-dT") + 1] == 0.5
        assert args[-3:] == ("material", "1", "strain")

    def test_one_record_over_two_tri_pgs(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg=("TriA", "TriB"), name="s")
        (args,) = _recorders(ops)
        assert args[-3:] == ("material", "1", "strain")
        assert sorted(_ele(args)) == sorted(_element_tags(ops)["tri31"])

    def test_several_records_on_one_pg(self) -> None:
        """Strain goes per material; stress on the same PG keeps the
        element-level ``stresses`` Tri31 answers."""
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg="TriA", name="e")
        ops.recorder.declare(gauss=("stress_xx",), pg="TriA", name="s")
        strain, stress = _recorders(ops)
        assert strain[-3:] == ("material", "1", "strain")
        assert stress[-1] == "stresses"
        assert stress[stress.index("-file") + 1] == "./s__default__gauss.out"

    def test_ids_targets_resolve_the_class_from_the_plan(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, ids=(1, 2), name="s")
        (args,) = _recorders(ops)
        assert args[-3:] == ("material", "1", "strain")

    def test_empty_pg_beside_a_tri_pg(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg=("Empty", "TriA"), name="s")
        (args,) = _recorders(ops)
        assert args[-3:] == ("material", "1", "strain")
        assert len(_ele(args)) == 2

    def test_empty_pg_alone_emits_nothing(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg="Empty", name="s")
        assert _recorders(ops) == []

    def test_quad_strain_is_unchanged(self) -> None:
        ops = _ops(quad=True)
        ops.recorder.declare(gauss=STRAIN, pg="Quad", name="s")
        (args,) = _recorders(ops)
        assert args[-1] == "strains"
        assert args[args.index("-file") + 1] == "./s__default__gauss.out"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

class TestRefusals:
    def test_mixed_tri31_and_quad_pgs_refuse_naming_both(self) -> None:
        ops = _ops(quad=True)
        ops.recorder.declare(gauss=STRAIN, pg=("TriA", "Quad"), name="s")
        with pytest.raises(BridgeError) as ei:
            _recorders(ops)
        msg = str(ei.value)
        assert "'Tri31'" in msg and "'FourNodeQuad'" in msg
        assert "own record" in msg

    def test_mixed_ids_refuse_naming_both(self) -> None:
        ops = _ops(quad=True)
        ops.recorder.declare(gauss=STRAIN, ids=(1, 4), name="s")
        with pytest.raises(BridgeError, match="'Tri31'.*'FourNodeQuad'"):
            _recorders(ops)

    def test_explicit_class_contradicting_a_tri_pg_refuses(self) -> None:
        ops = _ops()
        ops.recorder.declare(gauss=STRAIN, pg="TriA", name="s",
                             element_class_name="FourNodeQuad")
        with pytest.raises(BridgeError) as ei:
            _recorders(ops)
        msg = str(ei.value)
        assert "element_class_name='FourNodeQuad'" in msg
        assert "'Tri31'" in msg

    def test_explicit_tri31_contradicting_a_quad_pg_refuses(self) -> None:
        ops = _ops(quad=True)
        ops.recorder.declare(gauss=STRAIN, pg="Quad", name="s",
                             element_class_name="Tri31")
        with pytest.raises(BridgeError) as ei:
            _recorders(ops)
        msg = str(ei.value)
        assert "element_class_name='Tri31'" in msg
        assert "'FourNodeQuad'" in msg

    def test_refusal_writes_no_deck(self, tmp_path: Path) -> None:
        """The raise comes at emit, after registration: no deck file is
        left behind for a run to pick up."""
        ops = _ops(quad=True)
        ops.recorder.declare(gauss=STRAIN, pg=("TriA", "Quad"), name="s")
        deck = tmp_path / "d.tcl"
        with pytest.raises(BridgeError, match="'Tri31'"):
            ops.tcl(str(deck))
        assert not deck.exists()
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# ModelData: no element plan
# ---------------------------------------------------------------------------

class TestModelData:
    def test_explicit_tri31_is_honoured(self) -> None:
        md = ModelData(cast("object", _fem()), ndm=2, ndf=2)
        md.recorders(gauss=STRAIN, pg="TriA", name="s",
                     element_class_name="Tri31")
        lines = md.recorder_commands(target="tcl")
        rec = [ln for ln in lines if ln.startswith("recorder")]
        assert len(rec) == 1
        assert rec[0].endswith("material 1 strain"), rec[0]
        assert "s__default__gauss_material1.out" in rec[0]

    def test_no_class_keeps_strains(self) -> None:
        """No plan and no class: nothing to resolve from (out of scope)."""
        md = ModelData(cast("object", _fem()), ndm=2, ndf=2)
        md.recorders(gauss=STRAIN, pg="TriA", name="s")
        rec = [ln for ln in md.recorder_commands(target="tcl")
               if ln.startswith("recorder")]
        assert len(rec) == 1 and rec[0].endswith("strains"), rec


# ---------------------------------------------------------------------------
# Live: the file holds strain
# ---------------------------------------------------------------------------

E, NU, Q = 1000.0, 0.25, 10.0
W, H = 2.0, 1.0


@pytest.mark.live
def test_live_tri31_strain_file_holds_the_closed_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plate in uniaxial plane stress: ``sigma_xx = q`` (thickness 1).
    Rollers on the left (x) and the base (y) leave the Poisson
    contraction free, so every constant-strain triangle holds
    ``(q/E, -nu q/E, 0)``."""
    pytest.importorskip("gmsh")
    pytest.importorskip("openseespy.opensees")
    from apeGmsh import apeGmsh

    monkeypatch.chdir(tmp_path)
    with apeGmsh(model_name="tri31_strain", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, W, H, label="plate")
        g.physical.add_surface("plate", name="Plate")
        sel = g.model.select(dim=1)
        sel.on_plane((0, 0, 0), (1, 0, 0), tol=1e-6).to_physical("Left")
        g.model.select(dim=1).on_plane(
            (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Bottom")
        g.model.select(dim=1).on_plane(
            (W, 0, 0), (1, 0, 0), tol=1e-6).to_physical("Right")
        with g.loads.case("pull"):
            g.loads.line(pg="Right", magnitude=Q, direction=(1.0, 0.0, 0.0))
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(2)
        g.mesh.partitioning.renumber(base=1)
        fem = g.mesh.queries.get_fem_data(dim=2)

        ops = apeSees(fem)
        ops.model(ndm=2, ndf=2)
        mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
        ops.element.Tri31(pg="Plate", material=mat, thickness=1.0,
                          plane_type="PlaneStress")
        ops.fix(pg="Left", dofs=(1, 0))
        ops.fix(pg="Bottom", dofs=(0, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("pull")
        ops.recorder.declare(gauss=STRAIN, pg="Plate", name="s",
                             file_root=str(tmp_path))
        ops.constraints.Plain()
        ops.numberer.RCM()
        ops.system.BandGeneral()
        ops.test.NormDispIncr(tol=1e-12, max_iter=10)
        ops.algorithm.Newton()
        ops.integrator.LoadControl(dlam=1.0)
        ops.analysis.Static()
        n_ele = len(fem.elements.select(pg="Plate").ids)
        deck = tmp_path / "deck.py"
        ops.py(str(deck), run=True, analyze_steps=1, python=sys.executable)

    assert not (tmp_path / "s__default__gauss.out").exists()
    data = np.atleast_2d(np.loadtxt(tmp_path / "s__default__gauss_material1.out"))
    # The time column, then 3 strain components per element: not the
    # time column alone.
    assert data.shape == (1, 1 + 3 * n_ele), data.shape
    assert n_ele >= 4
    strain = data[0, 1:].reshape(n_ele, 3)
    expected = np.array([Q / E, -NU * Q / E, 0.0])
    np.testing.assert_allclose(
        strain, np.broadcast_to(expected, strain.shape), rtol=1e-8, atol=1e-12)
