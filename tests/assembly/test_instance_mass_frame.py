"""A rotated instance turns its anisotropic nodal masses or refuses (#1600).

The instance frame rule (ADR 0117 INV-6, the mass note): a mass value per
DOF is a diagonal tensor, not a DOF index, so the translational triple
``(mx, my, mz)`` and the rotary triple ``(Ixx, Iyy, Izz)`` turn as
``R diag(d) Rᵀ``. An axis-aligned rotation permutes the values; a rotation
that makes the tensor non-diagonal raises, because OpenSees ``mass`` takes
a diagonal only. Before, the merge copied every mass row verbatim, so a
rotated anisotropic mass was silently wrong.

Oracles, each independent of the code under test:

* closed form: ``R diag(d) Rᵀ`` for a 90 degree turn about z carries the
  x value onto y and the y value onto x; about x it swaps y and z; the
  120 degree turn about ``(1, 1, 1)`` is the cyclic map ``x -> y -> z``.
  Each is asserted with ``==`` on records built by hand, so a surviving
  value is byte-identical to its source.
* lock: every field of ``MassRecord`` is classified in ``_MASS_FIELDS``.
* deck: a v2 instance turned 90 degrees about z carries ``mass_from_model``
  into the deck with ``mx/my`` and ``Ixx/Iyy`` swapped; the same archive
  turned 30 degrees refuses at ``bridge()``.
"""
from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.record_sets import MassSet
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh.mesh._compose import ComposeError, _rewrite_mass_set

ROT_Z_90 = (0.0, 0.0, 1.0, math.pi / 2.0)
ROT_X_90 = (1.0, 0.0, 0.0, math.pi / 2.0)
ROT_Z_180 = (0.0, 0.0, 1.0, math.pi)
ROT_111_120 = (1.0, 1.0, 1.0, 2.0 * math.pi / 3.0)
ROT_Z_30 = (0.0, 0.0, 1.0, math.pi / 6.0)
ANISO = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
ISO = (7.5, 7.5, 7.5, 0.25, 0.25, 0.25)


def _rewrite(records, rotate, *, offset=100, label="c"):
    return _rewrite_mass_set(MassSet(records), offset=offset, label=label,
                             rotate=rotate)


def _rows(mass_set) -> list[tuple[float, ...]]:
    return [tuple(float(v) for v in r.mass) for r in mass_set]


# ---------------------------------------------------------------------------
# Closed form: axis-aligned rotations permute the values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotate, want", [
    (ROT_Z_90, (2.0, 1.0, 3.0, 5.0, 4.0, 6.0)),      # x value lands on y
    (ROT_X_90, (1.0, 3.0, 2.0, 4.0, 6.0, 5.0)),      # y value lands on z
    (ROT_Z_180, ANISO),                               # a half turn keeps all
    (ROT_111_120, (3.0, 1.0, 2.0, 6.0, 4.0, 5.0)),   # cyclic x -> y -> z
])
def test_axis_aligned_rotation_permutes_anisotropic_masses(rotate, want):
    got = _rewrite([MassRecord(node_id=1, mass=ANISO, name="tip")], rotate)
    assert _rows(got) == [want]
    (rec,) = got
    assert rec.node_id == 101
    assert rec.name == "c.tip"


def test_ninety_degree_turn_about_z_swaps_mx_my_and_ixx_iyy():
    """The #1600 case: on main the row is copied verbatim and this fails."""
    (rec,) = _rewrite([MassRecord(node_id=1, mass=ANISO)], ROT_Z_90)
    mx, my, mz, ixx, iyy, izz = rec.mass
    assert (mx, my) == (ANISO[1], ANISO[0])
    assert (ixx, iyy) == (ANISO[4], ANISO[3])
    assert (mz, izz) == (ANISO[2], ANISO[5])


def test_rotary_triple_turns_on_its_own():
    rec = MassRecord(node_id=1, mass=(5.0, 5.0, 5.0, 1.0, 2.0, 3.0))
    assert _rows(_rewrite([rec], ROT_Z_90)) == [(5.0, 5.0, 5.0, 2.0, 1.0, 3.0)]


# ---------------------------------------------------------------------------
# Refusal: a turn that makes the tensor non-diagonal
# ---------------------------------------------------------------------------

def test_thirty_degree_turn_of_an_anisotropic_mass_raises():
    recs = [MassRecord(node_id=1, mass=ISO), MassRecord(node_id=2, mass=ANISO),
            MassRecord(node_id=3, mass=ISO)]
    with pytest.raises(ComposeError) as info:
        _rewrite(recs, ROT_Z_30, label="wing")
    msg = str(info.value)
    assert "'wing'" in msg and "30 degrees" in msg
    assert re.search(r"node\(s\) 2 into", msg), msg
    assert "multiple of 90 degrees" in msg


def test_thirty_degree_turn_of_an_anisotropic_rotary_inertia_raises():
    rec = MassRecord(node_id=4, mass=(1.0, 1.0, 1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ComposeError, match="rotary inertia on node\\(s\\) 4"):
        _rewrite([rec], ROT_Z_30)


def test_transversely_isotropic_triple_survives_a_turn_about_its_axis():
    """``diag(m, m, m3)`` turned about z stays diagonal: kept, verbatim."""
    rec = MassRecord(node_id=1, mass=(1.0, 1.0, 3.0, 0.5, 0.5, 9.0))
    assert _rows(_rewrite([rec], ROT_Z_30)) == [rec.mass]


# ---------------------------------------------------------------------------
# Byte identity: isotropic rows, rotate=None, empty sets, degenerate axes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotate", [None, ROT_Z_30, ROT_111_120, ROT_Z_90,
                                    (0.0, 0.0, 0.0, 1.0)])
def test_isotropic_masses_are_byte_identical_under_any_rotation(rotate):
    src = MassSet([MassRecord(node_id=1, mass=ISO, name="a"),
                   MassRecord(node_id=2, mass=(0.1, 0.1, 0.1, 0.0, 0.0, 0.0))])
    before = src.mass_array().tobytes()
    got = _rewrite_mass_set(src, offset=10, label="c", rotate=rotate)
    assert got.mass_array().tobytes() == before
    assert src.mass_array().tobytes() == before
    assert got.node_ids().tolist() == [11, 12]
    assert [r.name for r in got] == ["c.a", None]


def test_rotate_none_keeps_an_anisotropic_mass_verbatim():
    assert _rows(_rewrite([MassRecord(node_id=1, mass=ANISO)], None)) == [ANISO]


def test_empty_mass_set_under_a_rotation_is_empty():
    assert len(_rewrite([], ROT_Z_30)) == 0


def test_record_fallback_path_turns_too():
    """A plain iterable of records (no columnar accessors) turns alike."""
    got = _rewrite_mass_set(
        iter([MassRecord(node_id=1, mass=ANISO)]), offset=5, label="c",
        rotate=ROT_Z_90)
    assert _rows(got) == [(2.0, 1.0, 3.0, 5.0, 4.0, 6.0)]
    with pytest.raises(ComposeError, match="node\\(s\\) 1"):
        _rewrite_mass_set(iter([MassRecord(node_id=1, mass=ANISO)]), offset=5,
                          label="c", rotate=ROT_Z_30)


# ---------------------------------------------------------------------------
# Lock: every MassRecord field is classified
# ---------------------------------------------------------------------------

def test_every_mass_field_is_classified():
    from apeGmsh.mesh._compose import _MASS_FIELDS

    table = _MASS_FIELDS["MassRecord"]
    assert {f.name for f in dataclasses.fields(MassRecord)} == set(table)
    assert [f for f, c in table.items() if c == "diagonal"] == ["mass"]


# ---------------------------------------------------------------------------
# Deck: the v2 Assembly path reaches the rewrite
# ---------------------------------------------------------------------------

def _archive(workdir: Path) -> Path:
    """One column along +z with ``ANISO`` on its tip and ``ISO`` at its base."""
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="col", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, 0.0, 3.0)
        ln = geo.add_line(p0, p1)
        g.model.sync()
        g.physical.add(1, [ln], name="Col")
        g.physical.add(0, [p1], name="Tip")
        g.physical.add(0, [p0], name="Base")
        g.mesh.sizing.set_global_size(3.0)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    (tip,) = (int(i) for i in fem.nodes.select(pg="Tip").ids)
    (base,) = (int(i) for i in fem.nodes.select(pg="Base").ids)
    fem = fem.with_mass(MassRecord(node_id=tip, mass=ANISO, name="tipmass"))
    fem = fem.with_mass(MassRecord(node_id=base, mass=ISO))
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), name="t")
    ops.element.elasticBeamColumn(pg="Col", transf=t, A=100.0, E=2.0e5,
                                  Iz=8.0e3, Iy=2.0e3, G=8.0e4, J=5.0e3)
    path = workdir / "col.h5"
    ops.h5(str(path))
    return path


def _bridge(workdir: Path, rotate):
    from apeGmsh.assembly import Assembly

    ops = (Assembly("turned")
           .instance("c", _archive(workdir), rotate=rotate,
                     translate=(10.0, 20.0, 30.0))
           .bridge(ndm=3, ndf=6))
    ops.mass_from_model()
    return ops


def _mass_lines(ops, path: Path) -> dict[int, tuple[float, ...]]:
    ops.tcl(str(path), flat=True)
    out: dict[int, tuple[float, ...]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        tok = ln.split()
        if tok[:1] == ["mass"]:
            out[int(tok[1])] = tuple(float(v) for v in tok[2:8])
    return out


def test_rotated_instance_deck_carries_permuted_masses(tmp_path):
    ops = _bridge(tmp_path, ((0.0, 0.0, 1.0), math.pi / 2.0))
    (tip,) = (int(i) for i in ops.fem.nodes.select(pg="c.Tip").ids)
    (base,) = (int(i) for i in ops.fem.nodes.select(pg="c.Base").ids)
    got = _mass_lines(ops, tmp_path / "turned.tcl")
    assert got[tip] == (2.0, 1.0, 3.0, 5.0, 4.0, 6.0)
    assert got[base] == ISO
    assert set(got) == {tip, base}


def test_unrotated_instance_deck_keeps_source_masses(tmp_path):
    ops = _bridge(tmp_path, None)
    (tip,) = (int(i) for i in ops.fem.nodes.select(pg="c.Tip").ids)
    assert _mass_lines(ops, tmp_path / "plain.tcl")[tip] == ANISO


def test_thirty_degree_instance_refuses_at_bridge(tmp_path):
    with pytest.raises(ComposeError, match="instance 'c' is rotated by 30"):
        _bridge(tmp_path, ((0.0, 0.0, 1.0), math.pi / 6.0))


def test_refusal_leaves_the_host_model_unchanged(tmp_path):
    """The raise fires inside the rewrite, before any record is merged."""
    from apeGmsh.mesh._compose import _compose_module

    src = _archive(tmp_path)
    ops = _bridge(tmp_path, ((0.0, 0.0, 1.0), math.pi / 2.0))
    host = ops.fem
    n_mass = len(host.nodes.masses)
    n_nodes = host.nodes.ids.size
    n_modules = len(host.composed_from)
    masses_before = host.nodes.masses.mass_array().tobytes()
    with pytest.raises(ComposeError, match="instance 'd'"):
        _compose_module(host, src, label="d", translate=(0.0, 0.0, 0.0),
                        rotate=ROT_Z_30, partition_rank=None)
    assert len(host.nodes.masses) == n_mass
    assert host.nodes.ids.size == n_nodes
    assert len(host.composed_from) == n_modules
    assert host.nodes.masses.mass_array().tobytes() == masses_before
