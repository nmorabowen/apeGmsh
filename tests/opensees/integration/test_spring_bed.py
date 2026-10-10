"""``ops.spring_bed`` — the emitted deck of a ground-spring bed (ADR 0118 D2).

The deck form under test is the San Ramón Tier-2 T2S topology
(``Tier2_springs.md`` section 6): a ground node offset and fixed, a 3-dof
side node at each structural node tied by ``equalDOF 1 2 3``, one
``zeroLength`` from ground to side with three ``Parallel <E1> <V1>
-factors k c`` materials over one shared ``Elastic 1.0`` / ``Viscous 1.0
1.0`` pair.

Oracles, independent of the code under test: the tributary areas of the
1.0 grid are written out here by position (corner 1/4, edge 1/2, interior
1); the expected factors are intensity times that area; node roles are
read back from the deck's own ``fix`` and ``equalDOF`` lines.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

LX, LY = 4.0, 2.0
ORIENT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
KX, KY, KZ = 2.0e3, 3.0e3, 5.0e3            # intensities per unit area
AX = 1.5                                    # x end-zone increment
CX, CZ = 10.0, 17.6                         # dashpot intensities


def _mat_fem(model_name: str = "spring_bed"):
    with apeGmsh(model_name=model_name, verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, LX, LY, label="mat")
        g.physical.add_surface("mat", name="Mat")
        side = g.decouple_node_set("Mat", label="side", tie_dofs=(1, 2, 3))
        gnd = g.decouple_node_set("Mat", offset=(0.0, 0.0, -5.0),
                                  label="ground")
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=2)
    return fem, side, gnd


def _shell_ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(
        E=30.0e3, nu=0.2, h=0.3, rho=0.0)
    ops.element.ShellMITC4(pg="Mat", section=sec)
    return ops


def _end_zone(xyz):
    return (np.abs(xyz[:, 0] - LX / 2) >= 1.5).astype(float)


def _k(xyz, area):
    kz = KZ * (1.0 + AX * _end_zone(xyz))
    return area[:, None] * np.column_stack(
        [np.full(len(xyz), KX), np.full(len(xyz), KY), kz])


def _c(xyz, area):
    return area[:, None] * np.array([CX, CX, CZ])


def _area_by_position(x: float, y: float) -> float:
    on_x = x in (0.0, LX)
    on_y = y in (0.0, LY)
    return 0.25 if (on_x and on_y) else 0.5 if (on_x or on_y) else 1.0


def _lines(deck: Path, head: str) -> list[list[str]]:
    return [ln.split() for ln in deck.read_text(encoding="utf-8").splitlines()
            if ln.startswith(head)]


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    fem, side, gnd = _mat_fem()
    ops = _shell_ops(fem)
    bed = ops.spring_bed(gnd, at=side, k=_k, c=_c, tributary="Mat",
                         orient=ORIENT)
    path = tmp_path_factory.mktemp("spring_bed") / "bed.tcl"
    with pytest.warns(UserWarning, match="Transformation"):
        ops.tcl(str(path))
    return fem, side, gnd, bed, path


def test_counts_match_the_deck_topology(deck) -> None:
    fem, side, gnd, bed, path = deck
    n = len(gnd.tags)
    assert n == len(bed) == 15
    assert len(_lines(path, "element zeroLength ")) == n
    assert len(_lines(path, "uniaxialMaterial Parallel ")) == 3 * n
    assert [t[1:2] + t[3:] for t in _lines(path, "uniaxialMaterial Elastic ")] \
        == [["Elastic", "1.0"]]
    assert [t[1:2] + t[3:] for t in _lines(path, "uniaxialMaterial Viscous ")] \
        == [["Viscous", "1.0", "1.0"]]
    assert len(_lines(path, "equalDOF ")) == n
    assert len(_lines(path, "fix ")) == n


def test_each_spring_joins_a_fixed_ground_to_a_tied_side_node(deck) -> None:
    fem, side, gnd, bed, path = deck
    fixed = {int(t[1]): t[2:] for t in _lines(path, "fix ")}
    tied = {int(t[2]): (int(t[1]), t[3:]) for t in _lines(path, "equalDOF ")}
    assert set(fixed) == set(gnd.tags)
    assert all(v == ["1", "1", "1"] for v in fixed.values())
    src_of_ground = {g: s for s, g in gnd.pairs().items()}
    for tok in _lines(path, "element zeroLength "):
        i, j = int(tok[3]), int(tok[4])
        assert i in fixed and j in tied
        master, dofs = tied[j]
        assert dofs == ["1", "2", "3"]
        assert master == src_of_ground[i]          # same structural node
        assert tok[tok.index("-dir") + 1:tok.index("-dir") + 4] == ["1", "2", "3"]
        o = tok.index("-orient")
        assert tok[o + 1:o + 7] == [repr(v) for v in ORIENT]
        assert "-doRayleigh" not in tok            # OpenSees default 0


def test_side_and_ground_nodes_are_three_dof(deck) -> None:
    fem, side, gnd, bed, path = deck
    ndf3 = {int(t[1]) for t in _lines(path, "node ")
            if t[-2:] == ["-ndf", "3"]}
    assert ndf3 == set(side.tags) | set(gnd.tags)


def test_factors_are_intensity_times_tributary_area(deck) -> None:
    fem, side, gnd, bed, path = deck
    xyz = {int(n): np.asarray(c, dtype=float)
           for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    par = {int(t[2]): (t[3:t.index("-factors")],
                       [float(v) for v in t[t.index("-factors") + 1:]])
           for t in _lines(path, "uniaxialMaterial Parallel ")}
    (unit_e,) = [int(t[2]) for t in _lines(path, "uniaxialMaterial Elastic ")]
    (unit_v,) = [int(t[2]) for t in _lines(path, "uniaxialMaterial Viscous ")]
    src_of_ground = {g: s for s, g in gnd.pairs().items()}
    total_area = 0.0
    for tok in _lines(path, "element zeroLength "):
        s = src_of_ground[int(tok[3])]
        x, y = float(xyz[s][0]), float(xyz[s][1])
        a = _area_by_position(x, y)
        total_area += a
        ez = 1.0 + AX * float(abs(x - LX / 2) >= 1.5)
        want = [(a * KX, a * CX), (a * KY, a * CX), (a * KZ * ez, a * CZ)]
        m = tok.index("-mat")
        for tag, (k, c) in zip(tok[m + 1:m + 4], want):
            members, factors = par[int(tag)]
            assert members == [str(unit_e), str(unit_v)]
            assert factors == pytest.approx([k, c], rel=1e-12)
    assert total_area == pytest.approx(LX * LY)
    assert bed.area is not None and bed.area.sum() == pytest.approx(LX * LY)


def test_t2s_worked_spring_lines_are_byte_equal(tmp_path) -> None:
    """Element 9290 of the 2B T2S deck: k and c per direction as arrays,
    the three Parallel lines equal the deck's up to the tags."""
    fem, side, gnd = _mat_fem("spring_bed_t2s")
    ops = _shell_ops(fem)
    n = len(gnd.tags)
    k = np.ones((n, 3))
    c = np.ones((n, 3))
    k[0] = (103866.00979083704, 107051.65850915055, 259060.47502682978)
    c[0] = (1938.5165844371518, 1938.5165844371518, 3413.5732236865174)
    ops.spring_bed(gnd, at=side, k=k, c=c, orient=ORIENT)
    path = tmp_path / "t2s.tcl"
    with pytest.warns(UserWarning, match="Transformation"):
        ops.tcl(str(path))
    text = path.read_text(encoding="utf-8")
    e1 = re.search(r"^uniaxialMaterial Elastic (\d+) 1\.0$", text, re.M)
    v1 = re.search(r"^uniaxialMaterial Viscous (\d+) 1\.0 1\.0$", text, re.M)
    assert e1 and v1
    first = next(t for t in _lines(path, "element zeroLength ")
                 if int(t[3]) == gnd.tags[0])
    m = first.index("-mat")
    mats = first[m + 1:m + 4]
    lines = {ln.split()[2]: ln for ln in text.splitlines()
             if ln.startswith("uniaxialMaterial Parallel ")}
    tail = [
        "-factors 103866.00979083704 1938.5165844371518",
        "-factors 107051.65850915055 1938.5165844371518",
        "-factors 259060.47502682978 3413.5732236865174",
    ]
    for tag, t in zip(mats, tail):
        assert lines[tag] == (
            f"uniaxialMaterial Parallel {tag} {e1.group(1)} {v1.group(1)} {t}")
    assert " ".join(first[5:]) == (
        f"-mat {' '.join(mats)} -dir 1 2 3 -orient 1.0 0.0 0.0 0.0 1.0 0.0")


def test_options_rayleigh_no_dashpot_shared_units(tmp_path) -> None:
    with apeGmsh(model_name="spring_bed_opts", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0, label="p")
        g.physical.add_surface("p", name="P")
        a = g.decouple_node_set("P", label="a")
        b = g.decouple_node_set("P", offset=(0.0, 0.0, -1.0), label="b")
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=2)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.spring_bed(a, k=(1.0, 2.0, 3.0), do_rayleigh=True)
    ops.spring_bed(b, k=(4.0, 5.0, 6.0), c=(0.5, 0.5, 0.5), dirs=(1, 2, 3))
    path = tmp_path / "opts.tcl"
    ops.tcl(str(path))
    assert len(_lines(path, "uniaxialMaterial Elastic ")) == 1
    assert len(_lines(path, "uniaxialMaterial Viscous ")) == 1
    pars = _lines(path, "uniaxialMaterial Parallel ")
    no_dash = [t for t in pars if len(t) == 6]       # Parallel t E -factors k
    assert len(no_dash) == 12 and len(pars) == 24
    zl = _lines(path, "element zeroLength ")
    a_lines = [t for t in zl if int(t[3]) in set(a.tags)]
    assert a_lines and all(t[-2:] == ["-doRayleigh", "1"] for t in a_lines)
    # at=None: the spring reaches the source (mesh) node itself.
    assert {int(t[4]) for t in a_lines} == set(a.source_ids)


@pytest.fixture(scope="module")
def setup():
    return _mat_fem("spring_bed_refusals")


class TestRefusals:
    def _ops(self, fem):
        return _shell_ops(fem)

    def test_not_a_node_set(self, setup) -> None:
        fem, side, gnd = setup
        with pytest.raises(TypeError, match="decouple_node_set"):
            self._ops(fem).spring_bed(gnd.tags, k=(1.0, 1.0, 1.0))  # type: ignore[arg-type]

    def test_wrong_k_shape(self, setup) -> None:
        fem, side, gnd = setup
        with pytest.raises(ValueError, match="k must be 3 values"):
            self._ops(fem).spring_bed(gnd, at=side, k=np.ones((2, 3)))

    def test_negative_k(self, setup) -> None:
        fem, side, gnd = setup
        with pytest.raises(ValueError, match=">= 0"):
            self._ops(fem).spring_bed(gnd, at=side, k=(1.0, -1.0, 1.0))

    def test_dirs_beyond_ndf(self, setup) -> None:
        fem, side, gnd = setup
        with pytest.raises(ValueError, match="dirs"):
            self._ops(fem).spring_bed(gnd, at=side, k=(1.0,) * 4,
                                      dirs=(1, 2, 3, 4))

    def test_unknown_tributary(self, setup) -> None:
        fem, side, gnd = setup
        with pytest.raises(ValueError, match="tributary"):
            self._ops(fem).spring_bed(gnd, at=side, k=_k, tributary="Nope")

    def test_refusal_registers_nothing(self, setup) -> None:
        fem, side, gnd = setup
        ops = self._ops(fem)
        before = len(ops._primitives)
        with pytest.raises(ValueError):
            ops.spring_bed(gnd, at=side, k=(1.0, float("nan"), 1.0))
        assert len(ops._primitives) == before
