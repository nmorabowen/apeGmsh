"""Integration tests — ``/opensees/computed_sections`` provenance
sidecar (ADR 0078 Amendment A1, schema 2.20.0).

The resolved elastic numbers already persist through the ordinary
section capture; this sidecar adds provenance rows
``(tag, analyzer_name, payload_json)``.  Provenance is metadata, not
authored model state: excluded from ``model_hash``, group written only
when a ``ComputedSection`` emitted (byte-invariance otherwise), and
round-tripped by ``OpenSeesModel``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, cast

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal._computed_sections_h5 import (
    read_computed_sections,
)
from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION
from apeGmsh.sections import SectionMaterial, SectionProperties

from tests.fixtures.schema import OPENSEES_CURRENT
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _analyzer(g, *, name="PG24", material=None):
    g.sections.rect_face(2.0, 4.0, label="bar")
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=2)
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    if material is None:
        material = SectionMaterial(E=200e3, nu=0.3, fy=345.0)
    return SectionProperties(fem, materials={"bar": material}, name=name)


def _frame(section_factory) -> apeSees:
    """One-column frame on real FEMData; the section comes from
    ``section_factory(ops)``."""
    fem = build_simple_frame_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    sec = section_factory(ops)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    return ops


# --------------------------------------------------------------------- #
# Persistence + payload
# --------------------------------------------------------------------- #

def test_elastic_kind_writes_provenance(g, tmp_path: Path) -> None:
    sec = _analyzer(g)
    p = tmp_path / "model.h5"
    _frame(
        lambda ops: ops.section.ComputedSection(analysis=sec)
    ).h5(str(p))

    rows = read_computed_sections(str(p))
    assert len(rows) == 1
    tag, analyzer_name, payload_json = rows[0]
    assert analyzer_name == "PG24"
    payload = json.loads(payload_json)
    assert payload["kind"] == "elastic"
    assert payload["ndm"] == 3
    assert payload["E_ref"] == pytest.approx(200e3)
    assert payload["disconnected"] == "raise"
    assert payload["geometric_only"] is False
    assert payload["n_parts"] == 1
    assert payload["materials"]["bar"]["E"] == pytest.approx(200e3)
    assert payload["materials"]["bar"]["fy"] == pytest.approx(345.0)

    # the tag joins to the ordinary section capture
    with h5py.File(str(p), "r") as f:
        sec_tags = {
            int(f["opensees/sections"][k].attrs["tag"])
            for k in f["opensees/sections"]
        }
    assert tag in sec_tags


def test_fiber_kind_writes_provenance(g, tmp_path: Path) -> None:
    sec = _analyzer(g, name="fib24")
    p = tmp_path / "model.h5"

    def factory(ops):
        mat = ops.uniaxialMaterial.ElasticMaterial(E=200e3)
        return ops.section.ComputedSection(
            analysis=sec, kind="fiber", fibers={"bar": mat},
        )

    _frame(factory).h5(str(p))
    rows = read_computed_sections(str(p))
    assert len(rows) == 1
    payload = json.loads(rows[0][2])
    assert payload["kind"] == "fiber"
    assert payload["fiber_pgs"] == ["bar"]
    assert payload["GJ"] == pytest.approx(sec.warping().GJ)
    assert "E_ref" not in payload


def test_no_computed_section_writes_no_group(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _frame(
        lambda ops: ops.section.Elastic(
            E=200e3, A=8.0, Iz=10.7, Iy=2.7, G=76.9e3, J=7.0,
            alphaY=0.83, alphaZ=0.83,
        )
    ).h5(str(p))
    assert read_computed_sections(str(p)) == ()
    with h5py.File(str(p), "r") as f:
        assert "computed_sections" not in f["opensees"]


def test_schema_stamped_current(g, tmp_path: Path) -> None:
    sec = _analyzer(g)
    p = tmp_path / "model.h5"
    _frame(lambda ops: ops.section.ComputedSection(analysis=sec)).h5(str(p))
    with h5py.File(str(p), "r") as f:
        assert f["meta"].attrs["opensees_schema_version"] == SCHEMA_VERSION
        assert SCHEMA_VERSION == OPENSEES_CURRENT


# --------------------------------------------------------------------- #
# Lineage stability — provenance must not perturb model_hash
# --------------------------------------------------------------------- #

def _model_hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


def test_provenance_excluded_from_model_hash(g, tmp_path: Path) -> None:
    """A ComputedSection deck is byte-identical to the hand-typed
    ElasticSection with the same numbers (the S5 promise) — with the
    sidecar hash-excluded, the model_hash must be identical too."""
    sec = _analyzer(g)
    hand = sec.to_elastic_section()
    computed = tmp_path / "computed.h5"
    typed = tmp_path / "hand.h5"
    _frame(
        lambda ops: ops.section.ComputedSection(analysis=sec)
    ).h5(str(computed))
    _frame(
        lambda ops: ops.section.Elastic(
            E=hand.E, A=hand.A, Iz=hand.Iz, Iy=hand.Iy,
            G=hand.G, J=hand.J, alphaY=hand.alphaY, alphaZ=hand.alphaZ,
        )
    ).h5(str(typed))
    assert _model_hash(computed) == _model_hash(typed)


# --------------------------------------------------------------------- #
# OpenSeesModel surface + round-trip
# --------------------------------------------------------------------- #

@pytest.mark.filterwarnings(
    "error::apeGmsh.opensees._internal.replay_types.ReplaySlotTypeWarning")
def test_opensees_model_surfaces_and_round_trips(g, tmp_path: Path) -> None:
    """The round-trip recovers the section's slot types without a
    ``ReplaySlotTypeWarning`` (#2374): the analyzer is archived by name
    only, and the slot types do not need it."""
    sec = _analyzer(g)
    src = tmp_path / "src.h5"
    dst = tmp_path / "dst.h5"
    _frame(lambda ops: ops.section.ComputedSection(analysis=sec)).h5(str(src))

    om = OpenSeesModel.from_h5(str(src))
    rows = om.computed_sections()
    assert len(rows) == 1 and rows[0][1] == "PG24"

    om.to_h5(str(dst))
    assert read_computed_sections(str(dst)) == read_computed_sections(str(src))
    # round-trip is hash-stable
    assert _model_hash(dst) == _model_hash(src)


# --------------------------------------------------------------------- #
# Replay slot types (#2374): the replayed section line is the bridge's
# --------------------------------------------------------------------- #

def _fiber_on(ops, sec, **kw):
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e3)
    return ops.section.ComputedSection(
        analysis=sec, kind="fiber", fibers={"bar": mat}, **kw)


def _elastic(**kw):
    return lambda ops, sec: ops.section.ComputedSection(analysis=sec, **kw)


#: case -> (analyzer material, section factory, {token index: the bridge's
#: literal}). The literals pin the bridge side of the oracle: the 3-D line
#: is ``section Elastic tag E A Iz Iy G J alphaY alphaZ`` (E at 3, G at 7),
#: the 2-D one ``section Elastic tag E A Iz G alphaY`` (G at 6).
_REPLAY_CASES = {
    # E and G default from a float material: 200000.0, not 200000
    "elastic-float-E": (None, _elastic(), {3: "200000.0"}),
    "elastic-float-E-G": (SectionMaterial(E=200e3, nu=0.3, G=80e3),
                          _elastic(), {3: "200000.0", 7: "80000.0"}),
    # an int material modulus stays an int; the computed G is a float
    "elastic-int-E-nu": (SectionMaterial(E=200000, nu=0.3), _elastic(),
                         {3: "200000"}),
    "elastic-int-E-int-G": (SectionMaterial(E=200000, nu=0.3, G=80000),
                            _elastic(), {3: "200000", 7: "80000"}),
    # moduli the user set: an int stays an int, an integral float a float
    "elastic-2d-set-moduli": (None, _elastic(ndm=2, E=210000, G=80000.0),
                              {3: "210000", 6: "80000.0"}),
    "fiber": (None, lambda ops, sec: _fiber_on(ops, sec), {}),
    "fiber-int-gj": (None, lambda ops, sec: _fiber_on(ops, sec, GJ=5),
                     {4: "5"}),
}


def _section_lines(deck: str) -> list[str]:
    return [ln for ln in deck.splitlines() if ln.startswith("section ")]


def _bridge_and_replay(
        g, tmp_path: Path, case: str, *, drop_sidecar: bool = False,
) -> tuple[list[str], list[str]]:
    from apeGmsh._core import default_artifact_dir
    from apeGmsh.opensees._internal.artifact_write import BridgeArtifactWarning

    material, factory, _literals = _REPLAY_CASES[case]
    sec = _analyzer(g, material=material)
    ops = _frame(lambda o: factory(o, sec))
    deck = tmp_path / "bridge.tcl"
    archive = tmp_path / "model.h5"
    # The deck emit writes the automatic model.h5 (ADR 0112 D1).  A
    # ComputedSection's analyzer must pickle into the P5 digest, or the
    # write is skipped with only a warning (#2567).
    with warnings.catch_warnings():
        warnings.simplefilter("error", BridgeArtifactWarning)
        ops.tcl(str(deck), progress=False, flat=True)
    auto = default_artifact_dir() / "simple_frame.h5"
    assert auto.is_file()
    assert len(read_computed_sections(str(auto))) == 1
    ops.h5(str(archive))
    if drop_sidecar:
        with h5py.File(str(archive), "r+") as f:
            del f["opensees/computed_sections"]
    replay = OpenSeesModel.from_h5(str(archive)).build("tcl", flat=True)
    assert isinstance(replay, str)
    return (_section_lines(deck.read_text(encoding="utf-8")),
            _section_lines(replay))


@pytest.mark.parametrize("case", sorted(_REPLAY_CASES))
def test_replay_writes_the_bridge_section_line(
        g, tmp_path: Path, case: str) -> None:
    """The oracle is the bridge's own deck: the replay from the archive
    writes the same ``section`` line, literal for literal, and warns
    no ``ReplaySlotTypeWarning``."""
    from apeGmsh.opensees._internal.replay_types import ReplaySlotTypeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", ReplaySlotTypeWarning)
        bridge, replay = _bridge_and_replay(g, tmp_path, case)
    assert len(bridge) == 1
    tokens = bridge[0].split()
    for at, literal in _REPLAY_CASES[case][2].items():
        assert tokens[at] == literal
    assert replay == bridge


def test_a_missing_sidecar_row_warns_and_recovers_by_value(
        g, tmp_path: Path) -> None:
    """Fail closed: without its ``/opensees/computed_sections`` row a
    computed section's moduli are unknown, so the replay warns and keeps
    the by-value recovery (an integral float replays as an int)."""
    from apeGmsh.opensees._internal.replay_types import ReplaySlotTypeWarning

    with pytest.warns(ReplaySlotTypeWarning,
                      match="0 /opensees/computed_sections rows"):
        bridge, replay = _bridge_and_replay(
            g, tmp_path, "elastic-float-E", drop_sidecar=True)
    assert bridge[0].split()[3] == "200000.0"
    assert replay[0].split()[3] == "200000"


def _computed_table(path: Path) -> "tuple[Any, int]":
    from apeGmsh.opensees.emitter import h5_reader

    with h5_reader.open(str(path)) as m:
        table = m.declarations()
    assert table is not None
    row = next(i for i, ent in table.params.items()
               if ent[0].type == "ComputedSection")
    return table, row


@pytest.mark.parametrize("field, value, match", [
    ("analysis", "DeclOpaque:Frobnicator", "not a SectionProperties"),
    ("kind", "plastic", "kind 'plastic' is unknown"),
])
def test_an_unexpected_computed_section_field_still_warns(
        g, tmp_path: Path, field: str, value: str, match: str) -> None:
    """Fail closed: the stand-in covers exactly a ``SectionProperties``
    archived by name and the two lowerings; anything else warns and
    keeps the by-value recovery."""
    import dataclasses
    from types import MappingProxyType

    from apeGmsh.opensees._internal.replay_types import (
        ReplaySlotTypeWarning, SlotTypes,
    )
    from apeGmsh.opensees.emitter.h5_reader import DeclOpaque

    sec = _analyzer(g)
    archive = tmp_path / "model.h5"
    _frame(lambda ops: ops.section.ComputedSection(analysis=sec)).h5(
        str(archive))
    table, row = _computed_table(archive)
    (entry,) = table.params[row]
    held: object = (DeclOpaque(type=value.split(":", 1)[1])
                    if value.startswith("DeclOpaque:") else value)
    bad = dataclasses.replace(
        entry, params=MappingProxyType({**entry.params, field: held}))
    params = dict(table.params)
    params[row] = (bad,)
    st = SlotTypes(dataclasses.replace(table, params=MappingProxyType(params)),
                   computed=_computed_rows(archive))
    with pytest.warns(ReplaySlotTypeWarning, match=match):
        out = st.recover("section", 1, "Elastic", (200000, 8.5))
    assert out == (200000, 8.5)
    assert type(out[0]) is int


def _computed_rows(path: Path) -> list[tuple[int, str]]:
    return [(tag, payload) for tag, _name, payload
            in read_computed_sections(str(path))]


@pytest.mark.parametrize("fault, match", [
    ("duplicate", "2 /opensees/computed_sections rows"),
    ("kind", "not a 'elastic' lowering"),
    ("E_ref", "E_ref='x', not a number"),
    ("G_ref", "G_ref=True, not a number"),
])
def test_a_faulty_sidecar_row_warns(
        g, tmp_path: Path, fault: str, match: str) -> None:
    """Fail closed: the moduli come from exactly one well-formed row of
    the section's tag; any other row set warns and keeps the by-value
    recovery."""
    from apeGmsh.opensees._internal.replay_types import (
        ReplaySlotTypeWarning, SlotTypes,
    )

    sec = _analyzer(g)
    archive = tmp_path / "model.h5"
    _frame(lambda ops: ops.section.ComputedSection(analysis=sec)).h5(
        str(archive))
    table, _row = _computed_table(archive)
    ((tag, payload),) = _computed_rows(archive)
    if fault == "duplicate":
        rows = [(tag, payload), (tag, payload)]
    else:
        blob = json.loads(payload)
        blob[fault] = {"kind": "fiber", "E_ref": "x", "G_ref": True}[fault]
        rows = [(tag, json.dumps(blob))]
    st = SlotTypes(table, computed=rows)
    with pytest.warns(ReplaySlotTypeWarning, match=match):
        out = st.recover("section", tag, "Elastic", (200000, 8.5))
    assert out == (200000, 8.5)


def _pickled(obj: object) -> bytes:
    import pickle

    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _computed(sec: SectionProperties) -> SectionProperties:
    """Fill every lazy cache: geometric, warping, plastic, stress."""
    sec.analyze()
    sec.stress(N=1.0)
    return sec


def test_section_properties_pickle_round_trip(g) -> None:
    """``SectionProperties`` pickles (#2567): the read-only material view
    is thawed for the pickle and refrozen on load, with every material
    and the analysis numbers intact."""
    import pickle
    from types import MappingProxyType

    sec = _analyzer(g, material=SectionMaterial(E=200e3, nu=0.3, G=80e3))
    back = pickle.loads(_pickled(sec))
    assert type(back._materials) is MappingProxyType
    assert dict(back.materials) == dict(sec.materials)
    assert back.name == "PG24"
    assert back.warping().GJ == pytest.approx(sec.warping().GJ)
    assert back.geometric().area == pytest.approx(sec.geometric().area)
    # the source keeps its own frozen view
    assert type(sec._materials) is MappingProxyType


def test_an_unpickled_analyzer_recomputes_lazily(g) -> None:
    """The pickle carries the declaration, not the caches: an analyzer
    unpickled after every analysis ran starts with none cached, and
    recomputes each one to the same numbers."""
    import pickle

    sec = _computed(_analyzer(g))
    back = pickle.loads(_pickled(sec))
    assert back._geometric is None and back._warping is None
    assert back._plastic is None and back._unit_fields is None
    assert back._warp_solutions == ()
    assert back.geometric().area == pytest.approx(sec.geometric().area)
    assert back.warping().GJ == pytest.approx(sec.warping().GJ)
    assert back.plastic().Mp_xx == pytest.approx(sec.plastic().Mp_xx)
    assert back.stress(N=1.0).sigma_zz == pytest.approx(
        sec.stress(N=1.0).sigma_zz)


def test_equal_analyzers_pickle_equal_whatever_was_computed(g) -> None:
    """Two analyzers of one mesh and one material pickle to the same bytes
    when only one of them has run its analyses: a cache is not part of
    the declaration."""
    g.sections.rect_face(2.0, 4.0, label="bar")
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=2)
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    mat = SectionMaterial(E=200e3, nu=0.3, fy=345.0)
    fresh = SectionProperties(fem, materials={"bar": mat}, name="PG24")
    used = _computed(
        SectionProperties(fem, materials={"bar": mat}, name="PG24"))
    assert _pickled(used) == _pickled(fresh)


@pytest.mark.parametrize("query", ["stress", "plastic"])
def test_a_read_only_query_keeps_the_declaration_digest(
        g, tmp_path: Path, query: str) -> None:
    """A read-only ``stress()`` or ``plastic()`` between two emits is no
    new declaration: the P5 digest (and so the automatic write, the
    lineage memo and ``ops.results``) does not see it."""
    from apeGmsh.opensees._internal.artifact_write import lineage_key

    sec = _analyzer(g)
    ops = _frame(lambda o: o.section.ComputedSection(analysis=sec))
    ops.tcl(str(tmp_path / "bridge.tcl"), progress=False, flat=True)
    before = lineage_key(ops)
    assert before is not None
    if query == "stress":
        sec.stress(N=1.0)
    else:
        sec.plastic()
    assert lineage_key(ops) == before


def test_reader_tolerates_absence(tmp_path: Path) -> None:
    """Pre-2.20.0-shaped files (no sidecar) read as empty — from_h5
    surfaces an empty tuple, never an error."""
    p = tmp_path / "model.h5"
    _frame(
        lambda ops: ops.section.Elastic(
            E=200e3, A=8.0, Iz=10.7, Iy=2.7, G=76.9e3, J=7.0,
            alphaY=0.83, alphaZ=0.83,
        )
    ).h5(str(p))
    om = OpenSeesModel.from_h5(str(p))
    assert om.computed_sections() == ()
