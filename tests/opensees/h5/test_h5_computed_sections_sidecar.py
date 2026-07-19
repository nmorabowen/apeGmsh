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
from pathlib import Path
from typing import cast

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


def _analyzer(g, *, name="PG24"):
    g.sections.rect_face(2.0, 4.0, label="bar")
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=2)
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    return SectionProperties(
        fem,
        materials={"bar": SectionMaterial(E=200e3, nu=0.3, fy=345.0)},
        name=name,
    )


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

def test_opensees_model_surfaces_and_round_trips(g, tmp_path: Path) -> None:
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
