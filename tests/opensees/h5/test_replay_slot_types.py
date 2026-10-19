"""K R1b-S2 (#2194): replay keeps each slot's int/float type.

The bridge archives positional arguments as ``float64``, and the readers
used to recover every integral float as an ``int``: ``thickness=1.0``
replayed as ``1``. :class:`~apeGmsh.opensees._internal.replay_types.
SlotTypes` rebuilds each declaration from ``/opensees/decl_params``, dry
runs its ``_emit`` and takes the int/float type of each slot. Three read
sites use it, one test class each:

- ``H5Model._read_param_array`` (materials, sections, beam integration,
  time series, damping);
- ``OpenSeesModel._load_elements`` (the element argstack);
- the stage regions of ``H5Model.stages``, whose ``-rayleigh`` tail is
  floats by construction, as the flat regions' already was.

The oracle is the bridge itself: each typed value is the one the bridge's
own deck writes, and each test reads it back from the bridge's archive.
An archive without ``decl_params`` keeps the by-value recovery, and a
declaration this build cannot rebuild warns and keeps it too.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.replay_types import (
    ReplaySlotTypeWarning, SlotTypes, primitive_classes,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.time_series import Linear as LinearSeries
from apeGmsh.opensees.transform import Linear as LinearTransf

from tests.opensees.h5._opensees_model_fixtures import build_two_quad_fem
from tests.opensees.h5.test_h5_apesees_cuts import _build_minimal_ops
from tests.opensees.h5.test_h5_damping import _build_with_element_damp
from tests.opensees.h5.test_h5_stages_writer import (
    _build_kitchen_sink_bridge,
)


def _quad(thickness: float) -> apeSees:
    """Two quads of ``thickness`` on a real two-quad FEMData."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=thickness, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=thickness, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    return ops


def _archive(ops: apeSees, tmp_path: Path, name: str = "model.h5") -> Path:
    path = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(path))
    return path


def _lines(deck: str, prefix: str) -> list[str]:
    return [ln for ln in deck.splitlines() if ln.startswith(prefix)]


def _decks(ops: apeSees, tmp_path: Path) -> tuple[str, str]:
    """The bridge's Tcl deck and the one replayed from its archive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), progress=False, flat=True)
        archive = tmp_path / "model.h5"
        ops.h5(str(archive))
        replay = OpenSeesModel.from_h5(str(archive)).build("tcl", flat=True)
    assert isinstance(replay, str)
    return deck.read_text(encoding="utf-8"), replay


def _types(values: "tuple[object, ...]") -> tuple[str, ...]:
    return tuple(type(v).__name__ for v in values)


# ---------------------------------------------------------------------------
# Site 2: the element argstack (OpenSeesModel._load_elements)
# ---------------------------------------------------------------------------


class TestElementSlots:
    def test_quad_thickness_one_stays_float(self, tmp_path: Path) -> None:
        """The regression: ``thickness=1.0`` replays as ``1.0``, not ``1``,
        and the material tag after it stays an int."""
        path = _archive(_quad(1.0), tmp_path)
        model = OpenSeesModel.from_h5(str(path))
        quads = [e for e in model.elements() if e.type_token == "quad"]
        assert len(quads) == 2
        for rec in quads:
            tail = rec.args[len(rec.args) - 3:]
            assert tail == (1.0, "PlaneStrain", 1)
            assert _types(tail) == ("float", "str", "int")

    def test_quad_deck_line_equals_the_bridge(self, tmp_path: Path) -> None:
        bridge, replay = _decks(_quad(1.0), tmp_path)
        want = _lines(bridge, "element quad ")
        assert want and all(" 1.0 PlaneStrain 1" in ln for ln in want)
        assert _lines(replay, "element quad ") == want

    def test_an_int_thickness_stays_int(self, tmp_path: Path) -> None:
        """The type is the declaration's, not a blanket float: a thickness
        passed as ``2`` emits ``2`` on the bridge and on replay."""
        bridge, replay = _decks(_quad(2), tmp_path)
        want = _lines(bridge, "element quad ")
        assert want and all(" 2 PlaneStrain 1" in ln for ln in want)
        assert _lines(replay, "element quad ") == want

    def test_a_row_without_its_fem_element_raises(
            self, tmp_path: Path) -> None:
        """The dry run needs the row's node count; a fanned-out row whose
        FEM element the neutral zone lacks fails closed, naming both."""
        path = _archive(_quad(1.0), tmp_path)
        with h5_reader.open(str(path)) as m:
            with pytest.raises(h5_reader.MalformedH5Error,
                               match=r"element \d+ is fanned out from FEM "
                                     "element 1"):
                OpenSeesModel._load_elements(m, SimpleNamespace(elements=()))

    def test_beam_moduli_stay_float(self, tmp_path: Path) -> None:
        """``elasticBeamColumn`` ``E=200e9`` / ``G=80e9``: floats; the
        transform and ``-damp`` tags: ints."""
        bridge, replay = _decks(_build_with_element_damp(), tmp_path)
        want = _lines(bridge, "element elasticBeamColumn ")
        assert want and "200000000000.0 80000000000.0" in want[0]
        assert _lines(replay, "element elasticBeamColumn ") == want


# ---------------------------------------------------------------------------
# Site 1: the tagged stores (H5Model._read_param_array)
# ---------------------------------------------------------------------------


class TestStoreSlots:
    def test_nd_material_moduli_stay_float(self, tmp_path: Path) -> None:
        path = _archive(_quad(1.0), tmp_path)
        with h5_reader.open(str(path)) as m:
            (rec,) = m.materials_by_family()["nd"]
        assert rec.params == (1000000.0, 0.3, 0.0)
        assert _types(rec.params) == ("float", "float", "float")

    def test_steel02_and_fiber_gj_stay_float(self, tmp_path: Path) -> None:
        path = _archive(_build_minimal_ops(), tmp_path)
        with h5_reader.open(str(path)) as m:
            (steel,) = m.materials_by_family()["uniaxial"]
            (fiber,) = m.sections()
        assert steel.type_token == "Steel02"
        assert steel.params[:3] == (420e6, 200e9, 0.01)
        assert all(isinstance(v, float) for v in steel.params)
        assert fiber.params == ("-GJ", 1.0e6)
        assert _types(fiber.params) == ("str", "float")

    def test_damping_and_series_slots(self, tmp_path: Path) -> None:
        """``damping Uniform 1 0.03 0.5 10.0``: the bounds are floats."""
        path = _archive(_build_with_element_damp(), tmp_path)
        with h5_reader.open(str(path)) as m:
            (damp,) = m.dampings()
        assert damp.args == (0.03, 0.5, 10.0)
        assert _types(damp.args) == ("float", "float", "float")
        bridge, replay = _decks(_build_with_element_damp(), tmp_path)
        assert _lines(replay, "damping ") == _lines(bridge, "damping ")


# ---------------------------------------------------------------------------
# Site 3: the stage regions (H5Model.stages)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::apeGmsh.opensees.InertStaticDtWarning")
class TestStageRegionSlots:
    def test_stage_rayleigh_tail_is_float(self, tmp_path: Path) -> None:
        """``s.damping.rayleigh(alpha_m=0.1, on="Fill")`` writes
        ``-rayleigh 0.1 0.0 0.0 0.0``; the stage region reads it back
        as four floats, as the flat regions do."""
        path = _archive(_build_kitchen_sink_bridge(), tmp_path)
        with h5_reader.open(str(path)) as m:
            (stage,) = m.stages()
        (region,) = stage.regions
        at = region.args.index("-rayleigh")
        tail = region.args[at + 1:at + 5]
        assert tail == (0.1, 0.0, 0.0, 0.0)
        assert _types(tail) == ("float",) * 4

    def test_a_short_stage_rayleigh_tail_raises(self, tmp_path: Path) -> None:
        path = _archive(_build_kitchen_sink_bridge(), tmp_path)
        with h5py.File(str(path), "r+") as f:
            rgrp = f["opensees/stages/stage_000/regions"]
            (name,) = list(rgrp)
            g = rgrp[name]
            strs = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in g.attrs["params_str"]]
            at = strs.index("-rayleigh")
            nums = g.attrs["params"][:at + 3]
            g.attrs["params"] = nums
            g.attrs["params_str"] = strs[:at + 3]
        with h5_reader.open(str(path)) as m:
            with pytest.raises(h5_reader.MalformedH5Error,
                               match="stage_000/regions/.*-rayleigh"):
                m.stages()


# ---------------------------------------------------------------------------
# Seams: legacy archives, unknown classes, mismatches, the class registry
# ---------------------------------------------------------------------------


def _strip_decl_params(path: Path) -> None:
    """Make ``path`` the archive a writer below 2.26.0 left: no
    ``/opensees/decl_params`` (hash-excluded, so the lineage holds)."""
    with h5py.File(str(path), "r+") as f:
        del f["opensees/decl_params"]


class TestSeams:
    def test_a_legacy_archive_replays_as_before(self, tmp_path: Path) -> None:
        """No ``decl_params``: nothing is recovered, and every integral
        float replays as an int, exactly as before #2194."""
        path = _archive(_quad(1.0), tmp_path)
        _strip_decl_params(path)
        with h5_reader.open(str(path)) as m:
            assert not m.slot_types().active
            (rec,) = m.materials_by_family()["nd"]
        assert rec.params == (1000000, 0.3, 0)
        assert _types(rec.params) == ("int", "float", "int")
        model = OpenSeesModel.from_h5(str(path))
        quad = next(e for e in model.elements() if e.type_token == "quad")
        assert quad.args[len(quad.args) - 3:] == (1, "PlaneStrain", 1)
        replay = model.build("tcl", flat=True)
        assert isinstance(replay, str)
        assert all(" 1 PlaneStrain 1" in ln
                   for ln in _lines(replay, "element quad "))

    def test_no_declarations_recover_nothing(self) -> None:
        st = SlotTypes(None)
        assert not st.active
        assert st.recover("nDMaterial", 1, "ElasticIsotropic", (1, 2)) == (1, 2)

    def _table(self, path: Path) -> h5_reader.DeclarationTable:
        with h5_reader.open(str(path)) as m:
            table = m.declarations()
        assert table is not None
        return table

    def _retyped(
        self, table: h5_reader.DeclarationTable, family: str, new_type: str,
    ) -> h5_reader.DeclarationTable:
        row = next(i for i, d in enumerate(table.decls) if d.family == family)
        (entry,) = table.params[row]
        params = dict(table.params)
        params[row] = (dataclasses.replace(entry, type=new_type),)
        return dataclasses.replace(table, params=MappingProxyType(params))

    def test_an_unknown_class_warns_and_keeps_the_values(
            self, tmp_path: Path) -> None:
        table = self._table(_archive(_quad(1.0), tmp_path))
        st = SlotTypes(self._retyped(table, "nDMaterial", "Frobnicator"))
        with pytest.warns(ReplaySlotTypeWarning,
                          match="Frobnicator.*no nDMaterial class"):
            out = st.recover(
                "nDMaterial", 1, "ElasticIsotropic", (1000000, 0.3, 0))
        assert out == (1000000, 0.3, 0)
        assert _types(out) == ("int", "float", "int")

    def test_a_failed_dry_run_warns(self, tmp_path: Path) -> None:
        """A declaration whose fields its class refuses (here, a thickness
        of zero): the rebuild raises, so the replay warns and keeps the
        by-value types."""
        table = self._table(_archive(_quad(1.0), tmp_path))
        row = next(i for i, d in enumerate(table.decls)
                   if d.family == "element")
        (entry,) = table.params[row]
        bad = dataclasses.replace(
            entry, params=MappingProxyType({**entry.params, "thickness": 0.0}))
        params = dict(table.params)
        params[row] = (bad,)
        st = SlotTypes(dataclasses.replace(
            table, params=MappingProxyType(params)))
        first = next(f for k, f, _c, d in table.tags
                     if k == "element" and d == row)
        with pytest.warns(ReplaySlotTypeWarning, match="ValueError.*thickness"):
            out = st.recover("element", first, "quad", (1, "PlaneStrain", 1),
                             nodes=4)
        assert out == (1, "PlaneStrain", 1)

    def test_a_length_mismatch_changes_nothing(self, tmp_path: Path) -> None:
        table = self._table(_archive(_quad(1.0), tmp_path))
        st = SlotTypes(table)
        assert st.active
        with warnings.catch_warnings():
            warnings.simplefilter("error", ReplaySlotTypeWarning)
            got = st.recover(
                "nDMaterial", 1, "ElasticIsotropic", (1000000, 0.3, 0))
            longer = st.recover(
                "nDMaterial", 1, "ElasticIsotropic", (1000000, 0.3, 0, 7))
            untagged = st.recover("nDMaterial", 99, "ElasticIsotropic", (1,))
        assert _types(got) == ("float", "float", "float")
        assert longer == (1000000, 0.3, 0, 7)
        assert _types(longer) == ("int", "float", "int", "int")
        assert untagged == (1,)

    def test_the_registry_keys_classes_by_family(self) -> None:
        """``Linear`` is a time series and a transform: one name, two
        families, two classes."""
        classes = primitive_classes()
        assert classes["timeSeries"]["Linear"] is LinearSeries
        assert classes["geomTransf"]["Linear"] is LinearTransf
        assert "FourNodeQuad" in classes["element"]
