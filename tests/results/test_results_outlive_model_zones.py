"""ADR 0113 INV-11 at the facade: a results file outlives its embedded
model zones (D9).

The reader half lives in ``tests/opensees/h5/test_h5_schema_compat.py``
(``NativeReader.unavailable_zones``, the per-zone warning, the newer-zone
refusal). This module holds what the user reaches: ``Results.from_native``
on a file whose embedded ``/model`` (neutral) or ``/opensees`` is below
its floor, with and without ``fem=``, and the positive half: a file
stamped at both floors opens and ``Results.model`` resolves.

Warn-as-contract: run with ``-W error::UserWarning``. Every open of a
flagged file is wrapped in ``pytest.warns`` and asserts exactly one
record; the floor-stamped open runs under ``simplefilter("error")``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees._internal.schema_version import (
    NEUTRAL,
    OPENSEES,
    SchemaVersionError,
)
from apeGmsh.results import Results
from tests.fixtures.schema import NEUTRAL_FLOOR, OPENSEES_FLOOR, RESULTS_FLOOR
from tests.opensees.h5._opensees_model_fixtures import (
    build_simple_frame_fem,
    build_simple_frame_h5,
)
from tests.opensees.h5.test_h5_schema_compat import (
    _below,
    _build_composed_results,
    _flagged_results,
    _restamp_results,
)

_FLOOR = {NEUTRAL: NEUTRAL_FLOOR, OPENSEES: OPENSEES_FLOOR}
_ZONES = [NEUTRAL, OPENSEES]


@pytest.fixture
def sidecar(tmp_path: Path):
    """An inside-floor ``model.h5`` from a different directory than the
    results file, plus its FEMData: what a user supplies as ``model=`` /
    ``fem=`` when the results file's own zones cannot be read."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    side = tmp_path / "sidecar"
    side.mkdir()
    model_path, fem = build_simple_frame_h5(side)
    return OpenSeesModel.from_h5(model_path), fem


def _flagged(tmp_path: Path, zone: str) -> Path:
    flagged = tmp_path / "flagged"
    flagged.mkdir()
    return _flagged_results(flagged, zone, _below(_FLOOR[zone]))


def _open_flagged(path: Path, zone: str, **kw) -> Results:
    """Open through the facade, asserting exactly one D9 warning that
    names the zone and the file."""
    with pytest.warns(UserWarning, match=f"{zone}_schema_version") as rec:
        results = Results.from_native(path, **kw)
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "too old" in msg and str(path) in msg and "ADR 0113 D9" in msg
    return results


def _assert_stages_read(results: Results) -> None:
    (stage,) = results.stages
    assert stage.kind == "static" and stage.n_steps == 1
    slab = results.nodes.get(component="displacement_z")
    assert slab.values.shape == (1, 2)


@pytest.mark.parametrize("zone", _ZONES)
def test_model_from_the_flagged_file_itself_refuses(
    tmp_path: Path, zone: str,
) -> None:
    """D9 "Results.model is unavailable": the Composed-file route,
    ``OpenSeesModel.from_h5(results_path)``, refuses the flagged file and
    names the zone, so ``model=`` must come from a sidecar archive."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    path = _flagged(tmp_path, zone)
    with pytest.raises(SchemaVersionError, match=f"{zone}_schema_version"):
        OpenSeesModel.from_h5(path)


@pytest.mark.parametrize("zone", _ZONES)
def test_flagged_zone_opens_with_supplied_fem(
    tmp_path: Path, zone: str, sidecar,
) -> None:
    """``fem=`` supplied: the embedded ``/model`` is never read, so the
    file opens for both zones, ``/stages`` read and the supplied
    snapshot is the bound one (the review's blocking case)."""
    model, fem = sidecar
    path = _flagged(tmp_path, zone)
    with _open_flagged(path, zone, model=model, fem=fem) as results:
        _assert_stages_read(results)
        assert results.fem is fem
        assert results.model is model
        assert results.stage("s").fem is fem


def test_flagged_neutral_without_fem_opens_and_fem_refuses(
    tmp_path: Path, sidecar,
) -> None:
    """No ``fem=`` and the file's own ``/model`` is below its floor: the
    file opens, ``/stages`` read, and ``Results.fem`` raises the reader's
    refusal with the ``fem=`` hint instead of reading the zone. The
    refusal follows stage scoping; ``bind(fem)`` clears it."""
    model, fem = sidecar
    path = _flagged(tmp_path, NEUTRAL)
    with _open_flagged(path, NEUTRAL, model=model) as results:
        _assert_stages_read(results)
        assert results.model is model
        with pytest.raises(SchemaVersionError, match="neutral_schema_version") as exc:
            results.fem
        text = str(exc.value)
        assert "too old" in text and "ADR 0113 D9" in text and "fem=" in text
        with pytest.raises(SchemaVersionError, match="neutral_schema_version"):
            results.stage("s").fem
        # Display still works: repr / str / the summary read the stored
        # fields, not the raising property, and say why the FEM is absent.
        for shown in (repr(results), str(results), results.inspect.summary()):
            assert "FEM: unavailable" in shown and "fem=" in shown
            assert "stage_0" in shown
        # A composite that needs the FEM names the zone below its floor,
        # not a bare "requires a bound FEMData".
        with pytest.raises(SchemaVersionError, match="neutral_schema_version") as exc:
            results.nodes.nearest_to((0.0, 0.0, 0.0), component="displacement_z")
        assert "fem=" in str(exc.value)
        bound = results.bind(fem)
        assert bound.fem is fem
        assert bound.stage("s").fem is fem
        _assert_stages_read(bound)


def test_flagged_opensees_without_fem_opens_and_fem_resolves(
    tmp_path: Path, sidecar,
) -> None:
    """Only ``/opensees`` is below its floor: the neutral zone is inside
    its floor, so the embedded FEMData still binds; ``model=`` is the
    sidecar's, since ``from_native`` never reads ``/opensees``."""
    model, _ = sidecar
    path = _flagged(tmp_path, OPENSEES)
    with _open_flagged(path, OPENSEES, model=model) as results:
        _assert_stages_read(results)
        embedded = results.fem
        assert embedded is not None
        assert list(embedded.nodes.ids) == list(build_simple_frame_fem().nodes.ids)
        assert results.model is model


def test_floor_stamped_file_opens_and_model_resolves(tmp_path: Path) -> None:
    """INV-11, positive half: a results file whose embedded zones carry
    their floor stamps opens silently through the facade, and
    ``Results.model`` resolves from the file itself (the Composed-file
    route) with its FEMData reachable both ways."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    path, _ = _build_composed_results(tmp_path)
    _restamp_results(
        path, neutral=NEUTRAL_FLOOR, opensees=OPENSEES_FLOOR,
        results=RESULTS_FLOOR,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = OpenSeesModel.from_h5(path)
        results = Results.from_native(path, model=model)
    with results:
        assert results.model is model
        _assert_stages_read(results)
        fem = results.fem
        assert fem is not None
        assert list(fem.nodes.ids) == list(results.model.fem.nodes.ids)
