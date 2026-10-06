"""K19 (#1506): proof of the replacements that exist for ``ModelData``.

The inventory ``internal_docs/program/x1_modeldata_inventory.md`` lists
nine ``ModelData`` capabilities.  Two have a bridge-free replacement
today; this module proves each one gives the same observable result,
so the inventory can name a test for it.  The other rows have none and
are ledgered under "Gaps (kept)" there.

Oracles, each naming the right answer:

* **Row 2, the read accessors.**  On an archive, ``ModelData.from_h5(p)``
  and ``OpenSeesModel.from_h5(p)`` report the same ``ndm`` / ``ndf``
  (both read ``/meta`` through ``read_spatial_ndm``) and bind the same
  neutral zone (node ids, coordinates, element ids).  The right answer
  is the ``ndm`` / ``ndf`` the file was written with.
* **Row 8, the H5-to-H5 round trip.**  On a ``ModelData``-written file,
  ``OpenSeesModel.from_h5(p).to_h5(q)`` writes the same file, object by
  object and attribute by attribute (``/meta/created_iso`` aside, which
  every writer re-stamps), as ``ModelData.from_h5(p).write(q)``, and
  both equal the source.  On an ``apeSees``-written file the replacement
  still equals the source, while ``ModelData`` keeps only the
  orientation zone (ADR 0018 INV-5); that difference also proves the
  comparator is not vacuous.
* **Row 8, the staged case (ADR 0055).**  ``ModelData.from_h5`` warns on
  a staged archive because its ``write`` would drop ``/opensees/stages``.
  The replacement round trip raises no warning and keeps the stages
  zone equal to the source.

Not covered, because nothing replaces it: enriching a loaded archive
with orientation (``from_h5`` then ``oriented_elements``).  That is a
gap, not a cut.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import ModelData, OpenSeesModel

from tests.opensees.h5._opensees_model_fixtures import (
    build_simple_frame_fem,
    build_simple_frame_h5,
)
from tests.opensees.h5.test_h5_stages_reader import _real_two_stage_bridge
from tests.opensees.h5.test_h5_stages_writer import _collect_zone, _norm

#: The one ``/meta`` attribute every writer re-stamps (``OpenSeesModel.to_h5``
#: docstring: byte-equivalent "modulo ``/meta/created_iso``").
_RESTAMPED = "created_iso"


def _tree(path: Path, sub: str = "/") -> dict[str, Any]:
    """Every object under ``sub`` with its dtype, values and attributes."""
    with h5py.File(path, "r") as f:
        grp = f[sub]
        out = _collect_zone(grp)
        out["@root"] = {k: _norm(grp.attrs[k]) for k in sorted(grp.attrs)}
    meta = out.get("meta")
    if meta is not None:
        meta[1].pop(_RESTAMPED, None)
    return out


def _children(path: Path, sub: str) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(f[sub]) if sub in f else []


def _model_data_file(tmp_path: Path, *, ndm: int) -> Path:
    """A ``ModelData``-written archive: oriented in 3-D, bare in 2-D."""
    fem = build_simple_frame_fem(ndm=ndm)
    md = ModelData(fem, ndm=ndm, ndf=6 if ndm == 3 else 3, model_name="k19")
    if ndm == 3:
        md.oriented_elements(
            pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
        )
    out = tmp_path / f"model_data_{ndm}d.h5"
    md.write(str(out))
    return out


# ---------------------------------------------------------------------------
# Row 2: fem / ndm / ndf on a loaded archive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndm, ndf", [(2, 3), (3, 6)])
def test_opensees_model_reports_the_accessors_model_data_reports(
    tmp_path: Path, ndm: int, ndf: int,
) -> None:
    src = _model_data_file(tmp_path, ndm=ndm)
    md = ModelData.from_h5(str(src))
    om = OpenSeesModel.from_h5(str(src))

    assert (om.ndm, om.ndf) == (md.ndm, md.ndf) == (ndm, ndf)
    np.testing.assert_array_equal(om.fem.nodes.ids, md.fem.nodes.ids)
    np.testing.assert_array_equal(om.fem.nodes.coords, md.fem.nodes.coords)
    np.testing.assert_array_equal(om.fem.elements.ids, md.fem.elements.ids)


# ---------------------------------------------------------------------------
# Row 8: the H5-to-H5 round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndm", [2, 3])
def test_round_trip_matches_model_data_on_a_model_data_file(
    tmp_path: Path, ndm: int,
) -> None:
    src = _model_data_file(tmp_path, ndm=ndm)
    by_md, by_om = tmp_path / "by_md.h5", tmp_path / "by_om.h5"
    ModelData.from_h5(str(src)).write(str(by_md))
    OpenSeesModel.from_h5(str(src)).to_h5(str(by_om))

    assert _tree(by_om) == _tree(by_md) == _tree(src)
    if ndm == 3:
        # The orientation zone the viewer joins is really there.
        assert _children(src, "opensees/transforms")


def test_round_trip_keeps_the_bridge_deck_model_data_drops(
    tmp_path: Path,
) -> None:
    src, _ = build_simple_frame_h5(tmp_path)
    by_md, by_om = tmp_path / "by_md.h5", tmp_path / "by_om.h5"
    ModelData.from_h5(str(src)).write(str(by_md))
    OpenSeesModel.from_h5(str(src)).to_h5(str(by_om))

    assert _tree(by_om) == _tree(src)
    # Same orientation zone either way ...
    for sub in ("opensees/transforms", "opensees/element_meta"):
        assert _tree(by_md, sub) == _tree(by_om, sub)
    # ... but ModelData writes no materials (ADR 0018 INV-5), so the
    # comparator above does see a difference when one exists.
    assert "materials" in _tree(src, "opensees")
    assert "materials" not in _tree(by_md, "opensees")


def test_round_trip_keeps_a_staged_archive_without_warning(
    tmp_path: Path,
) -> None:
    src = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(src))
    with pytest.warns(UserWarning, match="STAGED"):
        ModelData.from_h5(str(src))

    by_om = tmp_path / "by_om.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        OpenSeesModel.from_h5(str(src)).to_h5(str(by_om))

    assert _children(src, "opensees/stages") == ["stage_000", "stage_001"]
    assert _children(by_om, "opensees/stages") == ["stage_000", "stage_001"]
    assert _tree(by_om, "opensees/stages") == _tree(src, "opensees/stages")
    assert _tree(by_om) == _tree(src)
