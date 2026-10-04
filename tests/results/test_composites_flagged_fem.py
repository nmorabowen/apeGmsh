"""ADR 0113 D9 at the composite seams (#1371).

A native results file whose embedded ``/model`` is below its floor opens
without ``fem=``; every composite that needs the FEM must then raise the
refusal that names the zone, not a bare "Pass fem=" ``RuntimeError``.
``test_results_outlive_model_zones`` covers ``nodes.nearest_to``; this
module covers the ``pg=`` selector resolvers (nodes and elements) and the
``plot.vector_glyph`` gate.

Warn-as-contract: run with ``-W error::UserWarning``; the one D9 warning
per open is consumed by ``_open_flagged``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.opensees._internal.schema_version import NEUTRAL, SchemaVersionError
from tests.results.test_results_outlive_model_zones import (
    _flagged,
    _open_flagged,
    sidecar,  # noqa: F401  (fixture)
)


def _assert_names_the_zone(exc: pytest.ExceptionInfo) -> None:
    text = str(exc.value)
    assert "neutral_schema_version" in text
    assert "too old" in text and "ADR 0113 D9" in text and "fem=" in text


def test_pg_selector_names_the_zone_below_its_floor(
    tmp_path: Path, sidecar,  # noqa: F811
) -> None:
    """``pg=`` on nodes and elements resolves through ``_require_fem``:
    the refusal names the neutral zone instead of "Pass fem="."""
    model, fem = sidecar
    path = _flagged(tmp_path, NEUTRAL)
    with _open_flagged(path, NEUTRAL, model=model) as results:
        with pytest.raises(SchemaVersionError) as exc:
            results.nodes.get(component="displacement_z", pg="Cols")
        _assert_names_the_zone(exc)
        with pytest.raises(SchemaVersionError) as exc:
            results.elements._resolve_element_ids(
                pg="Cols", label=None, selection=None, ids=None,
            )
        _assert_names_the_zone(exc)
        # Binding the FEM clears the refusal: the same selector resolves.
        slab = results.bind(fem).nodes.get(component="displacement_z", pg="Cols")
        assert sorted(int(i) for i in slab.node_ids) == [1, 2]


def test_vector_glyph_gate_names_the_zone_below_its_floor(
    tmp_path: Path, sidecar,  # noqa: F811
) -> None:
    """``plot.vector_glyph`` needs node coordinates; on the flagged file
    its gate raises the zone-naming refusal (matplotlib Agg, no window)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, _ = sidecar
    path = _flagged(tmp_path, NEUTRAL)
    try:
        with _open_flagged(path, NEUTRAL, model=model) as results:
            with pytest.raises(SchemaVersionError) as exc:
                results.plot.vector_glyph("displacement", with_mesh=False)
            _assert_names_the_zone(exc)
    finally:
        plt.close("all")
